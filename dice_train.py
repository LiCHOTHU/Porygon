"""DICE-RL-Robot-style fine-tuning for the LIBERO FM policy (single-process MVP).

Pipeline:
  1) load frozen BC FM cold-start (encoder + normalizer + teacher).
  2) build env_runner + single-env DiceCollector for online rollouts.
  3) probe the BC encoder once to discover (num_enc, hidden) state shape.
  4) instantiate DistilledRLModel (actor + critic ensemble + target critic).
  5) attach FMTeacher to the model for BC-MSE anchor inside actor_loss.
  6) main loop -- per outer iter:
        a) collect K episodes (student or teacher rollouts), push to replay
        b) N gradient steps (actor then critic then Polyak)
        c) eval at eval_interval via env_runner using the distilled student
        d) checkpoint at save_interval
"""

import os
import hydra
import torch
import wandb
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import imitation.utils.utils as utils
from imitation.algos.dice.distill_rl import DistilledRLModel
from imitation.algos.dice.teacher import FMTeacher
from imitation.algos.dice.replay_buffer import ReplayBuffer
from imitation.algos.dice.collector import DiceCollector


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _make_student_sample_actions(bc_policy, student_model):
    """Returns a sample_actions(data)->np closure to monkey-patch onto the BC policy so
    env_runner's existing get_action / _get_action_no_agg path drives the student."""
    @torch.no_grad()
    def sample_actions(data):
        was_training = bc_policy.training
        bc_policy.eval()
        data = bc_policy.preprocess_input(data, train_mode=False)
        cond = bc_policy.get_cond(data)
        B = cond.shape[0]
        noise = torch.randn(B, bc_policy.chunk_size, bc_policy.network_action_dim,
                            device=cond.device, dtype=cond.dtype)
        a = student_model.get_action(cond, noise)
        if was_training:
            bc_policy.train()
        return a.to(torch.float32).cpu().numpy()
    return sample_actions


@hydra.main(config_path="config", config_name="dice_train", version_base=None)
def main(cfg):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    save_dir, experiment_name = utils.get_experiment_dir(cfg, allow_overlap=True)
    os.makedirs(save_dir, exist_ok=True)
    logger = utils.setup_logger("dice_train.log", save_dir)
    logger.info(f"DICE fine-tuning experiment: {experiment_name}")
    logger.info(f"Output dir: {save_dir}")

    # --- frozen BC FM (encoder + normalizer + teacher) ---
    bc_policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(device)
    state_dict = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=logger)
    utils.soft_load_state_dict(bc_policy, state_dict["model"])
    norm_stats = state_dict["norm_stats"]
    bc_policy.normalizer.fit(norm_stats)
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad = False
    logger.info(f"Loaded frozen BC FM cold-start; K={bc_policy.num_inference_steps} chunk={bc_policy.chunk_size}")

    # --- env runner ---
    env_runner = instantiate(cfg.task.env_runner)
    task_indices = list(cfg.task_indices) if cfg.get("task_indices", None) else [cfg.task_index]
    task_names = [env_runner.env_names[t] for t in task_indices]
    multitask = len(task_indices) > 1
    logger.info(f"DICE tasks ({len(task_indices)}): indices={task_indices}")
    n_inits_by_task = {t: env_runner.benchmark.get_task_init_states(t).shape[0] for t in task_indices}

    # --- probe encoder once to learn (num_enc, hidden) ---
    task_emb = {k: v.repeat(1, 1) for k, v in env_runner.benchmark.get_task_emb(task_indices[0]).items()}
    probe_env = env_runner.env_factory(task_id=task_indices[0], benchmark=env_runner.benchmark)
    import imitation.envs.libero.wrappers as lw
    probe_env = lw.LiberoVectorWrapper(lambda: lw.LiberoFrameStack(probe_env, env_runner.frame_stack), 1)
    probe_obs, _ = probe_env.reset()
    probe_batch = bc_policy._make_batch({k: v for k, v in probe_obs.items()}, task_indices[0], **task_emb)
    probe_batch = bc_policy.preprocess_input(probe_batch, train_mode=False)
    with torch.no_grad():
        probe_cond = bc_policy.get_cond(probe_batch)
    num_enc, hidden = int(probe_cond.shape[1]), int(probe_cond.shape[2])
    logger.info(f"Probed encoder cond shape: (num_enc={num_enc}, hidden={hidden})")
    probe_env.close()

    # --- student model ---
    student = DistilledRLModel(
        state_dim=num_enc * hidden,
        action_dim=bc_policy.network_action_dim,
        horizon_steps=bc_policy.chunk_size,
        actor_hidden=list(cfg.dice.actor_hidden),
        critic_hidden=list(cfg.dice.critic_hidden),
        ensemble_size=cfg.dice.ensemble_size,
        q_depends_on_noise=cfg.dice.q_depends_on_noise,
        conservative=cfg.dice.conservative,
        lcb_kappa=cfg.dice.lcb_kappa,
        td_loss=cfg.dice.td_loss,
        bc_loss_weight=cfg.dice.bc_loss_weight,
        num_multi_z=cfg.dice.num_multi_z,
        clip_action=True,
        device=device,
    ).to(device)
    teacher = FMTeacher(bc_policy)
    student.attach_teacher(teacher)
    logger.info(f"Student actor params: {sum(p.numel() for p in student.actor.parameters())/1e6:.2f}M; "
                f"critic params: {sum(p.numel() for p in student.critic.parameters())/1e6:.2f}M")

    # --- optimizers ---
    actor_opt = torch.optim.AdamW(student.actor.parameters(), lr=cfg.dice.actor_lr)
    critic_opt = torch.optim.AdamW(student.critic.parameters(), lr=cfg.dice.critic_lr)

    # --- replay + collector ---
    replay = ReplayBuffer(max_size=cfg.dice.replay_size,
                          cond_shape=(num_enc, hidden),
                          horizon=bc_policy.chunk_size,
                          action_dim=bc_policy.network_action_dim,
                          device=device)
    collector = DiceCollector(env_runner, bc_policy, student, device,
                              max_episode_length=cfg.dice.max_episode_length,
                              use_teacher_for_collect=False)

    # --- wandb ---
    try:
        wandb.init(project=cfg.logging.project, group=cfg.logging.group,
                   name=experiment_name, config=OmegaConf.to_container(cfg, resolve=True),
                   mode=cfg.logging.mode)
    except Exception as e:
        logger.warning(f"wandb init failed ({e}); disabling.")
        wandb.init(mode="disabled")

    # --- eval helper: monkey-patch BC sample_actions to call the student, then restore ---
    def evaluate(it):
        student.eval()
        orig_sample = bc_policy.sample_actions
        bc_policy.sample_actions = _make_student_sample_actions(bc_policy, student)
        try:
            res = env_runner.run(bc_policy, n_video=0, do_tqdm=cfg.training.use_tqdm,
                                 env_names=task_names)
        finally:
            bc_policy.sample_actions = orig_sample
        student.train()
        sr = res["rollout"]["overall_success_rate"]
        logger.info(f"[iter {it}] eval success rate: {sr:.3f}")
        wlog = {"eval/success_rate": sr}
        if multitask:
            per_task = res.get("rollout_success_rate", {})
            for n, v in per_task.items():
                wlog[f"eval/per_task/{n}"] = v
            logger.info(f"[iter {it}] per-task: " +
                        " ".join(f"{n}={per_task[n]:.2f}" for n in task_names if n in per_task))
        wandb.log(wlog, step=it)
        return sr

    # ---- baseline: BC FM, BEFORE any RL ----
    logger.info("Evaluating frozen BC FM baseline...")
    bc_sr = env_runner.run(bc_policy, n_video=0, do_tqdm=cfg.training.use_tqdm,
                           env_names=task_names)["rollout"]["overall_success_rate"]
    logger.info(f"[iter 0] BC baseline success rate: {bc_sr:.3f}")
    wandb.log({"eval/bc_baseline_success_rate": bc_sr}, step=0)

    # ---- warm-up: collect teacher rollouts to seed the replay ----
    if cfg.dice.warmup_episodes > 0:
        logger.info(f"Warm-up: collecting {cfg.dice.warmup_episodes} teacher rollouts...")
        collector.use_teacher_for_collect = True
        rng = np.random.default_rng(cfg.seed)
        for ep in tqdm(range(cfg.dice.warmup_episodes), disable=not cfg.training.use_tqdm):
            ti = task_indices[ep % len(task_indices)]
            ii = int(rng.integers(0, n_inits_by_task[ti]))
            collector.rollout_episode(ti, ii, replay)
        collector.use_teacher_for_collect = False
        logger.info(f"Warm-up done; replay size = {len(replay)}")

    # ---- main loop ----
    rng = np.random.default_rng(cfg.seed + 1)
    training_step = 0
    for it in range(1, cfg.dice.n_iters + 1):
        # (a) collect
        collect_succ = []
        for _ in range(cfg.dice.episodes_per_iter):
            ti = int(rng.choice(task_indices))
            ii = int(rng.integers(0, n_inits_by_task[ti]))
            succ, _ = collector.rollout_episode(ti, ii, replay)
            collect_succ.append(float(succ))
        c_succ = float(np.mean(collect_succ)) if collect_succ else 0.0
        logger.info(f"[iter {it}] collect: success={c_succ:.3f} replay_size={len(replay)}")
        wandb.log({"collect/success": c_succ, "collect/replay_size": len(replay)}, step=it)

        # (b) update
        if len(replay) >= cfg.dice.batch_size:
            actor_losses, critic_losses, bc_losses, q_means = [], [], [], []
            for _ in range(cfg.dice.gradient_steps):
                batch = replay.sample(cfg.dice.batch_size)
                d = student.loss(
                    state=batch["cond"], noise=batch["noise"], action=batch["action"],
                    next_state=batch["next_cond"], reward=batch["reward"], done=batch["done"],
                    gamma=cfg.dice.gamma,
                )
                actor_opt.zero_grad(set_to_none=True)
                d["actor_total"].backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(), cfg.dice.grad_clip)
                actor_opt.step()

                critic_opt.zero_grad(set_to_none=True)
                # critic_loss is detached from actor params via target_q's no_grad and via
                # critic ensemble using buf actions, but recompute cleanly to be safe.
                with torch.no_grad():
                    next_noise = torch.randn(batch["cond"].shape[0], student.horizon_steps,
                                             student.action_dim, device=device)
                    next_a = student.get_action(batch["next_cond"], next_noise)
                    tgt_q = batch["reward"] + cfg.dice.gamma * (1.0 - batch["done"]) \
                            * student.target_critic(batch["next_cond"], next_noise, next_a)
                cl = student.critic_loss(batch["cond"], batch["noise"], batch["action"], tgt_q)
                cl["critic_loss"].backward()
                torch.nn.utils.clip_grad_norm_(student.critic.parameters(), cfg.dice.grad_clip)
                critic_opt.step()

                student.update_target_critic(cfg.dice.tau)
                training_step += 1
                actor_losses.append(d["actor_total"].item())
                critic_losses.append(cl["critic_loss"].item())
                bc_losses.append(d["actor_bc_loss"].item())
                q_means.append(d["q_actor_mean"].item())

            logger.info(f"[iter {it}] update steps={cfg.dice.gradient_steps} "
                        f"actor={np.mean(actor_losses):.4f} critic={np.mean(critic_losses):.4f} "
                        f"bc_mse={np.mean(bc_losses):.4f} Q(s,a_student)={np.mean(q_means):.4f}")
            wandb.log({"train/actor_loss": float(np.mean(actor_losses)),
                       "train/critic_loss": float(np.mean(critic_losses)),
                       "train/bc_mse": float(np.mean(bc_losses)),
                       "train/q_actor_mean": float(np.mean(q_means))}, step=it)
        else:
            logger.info(f"[iter {it}] skipping update -- replay {len(replay)} < batch_size {cfg.dice.batch_size}")

        # (c) eval
        if it % cfg.dice.eval_interval == 0:
            evaluate(it)

        # (d) ckpt
        if it % cfg.dice.save_interval == 0:
            ckpt = {"student_actor": student.actor.state_dict(),
                    "student_critic": student.critic.state_dict(),
                    "student_target_critic": student.target_critic.state_dict(),
                    "norm_stats": norm_stats,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "iter": it, "training_step": training_step}
            utils.save_state(ckpt, os.path.join(save_dir, f"dice_iter_{it:04d}.pth"))
            utils.save_state(ckpt, os.path.join(save_dir, "dice_latest.pth"))

    # final eval
    evaluate(cfg.dice.n_iters)
    logger.info("DICE fine-tuning complete.")


if __name__ == "__main__":
    main()
