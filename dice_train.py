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
        # student.get_action returns a_teacher + residual (UNCLAMPED). The env path
        # needs actions in [-1, 1] so clamp here, at the env boundary.
        a = torch.clamp(student_model.get_action(cond, noise), -1, 1)
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

    # --- student model (mirror official knobs) ---
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
        critic_weight=cfg.dice.get("critic_weight", 1.0),
        num_multi_z=cfg.dice.num_multi_z,
        use_soft_q_filtering=cfg.dice.get("use_soft_q_filtering", False),
        q_filtering_warmup_steps=cfg.dice.get("q_filtering_warmup_steps", 25000),
        q_underestimation_threshold=cfg.dice.get("q_underestimation_threshold", -0.1),
        replay_flow_warmup_steps=cfg.dice.get("replay_flow_warmup_steps", 1000),
        use_q_normalization=cfg.dice.get("use_q_normalization", False),
        multi_sample_next_noise=cfg.dice.get("multi_sample_next_noise", False),
        num_next_noise_samples=cfg.dice.get("num_next_noise_samples", 4),
        use_n_step=cfg.dice.get("use_n_step", False),
        n_step=cfg.dice.get("n_step", 1),
        disable_q_loss_for_expert_data=cfg.dice.get("disable_q_loss_for_expert_data", False),
        disable_td_loss_for_expert_data=cfg.dice.get("disable_td_loss_for_expert_data", False),
        always_retain_bc_loss_for_expert_data=cfg.dice.get("always_retain_bc_loss_for_expert_data", False),
        clip_action=cfg.dice.get("clip_action", True),
        zero_final_layer=cfg.dice.get("zero_final_layer", False),
        device=device,
    ).to(device)
    teacher = FMTeacher(bc_policy)
    student.attach_teacher(teacher)
    logger.info(f"Student actor params: {sum(p.numel() for p in student.actor.parameters())/1e6:.2f}M; "
                f"critic params: {sum(p.numel() for p in student.critic.parameters())/1e6:.2f}M")

    # Optional: resume a previously-trained DICE student for eval-only / continued training.
    dice_resume = cfg.get("dice_resume_checkpoint", None)
    if dice_resume:
        logger.info(f"Loading DICE student from {dice_resume}")
        dice_state = torch.load(dice_resume, map_location=device)
        student.actor.load_state_dict(dice_state["student_actor"])
        student.critic.load_state_dict(dice_state["student_critic"])
        if "student_target_critic" in dice_state:
            student.target_critic.load_state_dict(dice_state["student_target_critic"])
        logger.info(f"Resumed DICE student (saved at iter={dice_state.get('iter','?')}, "
                    f"training_step={dice_state.get('training_step','?')})")

    # --- optimizers (official uses Adam, not AdamW; match that) ---
    actor_opt = torch.optim.Adam(
        student.actor.parameters(), lr=cfg.dice.actor_lr,
        weight_decay=cfg.dice.get("actor_weight_decay", 0.0),
    )
    critic_opt = torch.optim.Adam(
        student.critic.parameters(), lr=cfg.dice.critic_lr,
        weight_decay=cfg.dice.get("critic_weight_decay", 0.0),
    )
    actor_update_freq = int(cfg.dice.get("actor_update_freq", 1))
    target_update_freq = int(cfg.dice.get("critic_target_update_freq", 1))
    log_q_overestimation = bool(cfg.dice.get("log_q_overestimation", False))

    # --- replay + collector ---
    replay = ReplayBuffer(
        max_size=cfg.dice.replay_size,
        cond_shape=(num_enc, hidden),
        horizon=bc_policy.chunk_size,
        action_dim=bc_policy.network_action_dim,
        device=device,
        gamma=cfg.dice.gamma,
        use_n_step=cfg.dice.get("use_n_step", False),
        n_step=cfg.dice.get("n_step", 1),
        use_rlpd=cfg.dice.get("use_rlpd", False),
        expert_ratio=cfg.dice.get("expert_ratio", 0.5),
        expert_dataset=None,  # RLPD demo loader hook -- TODO wiring
    )
    collector = DiceCollector(
        env_runner, bc_policy, student, device,
        max_episode_length=cfg.dice.max_episode_length,
        use_teacher_for_collect=False,
        online_explore_strategy=cfg.dice.get("online_explore_strategy", "standard"),
        num_exploration_samples=cfg.dice.get("num_exploration_samples", 10),
    )

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
        # (a) collect (pass training_step so Q-based exploration respects warmup)
        collect_succ = []
        for _ in range(cfg.dice.episodes_per_iter):
            ti = int(rng.choice(task_indices))
            ii = int(rng.integers(0, n_inits_by_task[ti]))
            succ, _ = collector.rollout_episode(ti, ii, replay, training_step=training_step)
            collect_succ.append(float(succ))
        c_succ = float(np.mean(collect_succ)) if collect_succ else 0.0
        logger.info(f"[iter {it}] collect: success={c_succ:.3f} replay_size={len(replay)}")
        wandb.log({"collect/success": c_succ, "collect/replay_size": len(replay)}, step=it)

        # (b) update -- official-style joint actor+critic loss, with delayed updates
        if len(replay) >= cfg.dice.batch_size:
            actor_losses, critic_losses, bc_losses, q_means, qfilt_active = [], [], [], [], []
            for gs in range(cfg.dice.gradient_steps):
                batch = replay.sample(cfg.dice.batch_size)

                # Q-overestimation = predicted_q - mc_return  (for soft Q-filtering).
                q_over = None
                if log_q_overestimation:
                    with torch.no_grad():
                        # When q_depends_on_noise=False, noise is unused.
                        predicted_q = student.critic(batch["cond"], batch["noise"], batch["action"])
                        q_over = predicted_q - batch["mc_return"]

                d = student.loss(
                    state=batch["cond"], noise=batch["noise"], action=batch["action"],
                    next_state=batch["next_cond"], reward=batch["reward"], done=batch["done"],
                    gamma=cfg.dice.gamma,
                    training_step=training_step,
                    q_overestimation=q_over,
                    n_steps=batch["n_steps"],
                    data_source=batch["data_source"],
                )

                # actor_total and critic_loss are INDEPENDENT graphs (two distinct
                # critic forwards: one on a_student, one on the buffer action), so we
                # can backward through them separately without retain_graph.
                #
                # Actor update (delayed): when we skip the actor step we don't backward
                # at all -- the actor graph buffers get freed when `d` goes out of scope.
                # backward-without-step would WRONGLY accumulate grads across skipped
                # iterations and corrupt the next real actor step.
                actor_step = ((training_step + 1) % actor_update_freq == 0)
                if actor_step:
                    actor_opt.zero_grad(set_to_none=True)
                    d["actor_total"].backward()  # also dirties critic grads -> cleared below
                    torch.nn.utils.clip_grad_norm_(student.actor.parameters(), cfg.dice.grad_clip)
                    actor_opt.step()

                # Critic update (always). zero_grad clears any critic grads dirtied by
                # the actor backward above (if it ran). The critic forward inside
                # actor_loss touched critic params but those grads must not contribute
                # to the critic optimizer step -- that step should follow ONLY the TD
                # gradient.
                critic_opt.zero_grad(set_to_none=True)
                d["critic_loss"].backward()
                torch.nn.utils.clip_grad_norm_(student.critic.parameters(), cfg.dice.grad_clip)
                critic_opt.step()

                # Polyak (delayed).
                if (training_step + 1) % target_update_freq == 0:
                    student.update_target_networks(cfg.dice.tau)

                training_step += 1
                actor_losses.append(float(d["actor_total"].detach()))
                critic_losses.append(float(d["critic_loss"].detach()))
                bc_losses.append(float(d["actor_bc_loss"].detach()))
                q_means.append(float(d["current_q_mean"].detach()))
                qfilt_active.append(float(d["q_filtering_active"].detach()))

            logger.info(f"[iter {it}] update steps={cfg.dice.gradient_steps} "
                        f"actor={np.mean(actor_losses):.4f} critic={np.mean(critic_losses):.4f} "
                        f"residual_mse={np.mean(bc_losses):.4f} Q(s,a_student)={np.mean(q_means):.4f} "
                        f"qfilt_keep={np.mean(qfilt_active):.3f}")
            wandb.log({"train/actor_loss": float(np.mean(actor_losses)),
                       "train/critic_loss": float(np.mean(critic_losses)),
                       "train/residual_mse": float(np.mean(bc_losses)),
                       "train/q_actor_mean": float(np.mean(q_means)),
                       "train/q_filtering_active": float(np.mean(qfilt_active)),
                       "train/training_step": training_step}, step=it)
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
