import os
import hydra
import torch
import wandb
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import imitation.utils.utils as utils
from imitation.algos.rl import grpo
from imitation.algos.rl.rollout_collector import FlowRLCollector

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="config", config_name="rl_train", version_base=None)
def main(cfg):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    save_dir, experiment_name = utils.get_experiment_dir(cfg, allow_overlap=True)
    os.makedirs(save_dir, exist_ok=True)
    logger = utils.setup_logger("rl_train.log", save_dir)
    logger.info(f"RL fine-tuning experiment: {experiment_name}")
    logger.info(f"Output dir: {save_dir}")

    # --- model + cold-start ---
    policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(device)
    state_dict = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=logger)
    utils.soft_load_state_dict(policy, state_dict["model"])
    norm_stats = state_dict["norm_stats"]
    policy.normalizer.fit(norm_stats)
    policy.eval()  # dropout OFF for the whole RL run; exploration is from rl_sigma
    logger.info(f"Loaded cold-start checkpoint; rl_sigma={policy.rl_sigma}, K={policy.num_inference_steps}")

    # --- env runner (deterministic eval) + collector (shares its benchmark/env_factory) ---
    env_runner = instantiate(cfg.task.env_runner)
    task_indices = list(cfg.task_indices) if cfg.get("task_indices", None) else [cfg.task_index]
    task_names = [env_runner.env_names[t] for t in task_indices]
    multitask = len(task_indices) > 1
    logger.info(f"RL tasks ({len(task_indices)}): indices={task_indices}")
    collector = FlowRLCollector(env_runner, policy, cfg.rl.group_size, device,
                                max_episode_length=cfg.rl.max_episode_length)

    optimizer = torch.optim.AdamW(policy.velocity_net.parameters(), lr=cfg.rl.lr)
    var = policy.step_var
    K = policy.num_inference_steps
    n_inits_by_task = {t: collector.benchmark.get_task_init_states(t).shape[0] for t in task_indices}

    # --- wandb ---
    try:
        wandb.init(project=cfg.logging.project, group=cfg.logging.group,
                   name=experiment_name, config=OmegaConf.to_container(cfg, resolve=True),
                   mode=cfg.logging.mode)
    except Exception as e:
        logger.warning(f"wandb init failed ({e}); disabling.")
        wandb.init(mode="disabled")

    def evaluate(it):
        policy.eval()
        res = env_runner.run(policy, n_video=0, do_tqdm=cfg.training.use_tqdm, env_names=task_names)
        sr = res["rollout"]["overall_success_rate"]
        logger.info(f"[iter {it}] deterministic eval success rate: {sr:.3f}")
        wandb_log = {"eval/success_rate": sr}
        if multitask:
            per_task = res.get("rollout_success_rate", {})
            for name, v in per_task.items():
                wandb_log[f"eval/per_task/{name}"] = v
            per_task_str = " ".join(f"{n}={per_task[n]:.2f}" for n in task_names if n in per_task)
            logger.info(f"[iter {it}] per-task: {per_task_str}")
        wandb.log(wandb_log, step=it)
        return sr

    evaluate(0)  # baseline (frozen cold-start)

    def collect_iter(it, rng):
        """One iter of collection. Multi-task: loop tasks, offset group_ids so the
        GRPO baseline is computed within (task, init), never across tasks."""
        merged = {k: [] for k in ("cond", "chain", "mu_old", "valid", "rewards", "group_ids")}
        t_grid = None
        per_task_succ = {}
        gid_offset = 0
        for ti in task_indices:
            n_inits = n_inits_by_task[ti]
            init_indices = rng.integers(0, n_inits, size=cfg.rl.inits_per_iter).tolist()
            buf, stats = collector.collect(ti, init_indices)
            per_task_succ[env_runner.env_names[ti]] = stats["mean_success"]
            if buf["group_ids"].numel() > 0:
                buf["group_ids"] = buf["group_ids"] + gid_offset
                gid_offset = int(buf["group_ids"].max().item()) + 1
            for k in merged:
                merged[k].append(buf[k])
            if t_grid is None:
                t_grid = buf["t_grid"]
        out = {k: torch.cat(v, dim=0) for k, v in merged.items()}
        out["t_grid"] = t_grid
        mean_succ = float(np.mean(list(per_task_succ.values()))) if per_task_succ else 0.0
        stats = {
            "mean_success": mean_succ,
            "n_rows": int(out["chain"].shape[0]) if out["chain"].numel() else 0,
            "frac_valid": float(out["valid"].mean()) if out["valid"].numel() else 0.0,
            "per_task_success": per_task_succ,
        }
        return out, stats

    rng = np.random.default_rng(cfg.seed)
    for it in range(1, cfg.rl.n_iters + 1):
        buf, stats = collect_iter(it, rng)
        advantages = grpo.compute_grpo_advantages(
            buf["rewards"], buf["group_ids"], std_normalize=cfg.rl.std_normalize)
        logger.info(f"[iter {it}] collect: success={stats['mean_success']:.3f} "
                    f"rows={stats['n_rows']} frac_valid={stats['frac_valid']:.2f} "
                    f"adv|max|={advantages.abs().max():.3f}")
        if multitask:
            pt_str = " ".join(f"{n}={v:.2f}" for n, v in stats["per_task_success"].items())
            logger.info(f"[iter {it}] collect per-task: {pt_str}")
        wandb.log({"collect/success": stats["mean_success"],
                   "collect/rows": stats["n_rows"],
                   "collect/frac_valid": stats["frac_valid"]}, step=it)

        # SimpleVLA-RL filter_accuracy: drop groups whose mean reward is outside [low, high]
        # *before* the PPO update. All-1 / all-0 groups have zero advantage anyway, but
        # dropping them concentrates the optimizer step on prompts with within-group variance.
        # Match SimpleVLA-RL's libero default [0.1, 0.9] -> drops only fully-saturated groups.
        if cfg.rl.get("filter_accuracy", False):
            gids = buf["group_ids"]
            keep_mask = torch.zeros_like(gids, dtype=torch.bool)
            kept = 0
            for g in torch.unique(gids):
                m = (gids == g)
                gm = buf["rewards"][m].float().mean().item()
                if cfg.rl.filter_low < gm < cfg.rl.filter_high:
                    keep_mask |= m
                    kept += 1
            n_total = int(torch.unique(gids).numel())
            n_rows = int(keep_mask.sum().item())
            logger.info(f"[iter {it}] filter [{cfg.rl.filter_low},{cfg.rl.filter_high}]: "
                        f"kept {kept}/{n_total} groups, {n_rows}/{gids.numel()} rows")
            wandb.log({"filter/groups_kept": kept,
                       "filter/groups_total": n_total,
                       "filter/frac_kept": kept / max(1, n_total)}, step=it)
            for k in ("cond", "chain", "mu_old", "valid", "rewards", "group_ids"):
                buf[k] = buf[k][keep_mask]
            advantages = advantages[keep_mask]

        if advantages.numel() == 0 or advantages.abs().max() < 1e-8:
            logger.info(f"[iter {it}] no learning signal (empty or all-equal); skipping update")
        else:
            N = buf["chain"].shape[0]
            t_grid = buf["t_grid"]
            last_info, n_updates, stop = None, 0, False
            for _ in range(cfg.rl.ppo_epochs):
                if stop:
                    break
                perm = torch.randperm(N)
                for s in range(0, N, cfg.rl.minibatch_size):
                    idx = perm[s:s + cfg.rl.minibatch_size]
                    cond_mb = buf["cond"][idx].to(device)
                    chain_mb = buf["chain"][idx].to(device)
                    mu_old_mb = buf["mu_old"][idx].to(device)
                    valid_mb = buf["valid"][idx].to(device)
                    adv_mb = advantages[idx].to(device)
                    logp_new, mu_new = policy.chain_logprob(cond_mb, chain_mb, t_grid)
                    step_mask = valid_mb[:, None].expand(-1, K)
                    loss, info = grpo.ppo_clip_loss(
                        mu_new, mu_old_mb, chain_mb[:, 1:], adv_mb, var,
                        clip_eps=cfg.rl.clip_eps, step_mask=step_mask)
                    # KL early-stop: skip the overshooting step and end the update for this iter
                    if info["approx_kl"] > cfg.rl.target_kl:
                        stop = True
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.velocity_net.parameters(), cfg.rl.grad_clip)
                    optimizer.step()
                    n_updates += 1
                    last_info = info
            last_info = last_info or info
            logger.info(f"[iter {it}] update ({n_updates} steps): {last_info}")
            wandb.log({f"ppo/{k}": v for k, v in last_info.items()}, step=it)

        if it % cfg.rl.eval_interval == 0:
            evaluate(it)
        if it % cfg.rl.save_interval == 0:
            ckpt = {"model": policy, "norm_stats": norm_stats,
                    "config": OmegaConf.to_container(cfg, resolve=True), "epoch": it}
            utils.save_state(ckpt, os.path.join(save_dir, f"rl_model_iter_{it:04d}.pth"))
            utils.save_state(ckpt, os.path.join(save_dir, "rl_model_latest.pth"))

    evaluate(cfg.rl.n_iters)
    logger.info("RL fine-tuning complete.")


if __name__ == "__main__":
    main()
