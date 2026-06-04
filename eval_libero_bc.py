"""Deterministic LIBERO rollout eval for a BC checkpoint (no RL).

Extracts only the `evaluate(0)` portion of rl_train.py so it works for the
BC-trained `FlowMatchingPolicy` (which does NOT have RL-only attrs like
`rl_sigma`).

Usage (matches rl_train.py hydra config):
    python eval_libero_bc.py \
        algo=fm_policy_r2 \
        cold_start_checkpoint=/path/to/ckpt.pth \
        task_index=0 \
        rl.eval_rollouts_per_env=5 \
        logging.mode=disabled \
        exp_name=eval_libero_bc_r2_t0

We reuse the rl_train hydra config because that's where the env_runner and
shape_meta are already wired. We just skip the RL-specific parts.
"""
import os
import hydra
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

import imitation.utils.utils as utils

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="config", config_name="rl_train", version_base=None)
def main(cfg):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    save_dir, experiment_name = utils.get_experiment_dir(cfg, allow_overlap=True)
    os.makedirs(save_dir, exist_ok=True)
    logger = utils.setup_logger("eval_libero_bc.log", save_dir)
    logger.info(f"BC eval experiment: {experiment_name}")
    logger.info(f"Output dir: {save_dir}")

    policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(device)
    state_dict = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=logger)
    utils.soft_load_state_dict(policy, state_dict["model"])
    norm_stats = state_dict["norm_stats"]
    policy.normalizer.fit(norm_stats)
    policy.eval()
    logger.info(f"Loaded BC checkpoint; K={getattr(policy, 'num_inference_steps', '?')}")

    env_runner = instantiate(cfg.task.env_runner)
    task_indices = list(cfg.task_indices) if cfg.get("task_indices", None) else [cfg.task_index]
    task_names = [env_runner.env_names[t] for t in task_indices]
    logger.info(f"Eval tasks ({len(task_indices)}): indices={task_indices}")

    res = env_runner.run(policy, n_video=0, do_tqdm=cfg.training.use_tqdm, env_names=task_names)
    sr = res["rollout"]["overall_success_rate"]
    logger.info(f"[iter 0] deterministic eval success rate: {sr:.3f}")
    if len(task_indices) > 1:
        per_task = res.get("rollout_success_rate", {})
        per_task_str = " ".join(f"{n}={per_task[n]:.2f}" for n in task_names if n in per_task)
        logger.info(f"[iter 0] per-task: {per_task_str}")
    return sr


if __name__ == "__main__":
    main()
