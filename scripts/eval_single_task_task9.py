#!/usr/bin/env python
"""Local task-9-only eval for an imitation FM checkpoint.

Loads cfg + ckpt from a hydra run dir, builds the model + env_runner via
hydra.instantiate, then calls env_runner.run() with env_names restricted to
the LIBERO task corresponding to task.task_subset[0] (defaults to task 9 if
unset). 10 rollouts on that task. Prints success rate and per-episode summary.

Usage:
    python scripts/eval_single_task_task9.py \\
        --run-dir /storage/scratch1/.../task9/run_000 \\
        --ckpt multitask_model_latest.pth \\
        --num-rollouts 10
"""
import argparse
import os
import sys
from pathlib import Path

# Add imitation repo root to sys.path so `import imitation` works regardless of
# CWD when this script is invoked.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

# imitation's configs use `${eval:...}` interpolations (train.py registers
# this resolver). Mirror it here so OmegaConf.load doesn't barf.
OmegaConf.register_new_resolver("eval", eval, replace=True)

import imitation.utils.utils as utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Hydra run dir (contains config.yaml + *.pth)")
    p.add_argument("--ckpt", default="multitask_model_latest.pth")
    p.add_argument("--num-rollouts", type=int, default=10,
                   help="Rollouts per task (overrides cfg.rollout.rollouts_per_env).")
    p.add_argument("--task-id", type=int, default=None,
                   help="LIBERO benchmark task index. Default = first entry of "
                        "cfg.task.task_subset, falling back to 9.")
    p.add_argument("--num-parallel-envs", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Pick the task to eval on.
    if args.task_id is not None:
        task_id = int(args.task_id)
    elif cfg.task.get("task_subset") is not None and len(cfg.task.task_subset) > 0:
        task_id = int(cfg.task.task_subset[0])
    else:
        task_id = 9
    print(f"[eval] task_id={task_id}  rollouts={args.num_rollouts}  "
          f"parallel_envs={args.num_parallel_envs}")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Build model (same path train.py uses).
    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] total params: {n_params/1e6:.2f}M")

    # Load checkpoint. utils.load_checkpoint returns a dict whose 'model' key
    # holds the policy state_dict and 'norm_stats' carries the action/lowdim
    # mean/std/min/max needed by model.normalizer.fit().
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    print(f"[ckpt] loading {ckpt_path}")
    blob = utils.load_checkpoint(str(ckpt_path), logger=None)
    if isinstance(blob, dict) and "model" in blob:
        model_sd = blob["model"]
    else:
        model_sd = blob
    miss, unexp = model.load_state_dict(model_sd, strict=False)
    print(f"[ckpt] missing={len(miss)}  unexpected={len(unexp)}")
    if isinstance(blob, dict) and "norm_stats" in blob:
        model.normalizer.fit(blob["norm_stats"])
        print(f"[ckpt] applied norm_stats to model.normalizer")
    else:
        print(f"[ckpt][warn] no norm_stats in ckpt — normalizer will crash")

    # Build env_runner. Override rollouts_per_env + num_parallel_envs.
    runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    runner_cfg["rollouts_per_env"] = int(args.num_rollouts)
    runner_cfg["num_parallel_envs"] = int(args.num_parallel_envs)
    env_runner = instantiate(runner_cfg)

    # Resolve task name from task_id.
    task_name = env_runner.benchmark.get_task(task_id).name
    print(f"[eval] task_name={task_name!r}")
    print(f"[eval] language: {env_runner.benchmark.get_task(task_id).language!r}")

    # Run.
    print(f"[eval] starting {args.num_rollouts} rollout(s) on task {task_id}...")
    results = env_runner.run(
        model,
        n_video=0,
        do_tqdm=False,
        fault_tolerant=False,
        env_names=[task_name],
    )
    sr = results["rollout"]["overall_success_rate"]
    es = results["rollout"]["environments_solved"]
    per_env = results["rollout"].get("per_env_success_rates", {})
    print()
    print(f"=================  RESULT  =================")
    print(f"  task: {task_name}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  overall_success_rate: {sr:.4f}  ({es}/1 environments solved)")
    print(f"  per-env: {per_env}")
    print(f"=============================================")


if __name__ == "__main__":
    main()
