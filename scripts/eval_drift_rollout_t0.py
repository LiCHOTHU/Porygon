"""Run rollout success-rate eval on the lambertae-port task-0 drift checkpoint
using the same LiberoRunner the imitation training loop uses internally.

Note: LIBERO's parallel-env wrapper uses multiprocessing.spawn, which requires
the main-module guard `if __name__ == "__main__":` so the child workers can
re-import this file without re-running the eval.
"""
import os, sys
sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

EXP_DIR = "/storage/scratch1/8/lwang831/imitation/local/libero/libero_90/drift_single_t0_lambertae"
CKPT = os.path.join(EXP_DIR, "multitask_model_epoch_0050.pth")
DEVICE = "cuda:0"
NUM_ROLLOUTS = 20
NUM_PARALLEL = 5
TASK_INDEX = 0


def main():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    with initialize_config_dir(config_dir=EXP_DIR, version_base="1.2"):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    if "feature_normalize" in cfg.algo.policy:
        del cfg.algo.policy.feature_normalize

    cfg.rollout.rollouts_per_env = NUM_ROLLOUTS
    cfg.rollout.num_parallel_envs = NUM_PARALLEL
    cfg.rollout.max_episode_length = cfg.task.horizon

    print("Building policy...")
    policy = instantiate(cfg.algo.policy)
    print(f"  params = {sum(p.numel() for p in policy.parameters())}")
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model"])
    policy.normalizer.fit(state["norm_stats"])
    policy.eval().to(DEVICE)

    print("Building env_runner...")
    env_runner = instantiate(cfg.task.env_runner)

    task_name = env_runner.env_names[TASK_INDEX]
    print(f"Eval target: task index {TASK_INDEX} -> env_name = '{task_name}'")
    print(f"Rollouts: {NUM_ROLLOUTS}  parallel: {NUM_PARALLEL}  max_steps: {cfg.task.horizon}")

    output = env_runner.run(
        policy,
        env_names=[task_name],
        do_tqdm=True,
        fault_tolerant=True,
    )

    sr = output["rollout"]["overall_success_rate"]
    per_env = output["rollout_success_rate"]
    n_solved = output["rollout"]["environments_solved"]
    print("\n========================================================")
    print(f"TASK '{task_name}'")
    print(f"  success rate:      {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  per-env:           {per_env}")
    print(f"  environments solved (any success): {n_solved}/1")
    print(f"  average reward:    {output['rollout']['overall_average_reward']:.4f}")
    print("========================================================")


if __name__ == "__main__":
    main()
