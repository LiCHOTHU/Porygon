"""Multi-task deterministic rollout eval for the drift policy on LIBERO-90.

Loads the drift_multitask_lib90 checkpoint once, then iterates over the 90
LIBERO-90 task names, calling env_runner.run(env_names=[task]) per task.
Resumable: per-task results are written to a text file; tasks already present
are skipped on re-launch (useful under embers preemption + --requeue).

Mirrors eval_drift_rollout_t0.py for the load/eval shape, generalized to the
full benchmark. LIBERO's parallel-env wrapper uses multiprocessing.spawn, so
the `if __name__ == "__main__":` guard is mandatory.
"""
import os
import sys
import argparse

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True,
                   help="Training exp dir containing config.yaml + ckpt.")
    p.add_argument("--ckpt_name", default="multitask_model_latest.pth")
    p.add_argument("--results_file", required=True,
                   help="Per-task success rates appended here; resume key.")
    p.add_argument("--rollouts_per_env", type=int, default=10)
    p.add_argument("--num_parallel_envs", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start_task", type=int, default=0)
    p.add_argument("--end_task", type=int, default=90,
                   help="Exclusive end (eval tasks in [start_task, end_task)).")
    return p.parse_args()


def load_done_tasks(results_file):
    """Return the set of task names whose final success rate already exists."""
    done = {}
    if not os.path.exists(results_file):
        return done
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TASK_RESULT") and "\t" in line:
                # Format: TASK_RESULT\t<idx>\t<name>\t<success_rate>
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        done[parts[2]] = float(parts[3])
                    except ValueError:
                        pass
    return done


def main():
    args = parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    ckpt_path = os.path.join(args.exp_dir, args.ckpt_name)
    assert os.path.exists(ckpt_path), f"Missing ckpt: {ckpt_path}"

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)

    with initialize_config_dir(config_dir=args.exp_dir, version_base="1.2"):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    if "feature_normalize" in cfg.algo.policy:
        del cfg.algo.policy.feature_normalize

    cfg.rollout.rollouts_per_env = args.rollouts_per_env
    cfg.rollout.num_parallel_envs = args.num_parallel_envs
    cfg.rollout.max_episode_length = cfg.task.horizon

    print(f"[eval] ckpt={ckpt_path}")
    print(f"[eval] rollouts_per_env={args.rollouts_per_env} "
          f"num_parallel={args.num_parallel_envs} "
          f"max_steps={cfg.task.horizon}")

    print("[eval] building policy ...")
    policy = instantiate(cfg.algo.policy)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model"])
    policy.normalizer.fit(state["norm_stats"])
    policy.eval().to(args.device)

    print("[eval] building env_runner ...")
    env_runner = instantiate(cfg.task.env_runner)
    env_names = env_runner.env_names
    print(f"[eval] {len(env_names)} tasks in benchmark "
          f"(evaluating [{args.start_task}, {args.end_task}))")

    done = load_done_tasks(args.results_file)
    print(f"[eval] {len(done)} task results already in {args.results_file}")

    # Append-mode results file; one line per task, plus a final SUMMARY.
    with open(args.results_file, "a") as rf:
        rf.write(f"# === eval run: ckpt={ckpt_path} "
                 f"rolls={args.rollouts_per_env} ===\n")
        rf.flush()

        per_task = {}
        for ti in range(args.start_task, args.end_task):
            name = env_names[ti]
            if name in done:
                sr = done[name]
                per_task[name] = sr
                print(f"[eval] CACHED  task={ti:02d}  sr={sr:.3f}  {name}")
                continue

            print(f"[eval] RUN     task={ti:02d}  {name} ...", flush=True)
            output = env_runner.run(
                policy,
                env_names=[name],
                do_tqdm=False,
                fault_tolerant=True,
            )
            sr = float(output["rollout"]["overall_success_rate"])
            per_task[name] = sr
            rf.write(f"TASK_RESULT\t{ti}\t{name}\t{sr:.4f}\n")
            rf.flush()
            print(f"[eval] DONE    task={ti:02d}  sr={sr:.3f}  {name}",
                  flush=True)

        evaluated = [v for v in per_task.values()]
        if evaluated:
            avg = sum(evaluated) / len(evaluated)
            rf.write(f"SUMMARY\t{len(evaluated)}\t{avg:.4f}\n")
            print(f"\n========================================================")
            print(f"SUMMARY  n_tasks={len(evaluated)}  mean_success={avg:.4f}")
            print(f"========================================================")


if __name__ == "__main__":
    main()
