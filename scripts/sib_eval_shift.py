"""Rollout success rate of a trained policy under held-out appearance shift.

The SIB claim is not that the penalty makes a policy better at the scenes it was
trained on -- it removes reliance, it does not add capability.  The claim is that
a policy which was *stopped* from steering on band-localized nuisances still works
when those nuisances change.  So the number that decides the experiment is
success rate on shifted scenes, and clean success rate is the control:

    clean      SIB ~= vanilla        (no capability was added)
    shifted    SIB >  vanilla, and by more than mixup / sam / dropout

Which shifts, and why these.  Chosen by measurement, not by argument -- see
``sib_shift_spectrum.py``, which renders the same scene from the same initial
state with and without each shift and asks where in the spectrum the difference
lives, relative to the image's own spectrum:

    shift        b0     b1     b2     b3     b4     b5     b6     b7   spread
    lighting   1.00   1.33   1.53   1.69   1.64   1.64   1.55   1.85     1.9x
    cam_shift  0.95   6.64   7.03   6.70   6.52   6.91   8.41   9.27     9.8x

``cam_shift`` is the primary test: it leaves band 0 essentially alone and adds
6-9x energy to every higher band, so there is real band structure for a
mode-resolved penalty to act on (SIB condition 2).

``lighting`` is kept as a NEGATIVE control.  Its ratios sit near 1, meaning it is
close to a global gain distributed like the image itself -- there is little for a
band-resolved method to exploit that a frequency-agnostic one cannot.  If SIB
wins on cam_shift but not on lighting, that is consistent with the stated
mechanism.  If it wins equally on both, it is acting as a generic regularizer and
the mode-resolution is not what is doing the work.

Two shifts are unavailable in this checkout and are not silently skipped:
``color`` raises ``MjModel has no attribute 'tex_rgb'`` (robosuite's texture
modder against the installed mujoco), and ``distractor_objects`` needs
``quest/libero_distractor/bddl_files``, which is absent.

Caveat worth keeping in view: the spectrum above is measured on the raw image,
while the penalty acts on the encoder feature map.  It selects the shift; it does
not by itself establish that the feature map inherits the same structure.

Training is never touched here.  The dataset is not even built -- normalization
statistics come out of the checkpoint -- so an eval is sim-bound, which is why
this runs as a separate job on whatever GPU is free rather than on the L40S.

Usage::

    python scripts/sib_eval_shift.py --arms vanilla sib mixup sam dropout \\
        --shifts clean lighting color --rollouts 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import imitation.utils.utils as utils  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)

# arm name -> algo config, matching scripts/sib_libero.sbatch
ARMS = {
    "vanilla": "fm_policy_S",
    "sib": "fm_sib_S",
    "mixup": "fm_mixup_S",
    "sam": "fm_sam_S",
    "dropout": "fm_dropout_S",
}

# shift name -> config overrides.  `clean` is the control and must be measured in
# the same process and with the same init states as the shifts, or the comparison
# picks up sim-version differences instead of the shift.
SHIFTS = {
    "clean": [],                                  # control: no capability was added
    "cam_shift": ["task.cam_shift=small"],        # primary: 9.8x band spread
    "cam_shift_medium": ["task.cam_shift=medium"],
    "lighting": ["task.lighting_shift=true"],     # negative control: 1.9x, near-global gain
}

TASK_SUBSET = list(range(20))  # the 20 tasks the arms were trained on


def build_cfg(arm: str, shift: str, rollouts: int, subset: list[int]):
    overrides = [
        "task=libero",
        f"algo={ARMS[arm]}",
        f"task.task_subset=[{','.join(str(i) for i in subset)}]",
        f"rollout.rollouts_per_env={rollouts}",
        "rollout.enabled=true",
        "logging.mode=disabled",
        # must match training: config/train.yaml defaults output_prefix to the
        # relative './experiments' inside the repo, i.e. inside the home quota
        "output_prefix=/storage/scratch1/8/lwang831/imitation/experiments",
        f"exp_name=sib_libero_{arm}",
    ] + SHIFTS[shift]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return compose(config_name="train", overrides=overrides)


def evaluate(arm: str, shift: str, rollouts: int, subset: list[int], seed: int) -> dict:
    cfg = build_cfg(arm, shift, rollouts, subset)
    # Same seed across arms and shifts: the init states drawn from the benchmark
    # pool are then identical, so a difference between two arms is the policy and
    # not the starting configuration.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    model.to(cfg.device)

    experiment_dir, _ = utils.get_experiment_dir(cfg)
    state_dict = utils.load_checkpoint(experiment_dir)
    if state_dict is None:
        raise FileNotFoundError(f"no checkpoint for arm {arm!r} under {experiment_dir}")
    utils.soft_load_state_dict(model, state_dict["model"])
    # The dataset is never constructed -- normalization comes from the checkpoint.
    # Rebuilding it here would cost minutes per (arm, shift) cell and could not
    # change the answer.
    model.normalizer.fit(state_dict["norm_stats"])
    model.eval()

    runner = instantiate(cfg.task.env_runner)
    results = runner.run(model, n_video=0, do_tqdm=False, fault_tolerant=False)
    per_env = results.get("rollout_success_rate", {})
    return {
        "arm": arm,
        "shift": shift,
        "epoch": int(state_dict.get("epoch", -1)),
        "success_rate": float(results["rollout"]["overall_success_rate"]),
        "environments_solved": int(results["rollout"]["environments_solved"]),
        "per_task": {k: float(v) for k, v in per_env.items()},
        "n_tasks": len(subset),
        "rollouts_per_env": rollouts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--shifts", nargs="+", default=["clean", "cam_shift", "lighting"],
                    choices=list(SHIFTS))
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=len(TASK_SUBSET))
    ap.add_argument("--out", type=Path,
                    default=Path("/storage/scratch1/8/lwang831/imitation/sib_shift_eval.jsonl"))
    args = ap.parse_args()

    subset = TASK_SUBSET[: args.n_tasks]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume-safe: a killed job re-runs only the cells it never finished.  Rollout
    # eval is slow and requeueing is normal on embers.
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["arm"], rec["shift"]))

    for shift in args.shifts:
        for arm in args.arms:
            if (arm, shift) in done:
                print(f"[cached] {arm:>8} / {shift}", flush=True)
                continue
            print(f"[run]    {arm:>8} / {shift} ...", flush=True)
            try:
                rec = evaluate(arm, shift, args.rollouts, subset, args.seed)
            except Exception as exc:  # one missing arm must not lose the others
                print(f"[FAIL]   {arm:>8} / {shift}: {type(exc).__name__}: {exc}", flush=True)
                continue
            with args.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[done]   {arm:>8} / {shift}  success={rec['success_rate']:.3f} "
                  f"solved={rec['environments_solved']}/{rec['n_tasks']}", flush=True)

    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
