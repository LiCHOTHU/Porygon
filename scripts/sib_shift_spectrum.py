"""Is the LIBERO appearance shift actually band-localized?  (SIB condition 2)

Before spending rollout compute, measure the thing the method assumes.  SIB can
only help if the nuisance that changes between train and test is concentrated in
some bands rather than spread over all of them -- a band-resolved penalty has no
leverage on a shift that moves every band equally, and would be doing the work of
a plain global penalty.

On CIFAR-10 this same check was decisive: the label explained 53% of band 0 and
under 6% of the other seven, so there was nothing for a mode-resolved method to
resolve, and it correctly predicted the null result before the runs finished.

Here it renders the *same* task from the *same* initial state with and without
each shift, and reports the per-band DCT energy of the difference image.  Output
is a fraction per band; flat means "not band-localized, do not expect SIB to win",
concentrated means the precondition holds.
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

from imitation.algos.sib import SpectralBands  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)

SHIFTS = {
    "lighting": ["task.lighting_shift=true"],
    "color": ["task.color_shift=true"],
    "cam_shift": ["task.cam_shift=small"],
}


def make_env(overrides: list[str], task_id: int):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="train",
                      overrides=["task=libero", "algo=fm_sib_S"] + overrides)
    factory = instantiate(cfg.task.env_factory)
    benchmark = instantiate(cfg.task.benchmark_instance)
    return factory(task_id=task_id, benchmark=benchmark), benchmark


def first_frame(env, benchmark, task_id: int, seed: int) -> np.ndarray:
    """One agentview frame from a fixed initial state, as float CHW in [0,1]."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    init_states = benchmark.get_task_init_states(task_id)
    obs = env.reset()
    env.set_init_state(init_states[0])
    for _ in range(5):  # let the sim settle so the frames are comparable
        obs, *_ = env.step(np.zeros(env.action_space.shape))
    key = next(k for k in obs if "agentview" in k and "eye_in_hand" not in k)
    img = np.asarray(obs[key], dtype=np.float32)
    if img.ndim == 3 and img.shape[0] not in (1, 3):
        img = img.transpose(2, 0, 1)
    return img / 255.0 if img.max() > 1.5 else img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--shifts", nargs="+", default=list(SHIFTS), choices=list(SHIFTS))
    ap.add_argument("--n-bands", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("/storage/scratch1/8/lwang831/imitation/sib_shift_spectrum.json"))
    args = ap.parse_args()

    env, benchmark = make_env([], args.task_id)
    clean = first_frame(env, benchmark, args.task_id, args.seed)
    env.close()

    bands = SpectralBands(clean.shape[1], clean.shape[2], args.n_bands)
    report: dict[str, list[float]] = {}

    clean_t = torch.from_numpy(clean).unsqueeze(0)
    clean_share = bands.band_energy(bands.dct(clean_t))[0]
    report["clean_image"] = (clean_share / clean_share.sum()).tolist()

    for shift in args.shifts:
        env, benchmark = make_env(SHIFTS[shift], args.task_id)
        shifted = first_frame(env, benchmark, args.task_id, args.seed)
        env.close()
        # the difference image isolates what the shift changed, holding the scene
        # and the initial state fixed
        diff = torch.from_numpy(shifted - clean).unsqueeze(0)
        energy = bands.band_energy(bands.dct(diff))[0]
        report[shift] = (energy / energy.sum().clamp_min(1e-12)).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    uniform = 1.0 / args.n_bands
    print(f"\nper-band share of energy (uniform = {uniform:.3f})")
    print("  " + "row".ljust(14) + "".join(f"b{k}".rjust(8) for k in range(args.n_bands)))
    for name, share in report.items():
        print("  " + name.ljust(14) + "".join(f"{v:8.3f}" for v in share))

    # The raw share is not the useful statistic.  Natural images are ~1/f, so on a
    # radial equal-count partition band 0 holds ~99% of the energy no matter what
    # the shift does, and every row looks "band 0 dominated".  What decides whether
    # a band-resolved penalty has leverage is whether the shift is distributed
    # DIFFERENTLY from the image: a shift proportional to the image spectrum
    # (all ratios ~1) is a global gain that any frequency-agnostic method handles,
    # while ratios far from 1 mean the shift lives somewhere specific.
    base = report["clean_image"]
    print(f"\nratio to the clean image spectrum (1.0 = proportional => no leverage)")
    print("  " + "shift".ljust(14) + "".join(f"b{k}".rjust(8) for k in range(args.n_bands)) + "     spread")
    for name, share in report.items():
        if name == "clean_image":
            continue
        ratio = [s / max(b, 1e-12) for s, b in zip(share, base)]
        print("  " + name.ljust(14) + "".join(f"{v:8.2f}" for v in ratio)
              + f"{max(ratio)/max(min(ratio), 1e-12):11.1f}x")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
