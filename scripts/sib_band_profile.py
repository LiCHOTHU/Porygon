"""Per-band S_k and V_k for a trained SIB policy -- the mechanism test.

The success-rate sweep cannot tell the two stories apart.  SIB scored 0.560
under cam_shift against SAM's 0.440, but a paired sign test over the 20 tasks
put that difference at p=0.581, and "SIB is indistinguishable from SAM" is
exactly what you would see if the spectral machinery were doing nothing and the
double backward were incidentally smoothing the model the way SAM does.  One
scalar at the end of a long causal chain cannot separate a working band-selective
penalty from a generic smoothness prior that happens to land in the same place.

The claim in sib.py is specific and directly checkable: nuisance is
band-localized, so the penalty should CONCENTRATE on a few bands.  What the
penalty actually charges is sqrt(S_k * V_k) summed over k, and both factors are
length-n_bands vectors -- but sib_penalty reduces them to S_max / V_max / a
median before they reach `info`, and training ran with logging disabled, so the
profile has never been looked at.  This dumps the whole vector.

Read it as:

  concentrated (a few bands carry most of sum_k sqrt(S_k V_k))
      the mechanism is doing what it claims; the SAM tie is an underpowered
      comparison and worth more seeds

  flat (every band charged about equally)
      SIB is a generic smoothness prior wearing a spectral costume.  The SAM
      equivalence is the honest result and no number of seeds changes it.

Nothing here is retrained: it runs on the existing checkpoint.  A handful of
batches is enough because these are expectations over the data distribution, and
they are reported with a spread across batches so a single unlucky batch cannot
be mistaken for structure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# config/ uses ${eval:...} interpolations, and the resolver is registered by
# train.py -- which this script does not import.  Without it, compose() dies on
# the first interpolated field.
OmegaConf.register_new_resolver("eval", eval, replace=True)

from imitation.algos.sib import reliance_band_energy  # noqa: E402
from imitation.utils import utils  # noqa: E402


def build_cfg(arm_target: str, exp_name: str, subset: list[int]):
    overrides = [
        "task=libero",
        f"algo={arm_target}",
        f"task.task_subset=[{','.join(str(i) for i in subset)}]",
        "rollout.enabled=false",
        "logging.mode=disabled",
        "output_prefix=/storage/scratch1/8/lwang831/imitation/experiments",
        "task.demos_per_env=50",
        f"exp_name={exp_name}",
    ]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return compose(config_name="train", overrides=overrides)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="fm_sib_S")
    ap.add_argument("--exp-name", default="sib_libero_sib")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", default="/storage/scratch1/8/lwang831/imitation/sib_band_profile.json")
    args = ap.parse_args()

    subset = list(range(20))
    cfg = build_cfg(args.algo, args.exp_name, subset)
    device = cfg.device

    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    model.to(device)

    experiment_dir, _ = utils.get_experiment_dir(cfg)
    state = utils.load_checkpoint(experiment_dir)
    if state is None:
        raise FileNotFoundError(f"no checkpoint under {experiment_dir}")
    model.normalizer.fit(state["norm_stats"])

    dataset = utils.make_dataset(cfg)
    model.preprocess_dataset(dataset, use_tqdm=False)
    loader = instantiate(cfg.train_dataloader, dataset=dataset)

    # The SIB submodules (bands / conditioner / EMA) are attached lazily on the
    # first compute_loss, once the encoder has revealed its feature-map shape --
    # so the checkpoint can only be loaded AFTER a forward pass has created the
    # parameters it is meant to populate.  Loading before this point silently
    # drops every sib_* tensor and would profile a randomly-initialised
    # conditioner, which is the failure this whole script exists to detect.
    model.train()
    warm = utils.map_tensor_to_device(next(iter(loader)), device)
    model.compute_loss(warm)
    utils.soft_load_state_dict(model, state["model"])
    model.eval()

    if getattr(model, "sib_bands", None) is None:
        raise RuntimeError("sib_bands never attached -- is this a SIB checkpoint?")

    n_bands = model.sib_bands.n_bands
    s_rows: list[list[float]] = []
    v_rows: list[list[float]] = []
    e_rows: list[list[float]] = []

    # The flow terms (velocity, x1, t) are locals inside compute_loss and are
    # never stored on the model, so the per-band vectors are recovered by
    # intercepting the real penalty call rather than by reconstructing its
    # inputs.  Recomputing them here instead would risk profiling a slightly
    # different quantity than the one training actually charged, which is the
    # one failure this script must not have.
    import imitation.algos.fm_policy as fm_policy
    real_penalty = fm_policy.sib_penalty
    captured: dict = {}

    def spy(zs, velocity, actions, t, bands, conditioner, ema, **kw):
        reps = zs[0].shape[0] // actions.shape[0]
        a, tt = (actions.repeat_interleave(reps, 0), t.repeat_interleave(reps, 0)) \
            if reps > 1 else (actions, t)
        v_k = sum(bands.band_energy(conditioner.residual(z, a, tt)).mean(dim=0)
                  for z in zs) / len(zs)
        # create_graph=False: the profile only reads S_k, nothing backprops
        # through it, and the second-order graph is the expensive part.
        s_k = reliance_band_energy(zs, velocity, bands, kw.get("n_probes", 1),
                                   create_graph=False)
        with torch.no_grad():
            e_k = sum(bands.band_energy(z.detach()).mean(dim=0) for z in zs) / len(zs)
        captured["s"] = [float(x) for x in s_k.detach()]
        captured["v"] = [float(x) for x in v_k.detach()]
        captured["e"] = [float(x) for x in e_k]
        return real_penalty(zs, velocity, actions, t, bands, conditioner, ema, **kw)

    fm_policy.sib_penalty = spy
    try:
        for i, data in enumerate(loader):
            if i >= args.batches:
                break
            data = utils.map_tensor_to_device(data, device)
            captured.clear()
            # grad is required: S_k is a Jacobian-vector norm, so no no_grad here.
            model.compute_loss(data)
            if not captured:
                raise RuntimeError("penalty never fired -- sib_beta is 0?")
            s_rows.append(captured["s"])
            v_rows.append(captured["v"])
            e_rows.append(captured["e"])
            model.zero_grad(set_to_none=True)
    finally:
        fm_policy.sib_penalty = real_penalty

    def col(rows, k):
        return [r[k] for r in rows]

    bands = []
    for k in range(n_bands):
        s = statistics.mean(col(s_rows, k))
        v = statistics.mean(col(v_rows, k))
        e = statistics.mean(col(e_rows, k))
        contrib = (s * v + 1e-8) ** 0.5
        bands.append({
            "band": k,
            "S": s, "V": v, "energy": e,
            # V/E near 0 means the representation collapsed onto the conditional
            # mean rather than the policy relying on the band less -- the
            # degenerate solution sib.py warns about, and it would look like
            # "success" in the penalty value alone.
            "V_over_E": v / max(e, 1e-12),
            "sqrt_SV": contrib,
            "sqrt_SV_sd": statistics.pstdev(
                [(sr * vr + 1e-8) ** 0.5 for sr, vr in zip(col(s_rows, k), col(v_rows, k))]
            ),
        })

    total = sum(b["sqrt_SV"] for b in bands) or 1e-12
    for b in bands:
        b["share"] = b["sqrt_SV"] / total

    shares = sorted((b["share"] for b in bands), reverse=True)
    top2 = sum(shares[:2])
    # Herfindahl index: 1/n_bands under a perfectly flat profile, 1.0 if a single
    # band carries everything.  Reported alongside top-2 share so "concentrated"
    # is a number rather than an impression formed from eyeballing the table.
    hhi = sum(s * s for s in shares)
    flat = 1.0 / n_bands

    out = {
        "arm": args.exp_name,
        "n_bands": n_bands,
        "batches": len(s_rows),
        "bands": bands,
        "top2_share": top2,
        "hhi": hhi,
        "hhi_if_flat": flat,
        "concentration_ratio": hhi / flat,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n{args.exp_name}: {len(s_rows)} batches, {n_bands} bands")
    print(f"{'band':>4} {'share':>7} {'sqrt(SV)':>11} {'+/-':>9} {'S':>11} {'V':>11} {'V/E':>7}")
    for b in bands:
        print("%4d %6.1f%% %11.4g %9.2g %11.4g %11.4g %7.3f"
              % (b["band"], 100 * b["share"], b["sqrt_SV"], b["sqrt_SV_sd"],
                 b["S"], b["V"], b["V_over_E"]))
    print(f"\ntop-2 share {100*top2:.1f}%   HHI {hhi:.3f} vs {flat:.3f} flat "
          f"({hhi/flat:.2f}x)")
    print("CONCENTRATED -- band-selective mechanism is live" if hhi / flat > 1.5
          else "FLAT -- indistinguishable from a generic smoothness prior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
