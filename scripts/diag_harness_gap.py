"""Why does ONE checkpoint score 0.279 in the baselines' harness and 0.520 in DPPO's?

The noise-floor explanation is dead: removing the floor moved the number by
0.001 (basedet_arch32 = 0.278 vs 0.279). So something else in the two sampling
loops differs. This loads the SAME weights into both wrappers, feeds them the
SAME observations and the SAME initial noise, and diffs the actions they emit.

If the actions differ, the sampler is responsible and the diff localises it.
If they match, the gap is in the environment loop, not the policy.

usage: python scripts/diag_harness_gap.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/workspace/dice_rl_official/dice-rl"))

CKPT = ("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/"
        "log_dir/robomimic-pretrain/square_diff_baselinearch_42/checkpoint/state_8000.pt")
OBS_DIM, ACTION_DIM, HORIZON, DENOISE = 23, 7, 4, 20
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(cls_path, **kw):
    mod, cls = cls_path.rsplit(".", 1)
    m = __import__(mod, fromlist=[cls])
    from model.diffusion.mlp_diffusion import DiffusionMLP
    net = DiffusionMLP(
        horizon_steps=HORIZON, action_dim=ACTION_DIM,
        cond_dim=OBS_DIM, time_dim=32,
        mlp_dims=[1024, 1024, 1024], cond_mlp_dims=[512, 64],
        residual_style=True,
    )
    return getattr(m, cls)(network=net, horizon_steps=HORIZON, obs_dim=OBS_DIM,
                           action_dim=ACTION_DIM, denoising_steps=DENOISE,
                           device=DEV, **kw)


def main():
    print(f"device {DEV}\nckpt   {os.path.basename(CKPT)}\n")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    weights = sd.get("ema", sd.get("model", sd))

    torch.manual_seed(0)
    B = 64
    obs = {"state": torch.randn(B, 1, OBS_DIM, device=DEV)}
    z = torch.randn(B, HORIZON, ACTION_DIM, device=DEV)      # identical init noise

    out = {}
    for tag, path, kw in [
        ("DIPO  (baselines)", "model.diffusion.diffusion_dipo.DIPODiffusion", {}),
        ("DPPO", "model.diffusion.diffusion_ppo.PPODiffusion",
         dict(ft_denoising_steps=10, actor=None)),
    ]:
        try:
            pol = build(path, **{k: v for k, v in kw.items() if v is not None})
            missing = pol.load_state_dict(
                {k.replace("network.", "network.", 1): v for k, v in weights.items()},
                strict=False)
            pol.to(DEV).eval()
            with torch.no_grad():
                a = pol.forward(cond=obs, deterministic=True)
            out[tag] = a.reshape(B, -1).cpu().numpy()
            print(f"{tag:18s} action mean={out[tag].mean():+.4f} std={out[tag].std():.4f} "
                  f"|a|max={np.abs(out[tag]).max():.4f}")
            print(f"{'':18s} load: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
        except Exception as e:
            print(f"{tag:18s} FAILED: {type(e).__name__}: {str(e)[:150]}")

    if len(out) == 2:
        a, b = list(out.values())
        d = np.abs(a - b)
        print(f"\nper-element |difference|: mean={d.mean():.5f} max={d.max():.5f}")
        print("VERDICT:", "SAMPLERS DIFFER -> the gap is in the policy loop"
              if d.max() > 1e-4 else "actions match -> the gap is NOT in the sampler")


if __name__ == "__main__":
    main()
