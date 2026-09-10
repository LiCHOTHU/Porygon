"""Why do QSM and DQL collapse on the first update? Local diagnostic.

QSM's actor loss (model/diffusion/diffusion_qsm.py:loss_actor) is

    target = q_grad_coeff * dQ/da
    loss   = MSE(-network(x_noisy, t, cond), target)

i.e. the pretrained denoiser is trained to output the *scaled critic gradient*
in place of the noise it was pretrained to predict. There is no behaviour-
cloning term holding it anywhere. So the update is only sane if

    ||q_grad_coeff * dQ/da||  ~=  ||noise||

because the left side is what the denoiser is now asked to emit and the right
side is what it currently emits. If the two differ by orders of magnitude the
denoiser is overwritten with an out-of-scale field on the first pass, which
would explain a collapse to 0.000 within one evaluation interval rather than
a gradual decline.

This script loads the pretrained base and a freshly initialised critic (the
state at the START of fine-tuning, which is when the collapse happens) and
compares the two magnitudes directly.

Usage: python scripts/diag_qsm_scale.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/workspace/dice_rl_official/dice-rl"))

CKPT = ("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/"
        "log_dir/robomimic-pretrain/square_diff_baselinearch_42/checkpoint/state_8000.pt")
OBS_DIM, ACTION_DIM, HORIZON = 23, 7, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"device: {DEV}")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"loaded base: {os.path.basename(CKPT)}  keys={list(sd.keys())[:4]}")

    from model.common.critic import CriticObsAct

    torch.manual_seed(0)
    critic = CriticObsAct(
        mlp_dims=[1024, 1024, 1024], activation_type="Mish", residual_style=True,
        cond_dim=OBS_DIM, action_dim=ACTION_DIM, action_steps=HORIZON,
    ).to(DEV)

    B = 256
    obs = {"state": torch.randn(B, 1, OBS_DIM, device=DEV)}
    act = torch.rand(B, HORIZON, ACTION_DIM, device=DEV) * 2 - 1   # in [-1,1]

    # dQ/da at a freshly initialised critic == the state at fine-tuning step 0
    act.requires_grad_(True)
    q1, q2 = critic(obs, act)
    g1 = torch.autograd.grad(q1.sum(), act, retain_graph=True)[0]
    g2 = torch.autograd.grad(q2.sum(), act)[0]
    g = torch.stack((g1, g2), 0).mean(0).detach()

    gn = g.reshape(B, -1).norm(dim=-1)
    noise = torch.randn(B, HORIZON, ACTION_DIM, device=DEV)
    nn_ = noise.reshape(B, -1).norm(dim=-1)

    print()
    print(f"  ||dQ/da||               mean {gn.mean():.5f}   max {gn.max():.5f}")
    print(f"  ||noise|| (what the denoiser emits today)  mean {nn_.mean():.5f}")
    print()
    for c in (1.0, 10.0, 50.0, 100.0):
        t = (c * gn).mean()
        print(f"  q_grad_coeff={c:6.1f} -> ||target|| mean {t:9.4f}   "
              f"ratio to noise {t / nn_.mean():8.4f}")
    print()
    print("  ratio ~1 would mean the new target is on the same scale as what the")
    print("  denoiser already outputs; far from 1 means the pretrained denoiser is")
    print("  being overwritten by an out-of-scale field on the very first update.")


if __name__ == "__main__":
    main()
