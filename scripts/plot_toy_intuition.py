"""Paper Fig (fig:intuition): WHY field-target RL on the drifting base — the picture.

Visualizes the HARD regime of the paper's two-regime toy (16-D actions, 60
critic points — toy_field_vs_gradient2.py's exact setup) in manifold
coordinates: x = position along the mode axis (a . e1), y = off-manifold
distance ||a_perp||. Everything is computed, nothing hand-drawn:

  (a) the critic's landscape on a 2-D slice through action space (mode axis x
      a fixed perpendicular direction): off the data manifold (y > 0) the
      scarce-data critic extrapolates a hallucinated high-Q region;
  (b) backprop fine-tuning (-Q + BC anchor, the DICE-RL residual actor): the
      action cloud rides the hallucinated gradient off-manifold; internal
      Q-hat soars while true reward collapses to zero;
  (c) the field-target update (clipped V_Q + V_BC + anchor, the Porygon
      primitive): bounded per-particle transport keeps the cloud on the
      manifold and shifts it toward the higher-reward mode.

Run on a GPU node: sbatch scripts/plot_toy_intuition.sbatch
Outputs <out>/toy_intuition.pdf + printed metrics for caption verification.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "run on GPU"
DEV = "cuda"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

SEED = 0
DIM = 16
N_CRITIC = 60
CRITIC_STEPS = 8000
STEPS = 1500
SNAPS = [0, 300, 1500]
S_MODE = 0.12

torch.manual_seed(SEED)
E1 = torch.zeros(DIM, device=DEV); E1[0] = 1.0
M1, M2 = -E1.clone(), E1.clone()
RW = 0.15 * math.sqrt(DIM / 2)
RADII = tuple(r * math.sqrt(DIM / 2) for r in (0.1, 0.3, 0.9))


def true_reward(a):
    r1 = torch.exp(-((a - M1) ** 2).sum(-1) / (2 * RW ** 2))
    r2 = 0.6 * torch.exp(-((a - M2) ** 2).sum(-1) / (2 * RW ** 2))
    return r1 + r2


def sample_data(n):
    pick = (torch.rand(n, device=DEV) < 0.5)[:, None]
    return torch.where(pick, M1, M2) + S_MODE * torch.randn(n, DIM, device=DEV)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))


def drift_field(query, pos, w_pos, neg, mask_pos_self=False):
    V = torch.zeros_like(query)
    K = len(query)
    for R in RADII:
        dq = torch.cdist(query, pos)
        if mask_pos_self and pos.shape == query.shape:
            dq = dq + torch.eye(K, device=DEV) * 1e6
        aff_p = (torch.softmax(-dq / R, 1) * torch.softmax(-dq / R, 0)
                 ).clamp_min(1e-9).sqrt() * w_pos[None, :]
        dn = torch.cdist(query, neg) + torch.eye(K, device=DEV) * 1e6
        aff_n = (torch.softmax(-dn / R, 1) * torch.softmax(-dn / R, 0)
                 ).clamp_min(1e-9).sqrt()
        V = V + (aff_p[:, :, None] * (pos[None] - query[:, None])).sum(1) \
              - (aff_n[:, :, None] * (neg[None] - query[:, None])).sum(1)
    return V


# ---- critic on scarce on-manifold data only (the hard regime) --------------
critic = mlp(DIM, 1).to(DEV)
da = sample_data(N_CRITIC)
dy = true_reward(da)[:, None] + 0.02 * torch.randn(N_CRITIC, 1, device=DEV)
opt = torch.optim.Adam(critic.parameters(), 1e-3)
for _ in range(CRITIC_STEPS):
    i = torch.randint(0, N_CRITIC, (64,), device=DEV)
    l = F.mse_loss(critic(da[i]), dy[i])
    opt.zero_grad(); l.backward(); opt.step()
for p in critic.parameters():
    p.requires_grad_(False)

# ---- drift-pretrained one-step base ----------------------------------------
base = mlp(DIM, DIM).to(DEV)
opt = torch.optim.Adam(base.parameters(), 1e-3)
for _ in range(3000):
    z = torch.randn(64, DIM, device=DEV)
    with torch.no_grad():
        cur = base(z)
        V = drift_field(cur, sample_data(64), torch.ones(64, device=DEV), cur)
        tgt = cur + 0.5 * V
    l = F.mse_loss(base(z), tgt)
    opt.zero_grad(); l.backward(); opt.step()
base_sd = {k: v.clone() for k, v in base.state_dict().items()}
for p in base.parameters():
    p.requires_grad_(False)


def fresh():
    m = mlp(DIM, DIM).to(DEV)
    m.load_state_dict(base_sd)
    return m


Z_VIS = torch.randn(48, DIM, device=DEV)  # fixed noise batch -> trackable cloud


def snapshot(pol, snaps, stats, t):
    with torch.no_grad():
        a = pol(Z_VIS)
        snaps[t] = a.cpu()
        stats[t] = (critic(a).mean().item(), true_reward(a).mean().item())


def run_grad_bc(lam=1.0):
    """Backprop -Q with BC anchor: the DICE-RL residual actor analog."""
    pol = fresh()
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    snaps, stats = {}, {}
    for t in range(STEPS + 1):
        if t in SNAPS:
            snapshot(pol, snaps, stats, t)
        if t == STEPS:
            break
        z = torch.randn(64, DIM, device=DEV)
        a = pol(z)
        loss = -critic(a).mean() + lam * F.mse_loss(a, base(z))
        opt.zero_grad(); loss.backward(); opt.step()
    return snaps, stats


def run_field():
    pol = fresh()
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, bc_step, clip_n, lam = 0.2, 0.2, 0.15, 1.0
    snaps, stats = {}, {}
    for t in range(STEPS + 1):
        if t in SNAPS:
            snapshot(pol, snaps, stats, t)
        if t == STEPS:
            break
        z = torch.randn(64, DIM, device=DEV)
        with torch.no_grad():
            cur = pol(z)
            q = critic(cur).squeeze(-1)
            adv = (q - q.mean()) / q.std().clamp_min(1e-6)
            w = torch.softmax(adv / 0.5, 0) * len(cur)
            VQ = drift_field(cur, cur, w, cur, mask_pos_self=True)
            bc = base(torch.randn(64, DIM, device=DEV))
            VBC = drift_field(cur, bc, torch.ones(64, device=DEV), cur)
            res = cur - base(z)
            delta = q_step * VQ + bc_step * VBC - lam * res
            dn = delta.norm(dim=-1, keepdim=True)
            tgt = cur + delta * torch.clamp(clip_n / (dn + 1e-8), max=1.0)
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    return snaps, stats


g_snaps, g_stats = run_grad_bc()
f_snaps, f_stats = run_field()

for name, st in [("backprop+anchor", g_stats), ("field", f_stats)]:
    for t in SNAPS:
        print(f"{name} t={t}: Q-hat={st[t][0]:.2f} true={st[t][1]:.3f}")


def manifold_coords(a):
    """x = position along the mode axis, y = off-manifold distance."""
    x = a[:, 0].numpy()
    y = a[:, 1:].norm(dim=-1).numpy()
    return x, y


# fixed perpendicular direction for the visualization slice
U = torch.zeros(DIM, device=DEV)
gy_dir = g_snaps[SNAPS[-1]].mean(0)[1:]        # where backprop actually went
if gy_dir.norm() > 1e-6:
    U[1:] = (gy_dir / gy_dir.norm()).to(DEV)
else:
    U[1] = 1.0

all_y = np.concatenate([manifold_coords(g_snaps[t])[1] for t in SNAPS] +
                       [manifold_coords(f_snaps[t])[1] for t in SNAPS])
all_x = np.concatenate([manifold_coords(g_snaps[t])[0] for t in SNAPS] +
                       [manifold_coords(f_snaps[t])[0] for t in SNAPS])
XL = max(2.6, float(np.abs(all_x).max()) * 1.1)
YL = max(2.2, float(all_y.max()) * 1.08)

gx = np.linspace(-XL, XL, 200)
gy = np.linspace(0, YL, 200)
GX, GY = np.meshgrid(gx, gy)
pts = (torch.tensor(GX.ravel(), dtype=torch.float32, device=DEV)[:, None] * E1
       + torch.tensor(GY.ravel(), dtype=torch.float32, device=DEV)[:, None] * U)
with torch.no_grad():
    Q = critic(pts).squeeze(-1).cpu().numpy().reshape(GX.shape)
    R = true_reward(pts).cpu().numpy().reshape(GX.shape)
dx, dy_ = manifold_coords(da.cpu())

C_BP, C_FLD, C_DATA = "#E69F00", "#0072B2", "#333333"
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), sharey=True)

ax = axes[0]
im = ax.contourf(GX, GY, Q, levels=24, cmap="RdBu_r")
ax.contour(GX, GY, R, levels=[0.2, 0.6], colors="k", linewidths=0.7,
           linestyles=":")
ax.scatter(dx, dy_, s=10, c=C_DATA, zorder=3, label="critic data")
qmax = np.unravel_index(Q.argmax(), Q.shape)
ax.scatter(GX[qmax], GY[qmax], marker="*", s=150, c="#D55E00",
           edgecolors="k", linewidths=0.5, zorder=4,
           label=r"$\arg\max \hat{Q}$ (slice)")
ax.set_title(rf"(a) critic fit on {N_CRITIC} on-manifold points", fontsize=10)
ax.set_ylabel(r"off-manifold distance $\|a_\perp\|$", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label(r"$\hat{Q}$", fontsize=9)

for ax, snaps, stats, color, title in [
    (axes[1], g_snaps, g_stats, C_BP,
     r"(b) backprop $-\hat{Q}$ + BC anchor"),
    (axes[2], f_snaps, f_stats, C_FLD,
     r"(c) field update (clipped, anchored)"),
]:
    ax.contourf(GX, GY, Q, levels=24, cmap="RdBu_r", alpha=0.35)
    ax.contour(GX, GY, R, levels=[0.2, 0.6], colors="k", linewidths=0.7,
               linestyles=":")
    ax.scatter(dx, dy_, s=8, c=C_DATA, alpha=0.45, zorder=2)
    for al, t in zip([0.25, 0.55, 1.0], SNAPS):
        sx, sy = manifold_coords(snaps[t])
        ax.scatter(sx, sy, s=13, c=color, alpha=al, zorder=3,
                   edgecolors="none")
    for i in range(0, len(Z_VIS), 6):
        xs = [manifold_coords(snaps[t])[0][i] for t in SNAPS]
        ys = [manifold_coords(snaps[t])[1][i] for t in SNAPS]
        ax.plot(xs, ys, color=color, lw=0.7, alpha=0.6, zorder=2)
    q0, r0 = stats[SNAPS[0]]
    qT, rT = stats[SNAPS[-1]]
    ax.set_title(title, fontsize=10)
    ax.text(0.03, 0.97,
            rf"$\hat{{Q}}$: {q0:.1f}$\to${qT:.0f}"
            "\n" rf"true reward: {r0:.2f}$\to${rT:.2f}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.5))

for ax in axes:
    ax.set_xlim(-XL, XL)
    ax.set_ylim(-0.03, YL)
    ax.set_xlabel(r"position along mode axis $a \cdot e_1$", fontsize=9)
    ax.tick_params(labelsize=7)

fig.tight_layout()
out = os.path.join(OUT, "toy_intuition.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
