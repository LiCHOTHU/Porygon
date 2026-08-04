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

C_BP, C_FLD, C_DATA = "#D55E00", "#0072B2", "#111111"
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True, sharex=True)
LEV = np.linspace(Q.min(), Q.max(), 12)

def _bg(ax):
    im = ax.contourf(GX, GY, Q, levels=LEV, cmap="Reds", alpha=0.55)
    ax.axhspan(-0.05, 1.0, color="#2ca02c", alpha=0.15, zorder=1)
    ax.axhline(1.0, color="#2ca02c", lw=1.3, ls="--", alpha=0.85, zorder=2)
    return im

def _data(ax, lab=True):
    ax.scatter(dx, dy_, s=60, c=C_DATA, marker="o", edgecolors="white",
               linewidths=0.9, zorder=6, label="expert actions" if lab else None)

def _cloud(ax, snaps, color):
    xs0, ys0 = manifold_coords(snaps[SNAPS[0]])
    xsT, ysT = manifold_coords(snaps[SNAPS[-1]])
    ax.scatter(xs0, ys0, s=50, facecolors="none", edgecolors="#555555",
               linewidths=1.1, zorder=5, label="policy before RL")
    for i in range(0, len(xs0), 3):
        ax.annotate("", xy=(xsT[i], ysT[i]), xytext=(xs0[i], ys0[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9,
                                    alpha=0.5, shrinkA=2, shrinkB=2), zorder=4)
    ax.scatter(xsT, ysT, s=80, c=color, marker="o", edgecolors="white",
               linewidths=0.9, zorder=7, label="policy after RL")

def _box(ax, stats, verdict, color):
    q0, r0 = stats[SNAPS[0]]; qT, rT = stats[SNAPS[-1]]
    txt = ("critic score: %.1f -> %.0f\n"
           "true success: %.0f%% -> %.0f%%  %s") % (q0, qT, r0*100, rT*100, verdict)
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9.0,
            ha="left", va="top", family="monospace", color=color,
            bbox=dict(fc="white", ec=color, lw=1.0, alpha=0.96, pad=4.0), zorder=10)

# --- (a) the critic's belief ---
ax = axes[0]
im = _bg(ax); _data(ax)
ax.annotate("", xy=(-2.6, YL * 0.92), xytext=(-2.6, YL * 0.22),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color="k"), zorder=8)
ax.text(-2.35, YL * 0.57, "the critic's score keeps\nrising the further you go\nfrom the data",
        fontsize=10, ha="left", va="center", fontweight="bold", zorder=9)
ax.annotate("all expert data is here", xy=(1.0, 0.7), xytext=(1.0, YL * 0.16),
            fontsize=10, ha="center", color="#1a6b1a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#1a6b1a"), zorder=9)
ax.set_title("(a) what the critic believes", fontsize=12.5, fontweight="bold")
ax.set_ylabel("distance away from expert data", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.96)

# --- (b) backprop ---
ax = axes[1]
_bg(ax); _data(ax, lab=False); _cloud(ax, g_snaps, C_BP)
ax.annotate("dragged far\noff the data", xy=(-1.3, YL * 0.70),
            xytext=(1.4, YL * 0.55), fontsize=10.5, ha="center",
            color=C_BP, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.6, color=C_BP), zorder=9)
_box(ax, g_stats, "FAILS", C_BP)
ax.set_title("(b) backpropagating the critic", fontsize=12.5, fontweight="bold")
ax.legend(loc="center right", fontsize=9, framealpha=0.96)

# --- (c) ours ---
ax = axes[2]
_bg(ax); _data(ax, lab=False); _cloud(ax, f_snaps, C_FLD)
ax.annotate("steps are capped:\nit stays on the data", xy=(0.5, 1.1),
            xytext=(1.3, YL * 0.55), fontsize=10.5, ha="center",
            color=C_FLD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.6, color=C_FLD), zorder=9)
_box(ax, f_stats, "HOLDS", C_FLD)
ax.set_title("(c) our bounded field update", fontsize=12.5, fontweight="bold")
ax.legend(loc="center right", fontsize=9, framealpha=0.96)

for ax in axes:
    ax.set_xlim(-XL, XL); ax.set_ylim(-0.05, YL * 1.02)
    ax.set_xlabel("action", fontsize=11)
    ax.tick_params(labelsize=9)
cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.022)
cb.set_label("value the critic predicts", fontsize=10)
out = os.path.join(OUT, "toy_intuition.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
