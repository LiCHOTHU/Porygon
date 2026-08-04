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


LOG_EVERY = 25


def dist_from_data(a):
    """Distance from the nearer expert mode, in per-dimension rms units."""
    d1 = (a - M1).norm(dim=-1)
    d2 = (a - M2).norm(dim=-1)
    return (torch.minimum(d1, d2) / math.sqrt(DIM)).mean().item()


@torch.no_grad()
def probe(pol):
    z = torch.randn(256, DIM, device=DEV)
    a = pol(z)
    return (critic(a).mean().item(), true_reward(a).mean().item(), dist_from_data(a))


def run_backprop(bc_lambda):
    pol = fresh(); opt = torch.optim.Adam(pol.parameters(), 3e-4)
    hist = []
    for t in range(STEPS + 1):
        if t % LOG_EVERY == 0:
            hist.append((t,) + probe(pol))
        if t == STEPS:
            break
        z = torch.randn(64, DIM, device=DEV)
        a = pol(z)
        loss = -critic(a).mean() + bc_lambda * F.mse_loss(a, base(z))
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(hist)


def run_field():
    pol = fresh(); opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, bc_step, clip_n, lam = 0.2, 0.2, 0.15, 1.0
    hist = []
    for t in range(STEPS + 1):
        if t % LOG_EVERY == 0:
            hist.append((t,) + probe(pol))
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
            delta = q_step * VQ + bc_step * VBC - lam * (cur - base(z))
            dn = delta.norm(dim=-1, keepdim=True)
            tgt = cur + delta * torch.clamp(clip_n / (dn + 1e-8), max=1.0)
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    return np.array(hist)


H_BP   = run_backprop(0.0)
H_BPBC = run_backprop(1.0)
H_FLD  = run_field()
BASE_R = probe(fresh())[1]
for nm, H in [("backprop", H_BP), ("backprop+BC", H_BPBC), ("field", H_FLD)]:
    print(f"{nm}: final true={H[-1,2]:.3f} critic={H[-1,1]:.2f} dist={H[-1,3]:.3f}")
print(f"base true={BASE_R:.3f}")

C_BP, C_BC, C_FLD, C_BASE = "#D55E00", "#CC79A7", "#0072B2", "#666666"
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

# (a) the delusion: critic's belief vs reality, for the backpropagated update
ax = axes[0]
ax.plot(H_BP[:, 0], H_BP[:, 2], color="k", lw=2.6, label="what it actually solves")
ax.set_ylim(-0.03, 0.75); ax.set_ylabel("true success rate", fontsize=11)
ax.set_xlabel("RL update", fontsize=11)
ax2 = ax.twinx()
ax2.plot(H_BP[:, 0], H_BP[:, 1], color=C_BP, lw=2.6, ls="--",
         label="what the critic thinks it is worth")
ax2.set_ylabel("critic's own score", color=C_BP, fontsize=11)
ax2.tick_params(axis="y", colors=C_BP)
ax.annotate("reality collapses", xy=(400, 0.05), xytext=(620, 0.34),
            fontsize=10.5, fontweight="bold", color="k",
            arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))
ax2.annotate("the critic is\ndelighted", xy=(1100, H_BP[-1, 1] * 0.93),
             xytext=(430, H_BP[-1, 1] * 0.55), fontsize=10.5, fontweight="bold",
             color=C_BP, arrowprops=dict(arrowstyle="->", lw=1.5, color=C_BP))
ax.set_title("(a) backpropagating a critic:\nit optimises the score, not the task",
             fontsize=11.5, fontweight="bold")

# (b) the cause: the policy walks off the data
ax = axes[1]
ax.set_yscale("log")
ax.axhspan(0.01, 0.2, color="#2ca02c", alpha=0.18)
ax.text(760, 0.028, "where the expert's actions are", ha="center", fontsize=10,
        color="#1a6b1a", fontweight="bold")
ax.plot(H_BP[:, 0], H_BP[:, 3], color=C_BP, lw=2.6, label="backprop $-Q$")
ax.plot(H_BPBC[:, 0], H_BPBC[:, 3], color=C_BC, lw=2.4, ls="--",
        label="backprop $-Q$ + BC penalty")
ax.plot(H_FLD[:, 0], H_FLD[:, 3], color=C_FLD, lw=2.8, label="ours (bounded field)")
ax.set_xlabel("RL update", fontsize=11)
ax.set_ylabel("how far the policy has moved\nfrom expert data (log scale)", fontsize=11)
ax.set_ylim(0.01, 5e4)
for H, c, lab in [(H_BP, C_BP, f"{H_BP[-1,3]:,.0f}$\\times$"),
                  (H_BPBC, C_BC, f"{H_BPBC[-1,3]:.1f}$\\times$"),
                  (H_FLD, C_FLD, f"{H_FLD[-1,3]:.2f}$\\times$")]:
    ax.text(1560, H[-1, 3], lab, color=c, fontsize=10, fontweight="bold", va="center")
ax.set_title("(b) why: the update size is set by\nthe critic, so the policy escapes",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

# (c) the outcome
ax = axes[2]
names = ["no RL\n(base)", "backprop\n$-Q$", "backprop\n+ BC penalty", "ours"]
vals = [BASE_R, H_BP[-1, 2], H_BPBC[-1, 2], H_FLD[-1, 2]]
cols = [C_BASE, C_BP, C_BC, C_FLD]
bars = ax.bar(range(4), vals, color=cols, edgecolor="white", linewidth=1.2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.018, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(BASE_R, color=C_BASE, ls=":", lw=1.4)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, max(vals) * 1.28); ax.set_ylabel("true success rate", fontsize=11)
ax.set_title("(c) the outcome: a penalty does not\nrescue it, a bounded step does",
             fontsize=11.5, fontweight="bold")

fig.tight_layout()
out = os.path.join(OUT, "toy_intuition.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
