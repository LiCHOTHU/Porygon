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


def fresh(arm_seed=1234):
    """A copy of the pretrained base, with the RNG reset.

    Each arm is seeded identically so that adding or removing an unrelated
    measurement elsewhere in this script cannot shift the reported numbers --
    which it otherwise does, since every draw shares one global stream.
    """
    torch.manual_seed(arm_seed)
    m = mlp(DIM, DIM).to(DEV)
    m.load_state_dict(base_sd)
    return m


LOG_EVERY = 25


def dist_from_data(a):
    """Distance from the nearer expert mode, in per-dimension rms units."""
    d1 = (a - M1).norm(dim=-1)
    d2 = (a - M2).norm(dim=-1)
    return (torch.minimum(d1, d2) / math.sqrt(DIM)).mean().item()


E2 = torch.zeros(DIM, device=DEV); E2[1] = 1.0   # an off-manifold direction


@torch.no_grad()
def best_of_n(pol, N=16, trials=400):
    """True reward of the action the CRITIC picks out of N draws.

    This is the protocol the policy is actually deployed under (max_q_min
    best-of-N in the real experiments), so it is the fair way to compare arms:
    a single mean action is not what any of these methods ships.
    """
    tot = 0.0
    for _ in range(trials):
        z = torch.randn(N, DIM, device=DEV)
        a = pol(z)
        j = critic(a).squeeze(-1).argmax()
        tot += true_reward(a[j:j + 1]).item()
    return tot / trials


@torch.no_grad()
def cloud(pol, n=400):
    """The action distribution itself, projected to (mode axis, off-manifold axis).

    Figure 1 shows the distribution rather than its mean, because the claim is
    about where the policy's MASS ends up: our update should move mass from the
    weaker mode to the better one while staying on the data.
    """
    z = torch.randn(n, DIM, device=DEV)
    a = pol(z)
    return np.stack([(a @ E1).cpu().numpy(), (a @ E2).cpu().numpy()], 1)


@torch.no_grad()
def probe(pol):
    """Returns (critic score, true reward, distance from data, x, y).

    x = a . e1 is the axis joining the two reward modes; y = a . e2 is a
    direction the demonstrations never occupy. The (x, y) pair lets us draw
    the optimisation path through action space, which is where the failure is
    actually legible: the policy leaves the data along y.
    """
    z = torch.randn(256, DIM, device=DEV)
    a = pol(z)
    return (critic(a).mean().item(), true_reward(a).mean().item(), dist_from_data(a),
            (a @ E1).mean().item(), (a @ E2).mean().item())


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
    return np.array(hist), cloud(pol), best_of_n(pol)


REQ_AG = []       # same quantity for the unbounded arm


def run_action_gradient(step=0.2):
    """The action-gradient family's primitive (DIPO/QSM): move each sampled
    action along grad_a Q, then regress the policy onto the moved action.

    No clip, no anchor -- the improved action is wherever the gradient step
    lands. This is the arm the paper's thesis predicts should fail for the
    same reason backprop does, and it is what distinguishes our contribution
    from prior work in our own family, so Figure 1 must show it.
    """
    pol = fresh(); opt = torch.optim.Adam(pol.parameters(), 3e-4)
    hist = []
    for t in range(STEPS + 1):
        if t % LOG_EVERY == 0:
            hist.append((t,) + probe(pol))
        if t == STEPS:
            break
        z = torch.randn(64, DIM, device=DEV)
        with torch.no_grad():
            cur = pol(z)
        cur_g = cur.clone().requires_grad_(True)
        q = critic(cur_g).sum()
        g, = torch.autograd.grad(q, cur_g)
        with torch.no_grad():
            REQ_AG.append((step * g).norm(dim=-1).cpu().numpy())
            tgt = cur + step * g          # unbounded displacement
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    return np.array(hist), cloud(pol), best_of_n(pol)


REQ = []          # pre-clip displacement magnitudes requested by the critic


def run_config(clip, use_anchor, steps=STEPS):
    """The field update with each guard switched on or off independently.

    This is the experiment that identifies the mechanism. Bounding the step
    (clip) and restoring toward the base (dead-zone anchor) are separable, and
    only one of them turns out to matter.
    """
    pol = fresh(); opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, bc_step = 0.2, 0.2
    RHO = 0.05 * math.sqrt(DIM)
    for t in range(steps):
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
            if use_anchor:
                rn = res.norm(dim=-1, keepdim=True)
                gate = (1.0 - RHO / rn.clamp_min(1e-8)).clamp(min=0.0)
                delta = q_step * VQ + bc_step * VBC - 1.0 * gate * res
            else:
                delta = q_step * VQ + bc_step * VBC
            if clip is not None:
                dn = delta.norm(dim=-1, keepdim=True)
                delta = delta * torch.clamp(clip / (dn + 1e-8), max=1.0)
            tgt = cur + delta
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    _, tr, d, _, _ = probe(pol)
    return tr, d


def run_field():
    pol = fresh(); opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, bc_step, clip_n, lam = 0.2, 0.2, 0.15, 1.0
    RHO = 0.05 * math.sqrt(DIM)   # dead-zone radius, in the toy's action scale
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
            # Dead-zone anchor, matching the method (Eq. 4): the restoring pull
            # is SILENT while the residual is inside radius rho, and only then
            # pulls back. The earlier version applied it unconditionally, which
            # is the ablated variant our own experiments show falls below base.
            res = cur - base(z)
            rn = res.norm(dim=-1, keepdim=True)
            gate = (1.0 - RHO / rn.clamp_min(1e-8)).clamp(min=0.0)
            delta = q_step * VQ + bc_step * VBC - lam * gate * res
            dn = delta.norm(dim=-1, keepdim=True)
            REQ.append(dn.squeeze(-1).cpu().numpy())     # what was asked for
            tgt = cur + delta * torch.clamp(clip_n / (dn + 1e-8), max=1.0)
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    return np.array(hist), cloud(pol), best_of_n(pol)


H_BP,   CL_BP,   BON_BP   = run_backprop(0.0)
H_BPBC, CL_BPBC, BON_BPBC = run_backprop(1.0)
H_AG,   CL_AG,   BON_AG   = run_action_gradient()
H_FLD,  CL_FLD,  BON_FLD  = run_field()
_b = fresh()
CL_BASE = cloud(_b)
BON_BASE = best_of_n(_b)
BASE_R = probe(_b)[1]
ISO = {}
for nm, clip, anc in [("clip + anchor\n(ours)", 0.15, True),
                      ("clip only\n(no anchor)", 0.15, False),
                      ("anchor only\n(no clip)", None, True),
                      ("neither", None, False)]:
    ISO[nm] = run_config(clip, anc)
print("=== which guard prevents the escape? ===")
for k, (tr, d) in ISO.items():
    print(f"  {k.replace(chr(10),' '):28s} reward={tr:.3f}  distance={d:,.2f}")
print("=== deployment protocol (best-of-16 by critic) ===")
for nm, v in [("no RL", BON_BASE), ("backprop", BON_BP), ("backprop+BC", BON_BPBC),
              ("action-gradient", BON_AG), ("ours", BON_FLD)]:
    print(f"  {nm:18s} {v:.3f}")
for nm, H in [("backprop", H_BP), ("backprop+BC", H_BPBC),
              ("action-gradient (unbounded)", H_AG), ("ours (bounded)", H_FLD)]:
    print(f"{nm}: final true={H[-1,2]:.3f} critic={H[-1,1]:.2f} dist={H[-1,3]:.3f}")
print(f"base true={BASE_R:.3f}")

# ---------------------------------------------------------------------------
# Figure 1, three panels, one message each:
#   (a) what goes wrong, in action space
#   (b) WHY, and what we change: the distribution of requested step sizes has a
#       heavy tail; the cap truncates it. This is the paper's actual thesis and
#       is the panel that must be unmissable.
#   (c) what it buys, on real benchmarks.
# Deliberately sparse annotation: earlier drafts collided text with data.
# ---------------------------------------------------------------------------
C_BP, C_AG, C_FLD, C_BASE = "#D55E00", "#E69F00", "#0072B2", "#555555"
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

# ---- (a) action space -------------------------------------------------------
ax = axes[0]
LIM = 2.2
gx = torch.linspace(-LIM, LIM, 170, device=DEV)
gy = torch.linspace(-0.55, LIM, 170, device=DEV)
GX, GY = torch.meshgrid(gx, gy, indexing="ij")
grid = torch.zeros(GX.numel(), DIM, device=DEV)
grid[:, 0] = GX.reshape(-1); grid[:, 1] = GY.reshape(-1)
with torch.no_grad():
    TR = true_reward(grid).reshape(GX.shape).cpu().numpy()
    CR = critic(grid).reshape(GX.shape).cpu().numpy()
gxn, gyn = gx.cpu().numpy(), gy.cpu().numpy()
ax.contourf(gxn, gyn, TR.T, levels=14, cmap="Greens", alpha=0.85)
ax.contour(gxn, gyn, CR.T, levels=6, colors=C_BP, linewidths=0.9, alpha=0.6)
ax.scatter(CL_BASE[:, 0], CL_BASE[:, 1], s=10, c="0.3", alpha=0.45)
ax.scatter(CL_FLD[:, 0], CL_FLD[:, 1], s=12, c=C_FLD, alpha=0.75)
ax.annotate("", xy=(-1.55, 1.95), xytext=(-0.5, 0.28),
            arrowprops=dict(arrowstyle="-|>", lw=3.0, color=C_BP, alpha=0.9))
ax.text(-1.45, 2.06, "unbounded updates\nleave entirely", fontsize=10,
        color=C_BP, fontweight="bold", ha="center", va="bottom")
ax.scatter([-1, 1], [0, 0], s=[200, 130], marker="*", c="white",
           zorder=6, edgecolor="#0b5c0b", linewidth=1.6)
ax.text(-1, -0.44, "reward 1.0", ha="center", fontsize=10, color="#0b5c0b", fontweight="bold")
ax.text(1, -0.44, "reward 0.6", ha="center", fontsize=10, color="#3f8f3f", fontweight="bold")
ax.text(-2.05, 1.30, "grey = before RL\nblue = after ours", fontsize=9.5, va="top")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-0.55, LIM + 0.35)
ax.set_xlabel("action (axis joining the two modes)", fontsize=11)
ax.set_ylabel("action (direction with no data)", fontsize=11)
ax.set_title("(a) The policy either stays on the data\nor leaves it entirely",
             fontsize=12, fontweight="bold")

# ---- (b) THE MECHANISM: which guard actually matters -----------------------
# The two guards are separable, so we switch each off independently. Only the
# restoring anchor prevents the escape; bounding the step size does not, and
# on its own is no better than leaving the update unconstrained.
ax = axes[1]
labels = list(ISO.keys())
rew = [ISO[k][0] for k in labels]
dist = [ISO[k][1] for k in labels]
cols = [C_FLD, C_BP, C_FLD, C_BP]
hatch = ["", "", "//", ""]
bars = ax.bar(range(4), rew, color=cols, hatch=hatch, edgecolor="white", linewidth=1.4)
ax.axhline(BASE_R, color=C_BASE, ls=":", lw=1.8)
ax.text(3.45, BASE_R + 0.018, "starting policy", ha="right", fontsize=9, color=C_BASE)
for i_, r in enumerate(rew):
    ax.text(i_, r + 0.022, f"{r:.2f}", ha="center", fontsize=12, fontweight="bold")
# distance travelled goes under the tick label, so it never sits on a bar
ticklabels = [lab + "\n\n" + (f"travelled {d:,.0f}x" if d > 1 else f"travelled {d:.2f}x")
              for lab, d in zip(labels, dist)]
ax.set_xticks(range(4)); ax.set_xticklabels(ticklabels, fontsize=8.4)
ax.set_ylim(0, max(max(rew), BASE_R) * 1.30)
ax.set_ylabel("true task success", fontsize=11)
ax.set_title("(b) Which guard does the work?\nthe anchor alone matches the full method;\n"
             "the step bound alone does nothing",
             fontsize=12, fontweight="bold")

# ---- (c) the payoff on real benchmarks --------------------------------------
ax = axes[2]
groups = ["robomimic\nsquare", "LIBERO hard-8\n(8-task mean)"]
base_v, bp_v, ours_v = [0.382, 0.661], [0.880, 0.670], [0.924, 0.755]
xs = np.arange(2); w = 0.26
ax.bar(xs - w, base_v, w, color=C_BASE, label="no RL (start)")
ax.bar(xs, bp_v, w, color=C_BP, label="backprop $-Q$ (DICE-RL)")
ax.bar(xs + w, ours_v, w, color=C_FLD, label="ours (anchored update)")
for x, v in list(zip(xs - w, base_v)) + list(zip(xs, bp_v)) + list(zip(xs + w, ours_v)):
    ax.text(x, v + 0.018, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("task success", fontsize=11)
ax.set_title("(c) What it buys, on real tasks\nsame base, critic, data and budget",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper center", fontsize=8.8, framealpha=0.95, ncol=1)

fig.tight_layout(w_pad=2.4)
out = os.path.join(OUT, "toy_intuition.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
print("Figure 1 written. Panel (b) numbers:", {k.replace(chr(10), " "): (round(v[0], 3), round(v[1], 2)) for k, v in ISO.items()})
