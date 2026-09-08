"""Figure 1: why critic-guided post-training leaves the data manifold, and what
a bounded update does about it.

Not a cartoon. Panels (a)-(c) come from a real fitted critic: a small MLP is
trained on (action, true-reward) pairs sampled ONLY from a narrow demonstration
band, exactly the situation of post-training a policy cloned from few demos.
Everything shown -- the critic's error off the band, the path the unconstrained
update takes, where the bounded update stops -- is computed, not drawn.
Panel (d) is measured from the paper's own training runs.
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

torch.manual_seed(0); np.random.seed(0)
C_DEMO, C_BAD, C_GOOD, C_INK, C_MUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

# ---- the manifold: a curved band of feasible actions ----
def band(x):            # the demonstrations lie along this curve
    return 0.55 * x ** 2 - 0.25

# True reward = quality along the band  x  a factor that DECAYS off it.
# Leaving the band is genuinely bad (the robot misses the slot), but every
# demonstration sits ON the band, so a critic fit to demos never observes
# the decay -- it has no data telling it that off-band is worse.
def true_r(a):
    x, y = a[..., 0], a[..., 1]
    along  = np.exp(-((x - 0.45) ** 2) / 0.30)        # better toward +x
    onband = np.exp(-((y - band(x)) ** 2) / 0.030)    # collapses off the band
    return along * onband

# FEW demonstrations -- the regime this paper is about. With hundreds of
# demos the critic learns the band and its gradient is trustworthy; with a
# handful it extrapolates, and that is where post-training actually lives.
N_DEMO = 25
rng = np.random.default_rng(0)
t = rng.uniform(-1, 1, N_DEMO)
demo = np.stack([t, band(t) + 0.04 * rng.standard_normal(N_DEMO)], 1)

# ---- fit a critic ONLY on demo actions ----
net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
X = torch.tensor(demo, dtype=torch.float32); Y = torch.tensor(true_r(demo), dtype=torch.float32)[:, None]
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for _ in range(3000):
    opt.zero_grad(); ((net(X) - Y) ** 2).mean().backward(); opt.step()

g = np.linspace(-1.6, 1.6, 220)
GX, GY = np.meshgrid(g, g); G = np.stack([GX, GY], -1)
with torch.no_grad():
    Qhat = net(torch.tensor(G.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(GX.shape)
Rtrue = true_r(G)

def qgrad(a):
    x = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    net(x[None]).backward()
    return x.grad.numpy()

# ---- roll the two updates from the same start ----
start = demo[np.argmin(np.abs(demo[:, 0] + 0.60))]
def roll(bounded, n=26, step=0.09, rho=0.14):
    p = start.copy(); path = [p.copy()]
    for _ in range(n):
        d = qgrad(p); d = d / (np.linalg.norm(d) + 1e-9)
        p = p + step * d
        if bounded:                                   # anchor to the FROZEN base
            off = p - start; r = np.linalg.norm(off)
            if r > rho: p = start + off * (rho / r)
        path.append(p.copy())
    return np.array(path)
free, held = roll(False), roll(True)

fig = plt.figure(figsize=(13.6, 4.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.28)

def frame(ax, title, sub):
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#d9d8d4")
    # title on its own line, subtitle on the line below it -- previously both
    # were drawn at the same height and collided into unreadable overprint
    ax.set_title(title, fontsize=11, color=C_INK, pad=24, loc="left", fontweight="bold")
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.2, color=C_MUT, va="bottom")

# (a) where the critic is trustworthy
ax = fig.add_subplot(gs[0, 0])
ax.contourf(GX, GY, Rtrue, levels=18, cmap="YlOrBr", alpha=.45)
ax.scatter(demo[:, 0], demo[:, 1], s=26, c=C_DEMO, alpha=.9, linewidths=0, zorder=5)
frame(ax, "(a) post-training starts here",
      "policy cloned from %d demos; the critic sees\nonly these (blue), all on the band" % N_DEMO)
ax.text(-1.5, 1.22, "true reward", fontsize=8, color=C_MUT)
ax.text(-1.5, -1.05, "demonstrations", fontsize=8, color=C_DEMO, fontweight="bold")

# (b) the critic's belief -- accurate on the band, fiction off it
ax = fig.add_subplot(gs[0, 1])
im = ax.contourf(GX, GY, Qhat, levels=18, cmap="YlOrBr", alpha=.45)
err = np.abs(Qhat - Rtrue)
ax.contour(GX, GY, err, levels=[.25], colors=[C_BAD], linewidths=1.6, linestyles="--")
ax.scatter(demo[:, 0], demo[:, 1], s=22, c=C_DEMO, alpha=.75, linewidths=0, zorder=5)
frame(ax, "(b) what the critic believes",
      "learned $\\hat{Q}$; dashed line = where it stops\nmatching reality")
ax.text(-1.65, -1.62, "outside the dashed line the critic is\nextrapolating: no data, no constraint",
        fontsize=7.8, color=C_BAD, ha="left", va="bottom")

# (c) the two updates
ax = fig.add_subplot(gs[0, 2])
ax.contourf(GX, GY, Rtrue, levels=18, cmap="YlOrBr", alpha=.30)
ax.scatter(demo[:, 0], demo[:, 1], s=22, c=C_DEMO, alpha=.65, linewidths=0, zorder=5)
ax.add_patch(Circle(start, .14, fill=False, ec=C_GOOD, lw=1.4, ls=":"))
ax.plot(free[:, 0], free[:, 1], "-", c=C_BAD, lw=2.0, zorder=4)
ax.plot(held[:, 0], held[:, 1], "-", c=C_GOOD, lw=2.4, zorder=5)
ax.scatter(*start, s=46, c=C_INK, zorder=6, marker="o")
ax.scatter(*free[-1], s=52, c=C_BAD, zorder=6, marker="X")
ax.scatter(*held[-1], s=52, c=C_GOOD, zorder=6, marker="*")
frame(ax, "(c) same critic, two updates",
      "both follow $\\nabla_a\\hat{Q}$; only one keeps\nthe policy where the critic is valid")
r0, rf, rh = true_r(start), true_r(free[-1]), true_r(held[-1])
import torch as _t
with _t.no_grad():
    qf = float(net(_t.tensor(free[-1], dtype=_t.float32)[None]))
ax.text(free[-1, 0] - .05, free[-1, 1] - .22,
        "unconstrained\ncritic says $\\hat{Q}$=%.2f\ntrue reward %.3f" % (qf, rf),
        fontsize=7.8, color=C_BAD, ha="center", va="top", fontweight="bold")
ax.text(held[-1, 0] + .16, held[-1, 1] + .02,
        "bounded (ours)\ntrue reward %.3f" % rh, fontsize=7.8, color=C_GOOD, fontweight="bold")
ax.text(start[0] - .05, start[1] - .34, "frozen base %.3f" % r0,
        fontsize=7.8, color=C_INK, ha="center")
print(f"PANEL C: base {r0:.3f} -> bounded {rh:.3f} (x{rh/max(r0,1e-9):.1f}) | unconstrained {rf:.3f}, critic claims {qf:.2f}")

# (d) the same thing, measured in the real system
ax = fig.add_subplot(gs[0, 3])
# Real logged residual norms from the two LIBERO red-mug (t65) training runs --
# the anchored recipe and its no-anchor ablation. Not a schematic: the earlier
# version of this panel interpolated two endpoints and disagreed with the data
# (the anchored run does not ride the radius; it sits an order of magnitude
# inside it).
import json as _json
_ser = _json.load(open("/storage/scratch1/8/lwang831/fig1d_series.json"))
d_free = np.array(_ser["field_st_BNONE_t65_s10000"])
d_held = np.array(_ser["field_st_B_t65_s10000"])
ax.semilogy(np.arange(len(d_free)), d_free, c=C_BAD, lw=2.2, label="no base anchor")
ax.semilogy(np.arange(len(d_held)), d_held, c=C_GOOD, lw=2.4, label="bounded (ours)")
ax.axhline(0.05, color=C_MUT, lw=.9, ls=(0, (1, 2)))
ax.text(1, .062, "dead-zone radius $\\rho=0.05$", fontsize=7.6, color=C_MUT)
ax.set_xlabel("logged field step", fontsize=9, color=C_MUT)
ax.set_ylabel("residual norm (rms)", fontsize=9, color=C_MUT)
ax.tick_params(labelsize=8, colors=C_MUT)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d4")
ax.set_title("(d) measured on the red-mug task", fontsize=11, color=C_INK, pad=24, loc="left", fontweight="bold")
ax.text(0, 1.03, "logged residual norm during real training\n(log scale)", transform=ax.transAxes,
        fontsize=8.2, color=C_MUT, va="bottom")
ax.text(20, 8.5, "success 0.000", fontsize=8.4, color=C_BAD, fontweight="bold")
ax.text(13, .0075, "success 0.600, inside the dead zone", fontsize=8.4, color=C_GOOD, fontweight="bold")
fig.savefig("iclr2026/figures/fig1_manifold.pdf", bbox_inches="tight", dpi=300)
print("wrote iclr2026/figures/fig1_manifold.pdf")
print(f"unconstrained endpoint |a-a0| = {np.linalg.norm(free[-1]-start):.3f}")
print(f"bounded      endpoint |a-a0| = {np.linalg.norm(held[-1]-start):.3f}  (rho=0.14)")
print(f"true reward  at unconstrained end = {true_r(free[-1]):.3f}   at bounded end = {true_r(held[-1]):.3f}")
print(f"critic Qhat  at unconstrained end = {float(net(torch.tensor(free[-1],dtype=torch.float32)[None])):.3f}")
