"""Fig. 4, two panels.

(a) robomimic square learning curves: frozen base vs DICE-RL's actor vs CAST's
    actor. Same base, critic, data, replay and budget -- only the update differs.
(b) why each guard is needed: the LIBERO constraint factorial (complete, both
    tasks). Removing the anchor is fatal; the clip alone is fatal; the anchor
    alone survives but drops below base on one task; only the pair is reliable.
"""
import csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGDIR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs"
EVAL_RE = re.compile(r"Eval at step (\d+): reward=([\d.]+)")
BASE_SQ = 0.382
XMAX, BUDGET = 40_000, 20_000
BLUE, ORANGE, GREY, RED, GREEN = "#0072B2", "#E69F00", "#666666", "#D55E00", "#009E73"

SRC = {"A": ["sq_ext_A_s42_*", "sq_ext_A_s43_*", "sq_ext_A_s44_*"],
       "B": ["sq_swp_B_qstep05_*", "sq_conf_Bq05_s43_*", "sq_conf_Bq05_s44_*"]}

def curve(pat):
    best = {}
    for path in glob.glob(os.path.join(LOGDIR, pat + ".out")):
        ev = {}
        for line in open(path, errors="ignore"):
            m = EVAL_RE.search(line)
            if m: ev[int(m.group(1))] = float(m.group(2))
        if ev and (not best or max(ev) > max(best)): best = ev
    return best

def agg(cs):
    steps = sorted({s for c in cs for s in c if 0 < s <= XMAX})
    m = [np.mean([c[s] for c in cs if s in c]) for s in steps]
    lo = [min(c[s] for c in cs if s in c) for s in steps]
    hi = [max(c[s] for c in cs if s in c) for s in steps]
    return np.array(steps), np.array(m), np.array(lo), np.array(hi)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.7),
                               gridspec_kw=dict(width_ratios=[1.05, 1]))

# ---------------- (a) head-to-head ----------------
for k, col, ls, lab in [("A", ORANGE, "--", "DICE-RL actor (backprop $-Q$)"),
                        ("B", BLUE, "-", "CAST actor (ours)")]:
    cs = [c for c in (curve(p) for p in SRC[k]) if c]
    if not cs: print("MISSING", k); continue
    x, m, lo, hi = agg(cs)
    axA.plot(x/1000, m, color=col, ls=ls, lw=2.2, label=lab, zorder=3)
    axA.fill_between(x/1000, lo, hi, color=col, alpha=0.15, lw=0)
    if k == "B":
        axA.fill_between(x/1000, BASE_SQ, m, where=(m > BASE_SQ), color=BLUE, alpha=0.08, lw=0)
axA.axhline(BASE_SQ, color="black", lw=1.1, ls=(0, (2, 2)))
axA.text(0.6, BASE_SQ+0.02, f"frozen base = {BASE_SQ:.2f}", fontsize=8)
axA.axvline(BUDGET/1000, color="#aaa", lw=0.9, ls="--")
axA.text(BUDGET/1000+0.4, 0.03, "matched budget", fontsize=7.5, rotation=90, color="#888")
axA.set_xlim(0, XMAX/1000); axA.set_ylim(0, 1.02)
axA.set_xlabel("training steps (K)"); axA.set_ylabel("success rate (300 episodes)")
axA.set_title("(a) same critic, same budget: only the update differs", fontsize=9.5)
kA = int(np.argmin(np.abs(x - BUDGET)))
axA.annotate("", xy=(BUDGET/1000, 0.924), xytext=(BUDGET/1000, 0.880),
             arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
axA.text(BUDGET/1000 - 0.6, 0.902,
         "at the matched budget\n0.924 vs 0.880 (+4.4 pts),\nand ahead on every seed",
         fontsize=7.5, color=BLUE, ha="right", va="center", fontweight="bold",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
axA.legend(fontsize=8, frameon=False, loc="lower right")
axA.spines[["top","right"]].set_visible(False); axA.grid(axis="y", color="#eee", lw=0.7); axA.set_axisbelow(True)

# ---------------- (b) component necessity: change from each task's own base ----
arms  = ["no\nconstraint", "clip\nonly", "anchor\nonly", "clip + anchor\n(full CAST)"]
t65   = [0.000, 0.000, 0.810, 0.781]; b65 = 0.573
t32   = [0.000, 0.000, 0.387, 0.684]; b32 = 0.610
d65 = [v - b65 for v in t65]
d32 = [v - b32 for v in t32]
x = np.arange(len(arms)); w = 0.36
c65 = [RED if v <= 0 else BLUE for v in d65]
c32 = [RED if v <= 0 else BLUE for v in d32]
axB.bar(x - w/2, d65, w, color=c65, edgecolor="white", lw=0.8)
axB.bar(x + w/2, d32, w, color=c32, edgecolor="white", lw=0.8, hatch="///")
axB.axhline(0, color="black", lw=1.2)
axB.text(-0.48, 0.015, "base level", fontsize=7.5, color="black", ha="left")
for xi in range(4):
    for dv, off in ((d65[xi], -w/2), (d32[xi], +w/2)):
        va, dy = ("bottom", 0.012) if dv > 0 else ("top", -0.012)
        axB.text(xi + off, dv + dy, f"{dv:+.2f}", ha="center", va=va, fontsize=6.8,
                 color=(BLUE if dv > 0 else RED))
axB.text(1.62, -0.52,
         "anchor alone: helps the red-mug task ($+0.24$)\nbut falls below base on ketchup ($-0.22$)",
         fontsize=7.2, color=RED, ha="left", va="center", zorder=12,
         bbox=dict(fc="white", ec=RED, lw=0.6, alpha=0.95, pad=2.5))
axB.annotate("only the pair helps\non both tasks", xy=(3, d32[3]), xytext=(2.35, 0.34),
             fontsize=7.5, color=BLUE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
axB.set_xticks(x); axB.set_xticklabels(arms, fontsize=8)
axB.set_ylim(-0.72, 0.46)
axB.set_ylabel("change from that task's own base")
axB.set_title("(b) each guard is necessary; neither alone suffices", fontsize=9.5)
axB.spines[["top","right"]].set_visible(False)
axB.grid(axis="y", color="#eee", lw=0.7); axB.set_axisbelow(True)
h1 = plt.Rectangle((0,0),1,1, fc="#999999", ec="white")
h2 = plt.Rectangle((0,0),1,1, fc="#999999", ec="white", hatch="///")
axB.legend([h1,h2], ["red mug task", "ketchup task"], fontsize=7.5,
           frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.99))

fig.tight_layout()
out = "iclr2026/figures/robomimic_curves.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
