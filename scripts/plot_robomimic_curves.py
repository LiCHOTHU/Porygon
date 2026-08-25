"""Paper Fig. 3: robomimic learning curves (square + can) from harness eval logs.

Parses "Eval at step N: reward=X" lines out of the official-harness job logs,
aggregates seeds per method (mean + min/max band), and renders the two-panel
figure used in iclr2026/sections/05_experiments.tex (fig:curves).

Usage:
    python scripts/plot_robomimic_curves.py [--out iclr2026/figures/robomimic_curves.pdf]

Regenerate after the s44 reruns finish; the SOURCES map below picks, per cell,
the log with the longest trajectory (pass --logdir to override).
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGDIR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs"
EVAL_RE = re.compile(r"Eval at step (\d+): reward=([\d.]+)")

# method -> task -> list of log-name globs (one per seed); the longest log wins per glob
SOURCES = {
    "square": {
        "B":  ["sq_swp_B_qstep05_*", "sq_conf_Bq05_s43_*", "sq_conf_Bq05_s44_*"],
        "A":  ["sq_ext_A_s42_*", "sq_ext_A_s43_*", "sq_ext_A_s44_*"],
        "FM": ["sq_ext_FM_s42_*", "sq_ext_FM_s43_*", "sq_ext_FM_s44_*"],
        "C":  ["sq_tabR_C_s42_*", "sq_tabR_C_s43_*", "sq_tabR_C_s44_*"],
        "T":  ["sq_tabR_T_s42_*", "sq_tabR_T_s43_*", "sq_tabR_T_s44_*"],
    },
    "can": {
        "B":  ["can_conf_Bq05_s42_*", "can_conf_Bq05_s43_*", "can_conf_Bq05_s44_*"],
        "A":  ["can_ext_A_s42_*", "can_ext_A_s43_*", "can_ext_A_s44_*"],
        "FM": ["can_ext_FM_s42_*", "can_ext_FM_s43_*", "can_ext_FM_s44_*"],
        "C":  ["can_tabR_C_s42_*", "can_tabR_C_s43_*", "can_tabR_C_s44_*"],
        "T":  ["can_tabR_T_s42_*", "can_tabR_T_s43_*", "can_tabR_T_s44_*"],
    },
}
BASE = {"square": 0.382, "can": 0.877}  # pretrained drift base, 300-ep evals
MATCHED_BUDGET = 20_000
XMAX = 40_000

# Okabe-Ito (CVD-safe); line style doubles as a second identity channel.
STYLE = {
    "B":  dict(color="#0072B2", ls="-",  lw=2.2, label="CAST"),
    "A":  dict(color="#E69F00", ls="--", lw=1.8, label="backprop actor"),
    "FM": dict(color="#009E73", ls="-.", lw=1.8, label="FM + DICE-RL"),
    "C":  dict(color="#CC79A7", ls=":",  lw=1.8, label="CAST (top-$k$)"),
    "T":  dict(color="#56B4E9", ls="-",  lw=1.2, label="CAST (tilted)"),
}


def read_curve(pattern, logdir):
    """Return {step: success} from the longest-trajectory log matching pattern."""
    best = {}
    for path in glob.glob(os.path.join(logdir, pattern + ".out")) or glob.glob(
        os.path.join(logdir, pattern)
    ):
        evals = {}
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = EVAL_RE.search(line)
                if m:
                    evals[int(m.group(1))] = float(m.group(2))
        if evals and (not best or max(evals) > max(best)):
            best = evals
    return best


def aggregate(curves, xmax):
    """Mean and min/max band on the union grid, per step where >=1 seed has data."""
    # step-0 evals are an init-time harness artifact (pre-warmup selection); drop them
    steps = sorted({s for c in curves for s in c if 0 < s <= xmax})
    mean, lo, hi = [], [], []
    for s in steps:
        vals = [c[s] for c in curves if s in c]
        mean.append(np.mean(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return np.array(steps), np.array(mean), np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=LOGDIR)
    ap.add_argument("--out", default="iclr2026/figures/robomimic_curves.pdf")
    args = ap.parse_args()

    fig, ax_ = plt.subplots(1, 1, figsize=(6.0, 3.6))
    axes = [ax_]
    TITLE = {"square": "robomimic square"}
    for ax, task in zip(axes, ["square"]):
        drawn = {}
        for method, patterns in SOURCES[task].items():
            if method not in ("A", "B"):   # only DICE-RL actor vs CAST actor
                continue
            curves = [c for c in (read_curve(p, args.logdir) for p in patterns) if c]
            if not curves:
                continue
            x, m, lo, hi = aggregate(curves, XMAX)
            drawn[method] = (x, m)
            st = STYLE[method]
            ax.plot(x / 1000, m, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3)
            ax.fill_between(x / 1000, lo, hi, color=st["color"], alpha=0.15, lw=0)

        base = BASE[task]
        # shade what post-training actually added, base -> CAST
        if "B" in drawn:
            x, m = drawn["B"]
            ax.fill_between(x / 1000, base, m, where=(m > base),
                            color="#0072B2", alpha=0.09, lw=0, zorder=1)
        ax.axhline(base, color="#444444", lw=1.1, ls=(0, (2, 2)), zorder=2)
        dy = 0.018 if task == "square" else -0.055
        ax.text(0.6, base + dy, f"frozen base, no RL = {base:.2f}",
                color="#444444", fontsize=8, ha="left", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))

        # the gain CAST delivers at the matched budget
        if "B" in drawn:
            x, m = drawn["B"]
            k = int(np.argmin(np.abs(x - MATCHED_BUDGET)))
            xb, yb = x[k] / 1000, m[k]
            ax.annotate("", xy=(xb, yb), xytext=(xb, base),
                        arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=1.3))
            ax.text(xb + 0.8, (yb + base) / 2, f"+{100*(yb-base):.0f} pts\nover the base",
                    color="#0072B2", fontsize=8.5, fontweight="bold",
                    ha="left", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))

        # speed: where CAST first reaches the backprop actor's matched-budget score
        if task == "square" and "B" in drawn and "A" in drawn:
            xa, ma = drawn["A"]; xb_, mb = drawn["B"]
            ka = int(np.argmin(np.abs(xa - MATCHED_BUDGET)))
            target = ma[ka]
            hit = np.where(mb >= target)[0]
            if len(hit) and xb_[hit[0]] < MATCHED_BUDGET:
                xs = xb_[hit[0]] / 1000
                ax.plot([xs], [target], marker="o", ms=5, color="#0072B2", zorder=5)
                ax.annotate(f"CAST reaches the backprop\nendpoint {(MATCHED_BUDGET-xb_[hit[0]])/1000:.0f}K steps early",
                            xy=(xs, target), xytext=(xs - 13.5, 0.10),
                            fontsize=7.5, color="#0072B2", zorder=6,
                            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9),
                            arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9))

        ax.axvline(MATCHED_BUDGET / 1000, color="#aaaaaa", lw=0.9, ls="--")
        ax.text(MATCHED_BUDGET / 1000 + 0.4, 0.03, "matched budget", color="#888888",
                fontsize=7.5, rotation=90, va="bottom")
        ax.set_title(TITLE[task], fontsize=10)
        ax.set_xlabel("training steps (K)")
        ax.set_xlim(0, XMAX / 1000)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("success rate (300 episodes)")
    handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[m].items() if k != "label"},
                          label=lab) for m, lab in
               (("A", "DICE-RL actor (backpropagate $-Q$)"),
                ("B", "CAST actor (bounded target displacement)"))]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
