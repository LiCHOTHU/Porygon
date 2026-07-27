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
    "B":  dict(color="#0072B2", ls="-",  lw=2.2, label="B: Porygon"),
    "A":  dict(color="#E69F00", ls="--", lw=1.8, label="A: DICE-RL actor"),
    "FM": dict(color="#009E73", ls="-.", lw=1.8, label="FM + DICE-RL"),
    "C":  dict(color="#CC79A7", ls=":",  lw=1.8, label="C: top-$k$ field"),
    "T":  dict(color="#56B4E9", ls="-",  lw=1.2, label="T: tilted field"),
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

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for ax, task in zip(axes, ["square", "can"]):
        for method, patterns in SOURCES[task].items():
            curves = [c for c in (read_curve(p, args.logdir) for p in patterns) if c]
            if not curves:
                continue
            x, m, lo, hi = aggregate(curves, XMAX)
            st = STYLE[method]
            ax.plot(x / 1000, m, color=st["color"], ls=st["ls"], lw=st["lw"])
            ax.fill_between(x / 1000, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.axhline(BASE[task], color="#777777", lw=1.0, ls=(0, (1, 3)))
        ax.text(XMAX / 1000 * 0.99, BASE[task] + 0.015, "drift base", color="#777777",
                ha="right", fontsize=8)
        ax.axvline(MATCHED_BUDGET / 1000, color="#aaaaaa", lw=0.8, ls="--")
        ax.text(MATCHED_BUDGET / 1000 + 0.5, 0.03, "matched budget", color="#888888",
                fontsize=8, rotation=90, va="bottom")
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("training steps (K)")
        ax.set_xlim(0, XMAX / 1000)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("success rate (300 ep)")
    handles = [plt.Line2D([], [], **{k: v for k, v in st.items() if k != "label"},
                          label=st["label"]) for st in STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
