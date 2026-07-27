"""Paper Fig (fig:libero-pertask): LIBERO hard-8 per-task grouped bars + tab:libero numbers.

Reads the powered-eval JSONs written by scripts/powered_eval_one.sbatch
(one per arm x training seed, 100 rollouts x 3 eval seeds each) plus the
BC-base row stored in powered_eval_2x2_drift_dice.json, pools the
3 train x 3 eval seed values per task, and renders grouped bars with 95% t-CIs.
Also prints the hard-8 mean per arm (the tab:libero row) to stdout.

Usage:
    python scripts/plot_libero_pertask.py [--out iclr2026/figures/libero_pertask.pdf]

Arms missing their JSONs (evals still running) are skipped with a warning, so
the script can be re-run as results land.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CEDAR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
BC_JSON = os.path.join(CEDAR, "powered_eval_2x2_drift_dice.json")

# display order: label -> (json LABEL prefix, style); seeds 10000/10001/10002
ARMS = {
    "BC base":   dict(prefix=None, color="#777777"),
    "backprop actor": dict(prefix="A_residual", color="#E69F00"),
    "FM + DICE-RL": dict(prefix="FMDICE", color="#009E73"),
    "Porygon": dict(prefix="B_grad", color="#0072B2"),
    "Porygon (top-$k$)": dict(prefix="C_zeroth", color="#CC79A7"),
    "Porygon (tilted)":  dict(prefix="T_tilted_gf", color="#56B4E9"),
}
SEEDS = [10000, 10001, 10002]
TCRIT = {3: 4.303, 6: 2.571, 9: 2.306}  # two-sided 95%, df = n-1

SHORT = {  # long env ids -> compact axis labels
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it": "K1 bowl→drawer",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it": "K3 stove+pan",
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet": "K5 ketchup",
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket": "L2 juice",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate": "L5 red mug",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": "S1 book front",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": "S1 book right",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": "S3 book front",
}


def load_target(path):
    """Return {env: [eval-seed successes]} plus pooled overall values."""
    with open(path) as fh:
        d = json.load(fh)
    row = next(iter(d.values())) if "per_env" not in d else d
    return {env: list(v["seeds"]) for env, v in row["per_env"].items()}, list(
        row["success_rates"]
    )


def collect(prefix):
    """Pool per-task eval-seed values across the arm's training seeds."""
    per_task, overall = {}, []
    for s in SEEDS:
        matches = glob.glob(
            os.path.join(CEDAR, f"powered_eval_one_hard8ff_{prefix}_s{s}.json")
        )
        if not matches:
            print(f"  [missing] {prefix}_s{s} (eval not finished?)")
            continue
        envs, rates = load_target(matches[0])
        overall.extend(rates)
        for env, vals in envs.items():
            per_task.setdefault(env, []).extend(vals)
    return per_task, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="iclr2026/figures/libero_pertask.pdf")
    args = ap.parse_args()

    data = {}  # label -> (per_task {env: values}, overall values)
    with open(BC_JSON) as fh:
        bc = json.load(fh)["BC"]
    data["BC base"] = (
        {env: list(v["seeds"]) for env, v in bc["per_env"].items()},
        list(bc["success_rates"]),
    )
    for label, spec in ARMS.items():
        if spec["prefix"] is None:
            continue
        print(f"{label}:")
        data[label] = collect(spec["prefix"])

    print("\n== tab:libero hard-8 means (pooled train x eval seeds) ==")
    for label, (_, overall) in data.items():
        if not overall:
            print(f"  {label}: ---")
            continue
        n = len(overall)
        m, sd = np.mean(overall), np.std(overall, ddof=1)
        ci = TCRIT.get(n, 2.0) * sd / np.sqrt(n)
        print(f"  {label}: {m:.3f} +- {ci:.3f} (n={n})")

    tasks = sorted(next(pt for pt, _ in data.values() if pt))
    present = [l for l in ARMS if data.get(l, ({}, []))[0]]
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    x = np.arange(len(tasks))
    for i, label in enumerate(present):
        per_task = data[label][0]
        means, errs = [], []
        for t in tasks:
            vals = per_task.get(t, [])
            if vals:
                n = len(vals)
                means.append(np.mean(vals))
                errs.append(TCRIT.get(n, 2.0) * np.std(vals, ddof=1) / np.sqrt(n))
            else:
                means.append(np.nan)
                errs.append(0)
        ax.bar(x + (i - len(present) / 2 + 0.5) * width, means, width,
               yerr=errs, error_kw=dict(lw=0.8, capsize=1.5),
               color=ARMS[label]["color"], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(t, t[:14]) for t in tasks], fontsize=7.5,
                       rotation=20, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(present), frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
