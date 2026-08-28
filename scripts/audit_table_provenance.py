"""Reconcile every number in the paper's tables against the measurements on disk.

Two questions, asked separately:

  (1) PROVENANCE -- for each numeric cell in each table, does a stored
      evaluation exist that produces it? Cells with no source are the ones
      that have bitten us (Table 5's full-CAST row read 0.781 for weeks; the
      matched arm actually gives 0.600).

  (2) WASTE -- which trained checkpoints have never been evaluated? Six times
      in one day a finished run was holding up a table cell because the cheap
      30-minute evaluation step was skipped.

Value-matching is a weak oracle: a cell could coincide with an unrelated
measurement. It is still enough to separate "has a plausible source" from
"has none at all", which is the distinction that matters. The durable fix is
for each cell to record the run that produced it; this script is the stopgap.

Usage: python scripts/audit_table_provenance.py [--tex <file>]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

SCRATCH = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
DICE_LOG = f"{SCRATCH}/dice_rl_official/log_dir"
EXP_DICE = f"{SCRATCH}/imitation/experiments_dice/libero/libero_90"
TEX = "/storage/home/hcoda1/8/lwang831/workspace/imitation/iclr2026/sections/05_experiments.tex"

CELL = re.compile(r"(?<![\d.])(0\.\d{3})(?![\d])")


def load_measurements():
    """Every success number we can find, mapped value -> [source, ...]."""
    seen = defaultdict(list)

    # LIBERO powered evaluations (100 rollouts x 3 seeds)
    for f in glob.glob(f"{SCRATCH}/powered_eval*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if k == "_config" or not isinstance(v, dict) or "mean" not in v:
                continue
            seen[round(float(v["mean"]), 3)].append(f"{k} [{os.path.basename(f)}]")
            for s in v.get("success_rates", []):
                seen[round(float(s), 3)].append(f"{k}:seed [{os.path.basename(f)}]")

    # robomimic evaluation CSVs (300 episodes)
    for f in glob.glob(f"{DICE_LOG}/**/evaluation_results.csv", recursive=True):
        try:
            rows = [r for r in csv.DictReader(open(f)) if r.get("eval_type") == "finetuned"]
        except Exception:
            continue
        sr = [float(r["success_rate"]) for r in rows if r.get("success_rate")]
        if not sr:
            continue
        run = os.path.basename(os.path.dirname(f))
        for v in sr:
            seen[round(v, 3)].append(f"{run}:point")
        for w in (3, 5):  # last-w-checkpoint averages, the reported protocol
            if len(sr) >= 1:
                seen[round(sum(sr[-w:]) / len(sr[-w:]), 3)].append(f"{run}:last{w}")
    return seen


def tables(tex):
    """Split the section into (label, body) per table."""
    out = []
    for blk in tex.split("\\begin{table}")[1:]:
        body = blk.split("\\end{table}")[0]
        m = re.search(r"\\label\{(tab:[^}]+)\}", body)
        out.append((m.group(1) if m else "<unlabelled>", body))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=TEX)
    a = ap.parse_args()

    seen = load_measurements()
    print(f"measurements indexed from disk: {sum(len(v) for v in seen.values())} "
          f"({len(seen)} distinct values)\n")

    tex = open(a.tex).read()
    print(f"{'table':22s} {'cells':>6s} {'sourced':>8s} {'orphan':>7s}   orphan values")
    print("-" * 96)
    tot = orph_tot = 0
    for label, body in tables(tex):
        rows = [l for l in body.splitlines()
                if "&" in l and not l.strip().startswith("%")]
        cells = []
        for line in rows:
            cells += [round(float(x), 3) for x in CELL.findall(line)]
        # bases/deltas below 0.30 are mostly deltas, not measured cells
        cells = [c for c in cells if c >= 0.30]
        orphans = sorted({c for c in cells if c not in seen})
        tot += len(cells)
        orph_tot += len(orphans)
        shown = ", ".join(f"{o:.3f}" for o in orphans[:8]) + (" ..." if len(orphans) > 8 else "")
        print(f"{label:22s} {len(cells):6d} {len(cells)-len(orphans):8d} {len(orphans):7d}   {shown}")
    print("-" * 96)
    print(f"{'TOTAL':22s} {tot:6d} {tot-orph_tot:8d} {orph_tot:7d}\n")

    # ---- (2) trained but never evaluated ----
    print("trained checkpoints with no stored evaluation:")
    evaluated = set()
    for f in glob.glob(f"{SCRATCH}/powered_eval*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if isinstance(v, dict) and "ckpt" in v:
                evaluated.add(v["ckpt"].split("/")[-2])
    n = 0
    for d in sorted(glob.glob(f"{EXP_DICE}/*")):
        name = os.path.basename(d)
        if not glob.glob(f"{d}/dice_latest.pth") and not glob.glob(f"{d}/*/dice_latest.pth"):
            continue
        if name not in evaluated:
            print(f"  {name}")
            n += 1
    print(f"  ({n} unevaluated arms)")


if __name__ == "__main__":
    main()
