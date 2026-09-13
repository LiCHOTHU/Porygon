"""Plot the recorded training and validation flow-matching losses."""

import argparse
import json
import yaml
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("real_robot/runs/jigglypuff"))
    args = parser.parse_args()
    with (args.run_dir / "metrics.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    best = min(rows, key=lambda row: row["val_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax in axes:
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                "o-", label="Training", markersize=4)
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows],
                "o-", label="Validation", markersize=4)
        ax.axvline(best["epoch"], color="gray", linestyle="--", alpha=0.65,
                   label=f"Best checkpoint: epoch {best['epoch']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Flow-matching velocity MSE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_title("Full training run")
    axes[1].set_title("Later epochs (zoomed)")
    axes[1].set_xlim(5, rows[-1]["epoch"] + 0.5)
    axes[1].set_ylim(0.075, 0.30)
    with (args.run_dir / "config.yaml").open() as stream:
        cfg = yaml.safe_load(stream)
    with (Path(cfg["cache_dir"]) / "manifest.json").open() as stream:
        manifest = json.load(stream)
    train = sum(e["split"] == "train" for e in manifest["episodes"])
    val = sum(e["split"] == "val" for e in manifest["episodes"])
    later = [r for r in rows if r["epoch"] >= 5]
    if later:
        values = [r[k] for r in later for k in ("train_loss", "val_loss")]
        margin = max(0.01, (max(values) - min(values)) * 0.1)
        axes[1].set_ylim(max(0, min(values) - margin), max(values) + margin)
    fig.suptitle(f"{args.run_dir.name.capitalize()} BC — flow-matching loss\n"
                 f"{train} train takes / {val} validation takes")
    fig.tight_layout()
    output = args.run_dir / "loss_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
