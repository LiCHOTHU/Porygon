"""ICRA task demonstrations and autonomous success-rate charts.

Uses the source manifest in iclr2026/figures, not unavailable raw videos.
Produces a self-contained PDF for LaTeX and a frame-selection manifest.
These images are demonstrations, not autonomous policy evaluation results.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "iclr2026" / "figures"


def main():
    source = json.loads((SOURCE / "real_robot_tasks.json").read_text())
    results = json.loads((HERE.parent / "real_robot_results.json").read_text())
    task_keys = {"jigglypuff": "jigglypuff_into_case", "stack": "red_on_purple_stacking"}
    colors = ["#8A96A3", "#D99B38", "#4477AA", "#168F80"]
    method_labels = ["Base", "BC", "DICE-RL", "CAST"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.15, 2.65), facecolor="white")
    columns = [("chest", 0, "Initial / chest"), ("left_wrist", 1, "Grasp / wrist"),
               ("chest", 3, "Align / chest"), ("left_wrist", 4, "Release / wrist")]
    manifest = {"kind": "teleoperated demonstration frames with separate autonomous evaluation charts",
                "source_manifest": "iclr2026/figures/real_robot_tasks.json",
                "results_source": "2026_ICRA_CAST/real_robot_results.json",
                "method_labels": method_labels, "method_colors": colors,
                "alignment": source["alignment"], "crop_xyxy": source["crop_xyxy"],
                "frames": [], "charts": []}
    for row, sequence in enumerate(source["sequences"]):
        y = 0.59 if row == 0 else 0.105
        title_y = 0.99 if row == 0 else 0.505
        fig.text(0.012, title_y, sequence["title"], fontsize=8.5,
                  fontweight="medium", color="#243D50", va="top")
        for col, (view, stage, title) in enumerate(columns):
            frame = sequence["frames"][stage]
            stem = Path(sequence["sources"][view]["video"]).stem
            path = SOURCE / "real_robot_frames" / f"{stem}_frame_{frame:04d}.png"
            image = plt.imread(path)
            x0, y0, x1, y1 = source["crop_xyxy"]
            assert image.shape[0] >= y1 and image.shape[1] >= x1
            ax = fig.add_axes([0.008 + col * 0.182, y, 0.174, 0.30])
            ax.imshow(image[y0:y1, x0:x1])
            ax.set(xticks=[], yticks=[])
            ax.set_title(title, fontsize=6.4, pad=3)
            for spine in ax.spines.values():
                spine.set_color("#CDD5DB")
                spine.set_linewidth(0.5)
            time = sequence["sources"][view]["nominal_times_s"][stage]
            manifest["frames"].append({"task": sequence["task"], "view": view,
                "stage": source["stages"][stage], "index": frame, "nominal_time_s": time,
                "path": str(path.relative_to(HERE.parents[1])),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        measurements = results["tasks"][task_keys[sequence["task"]]]["results"]
        rates = [100 * item["successes"] / item["trials"] for item in measurements]
        ax = fig.add_axes([0.785, y, 0.208, 0.30])
        bars = ax.bar(range(4), rates, width=0.67, color=colors, zorder=3)
        ax.set_title("Autonomous success (%)", fontsize=6.4, pad=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(range(4), method_labels)
        ax.tick_params(axis="both", labelsize=6, length=0, pad=2)
        ax.grid(axis="y", color="#E4E8EC", linewidth=0.5, zorder=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#CDD5DB")
        ax.spines["bottom"].set_linewidth(0.5)
        for bar, item, rate in zip(bars, measurements, rates):
            ax.text(bar.get_x() + bar.get_width()/2, rate + 2,
                    f"{item['successes']}/{item['trials']}", ha="center", va="bottom",
                    fontsize=6, clip_on=False, fontweight="bold" if rate == max(rates) else "normal",
                    color="#243D50")
        manifest["charts"].append({"task": sequence["task"], "metric": "success_percent",
                                   "results": measurements, "labels": method_labels, "values": rates})
    fig.savefig(HERE / "real_robot_tasks.pdf", dpi=220)
    fig.savefig(HERE / "real_robot_tasks.png", dpi=180)
    plt.close(fig)
    (HERE / "real_robot_tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote real-robot figure: 8 documented demonstration frames and 2 autonomous success charts")


if __name__ == "__main__":
    main()
