"""Compact, reproducible experiment charts for the ICRA paper.

Run from any directory with Python + matplotlib + numpy. All measurements
come from experiment_chart_data.json; no curves or uncertainty estimates
are reconstructed from summary measurements. PDF outputs retain vectors.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "experiment_chart_data.json").read_text())
CAST, DICE, INK = "#168F80", "#4477AA", "#242B33"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": "#66717C",
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    plt.close(fig)


def diffusion_efficiency():
    data = DATA["diffusion_efficiency"]
    fig = plt.figure(figsize=(3.5, 1.68))
    ax = fig.add_axes([0.20, 0.30, 0.77, 0.53])
    methods = ["CAST", "DICE-RL"]
    for y, method, color in zip([1, 0], methods, [CAST, DICE]):
        values = np.asarray(data["environment_steps"][method]) / 1000
        # Fixed vertical offsets separate the two actual 160K observations;
        # they encode no additional numerical measurement or uncertainty.
        offsets = np.array([0.0, -0.10, 0.10]) if method == "CAST" else np.zeros(3)
        ax.scatter(values, y + offsets, s=25, color=color,
                   edgecolors="white", linewidths=0.45, zorder=4)
        ax.scatter([values.mean()], [y - 0.22], marker="D", s=24,
                   facecolor=INK, edgecolor="white", linewidth=0.45, zorder=5)
        ax.text(values.mean(), y + 0.26, f"Mean {values.mean():.1f}K",
                ha="center", va="center", fontsize=7.2, fontweight="medium")
    ax.set_yticks([1, 0], methods)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_ylim(-0.46, 1.47)
    ax.set_xlim(100, 280)
    ax.set_xticks([100, 150, 200, 250], ["100K", "150K", "200K", "250K"])
    ax.set_xlabel("Environment steps to reach 90% success", labelpad=4)
    ax.grid(axis="x", color="#E4E7EA", linewidth=0.55)
    ax.set_axisbelow(True)
    for edge in ["left", "right", "top"]:
        ax.spines[edge].set_visible(False)
    ax.tick_params(axis="x", length=3)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4,
               color="#677580", label="Training run"),
        Line2D([], [], linestyle="none", marker="D", markersize=4,
               color=INK, label="Mean of three runs"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.0),
               ncol=2, frameon=False, fontsize=7, handletextpad=0.45,
               columnspacing=1.3)
    save(fig, "diffusion_efficiency")


def component_comparison():
    data = DATA["component_comparison"]
    keys = [cfg["key"] for cfg in data["configurations"]]
    tasks = data["tasks"]
    success = np.array([[task[key] for key in keys] for task in tasks])
    base = np.array([task["base"] for task in tasks])
    delta = success - base[:, None]
    assert np.all(success[:, :2] == 0)
    assert np.all(delta[:, -1] > 0)
    assert np.isclose(base.mean(), 67.28)
    assert np.allclose(success.mean(axis=0), [0, 0, 65.74, 70.34])

    fig = plt.figure(figsize=(3.5, 2.3))
    ax = fig.add_axes([0.185, 0.39, 0.795, 0.45])
    colors = ["#AA4499", "#CC6677", "#4477AA", "#D99B38", "#168F80"]
    markers = ["o", "s", "^", "v", "P"]
    offsets = np.linspace(-0.23, 0.13, len(tasks))
    handles = []
    for i, (task, color, marker, offset) in enumerate(zip(tasks, colors, markers, offsets)):
        handle = ax.scatter(np.arange(4) + offset, delta[i], s=21,
                            marker=marker, color=color, edgecolor="white",
                            linewidth=0.4, label=task["name"], zorder=4)
        handles.append(handle)
    mean = ax.scatter(np.arange(4) + 0.29, delta.mean(axis=0), s=30, marker="D",
                      color=INK, edgecolor="white", linewidth=0.5,
                      label="Task mean", zorder=6)
    handles.append(mean)
    ax.axhline(0, color="#68737C", linewidth=0.8, zorder=2)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-88, 30)
    ax.set_yticks([-80, -60, -40, -20, 0, 20])
    ax.set_ylabel("Success change (pp)", labelpad=3)
    ax.set_xticks(np.arange(4), ["No\ncontrols", "Clipping\nonly", "Radial\nonly", "Full\nCAST"])
    ax.tick_params(axis="x", length=0, pad=5, labelsize=7)
    ax.tick_params(axis="y", length=3)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.5)
    ax.set_axisbelow(True)
    for edge in ["right", "top"]:
        ax.spines[edge].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.005),
               ncol=3, frameon=False, fontsize=6.7, handletextpad=0.15,
               columnspacing=0.65, borderpad=0.2, labelspacing=0.45)

    # Small control matrix makes the non-factorial design explicit.
    key_ax = fig.add_axes([0.185, 0.028, 0.795, 0.18])
    key_ax.set_xlim(-0.5, 3.5)
    key_ax.set_ylim(-0.5, 2.5)
    key_ax.axis("off")
    rows = [("Weak", "weak", 2), ("Radial", "radial", 1), ("Caps", "caps", 0)]
    for title, key, y in rows:
        key_ax.text(-0.55, y, title, ha="right", va="center", fontsize=6.8)
        for x, cfg in enumerate(data["configurations"]):
            key_ax.text(x, y, "on" if cfg[key] else "–", ha="center", va="center",
                        fontsize=6.8, color=INK if cfg[key] else "#7C858D")
    save(fig, "component_comparison")


if __name__ == "__main__":
    diffusion_efficiency()
    component_comparison()
    runs = DATA["diffusion_efficiency"]["environment_steps"]
    gain = 1 - np.mean(runs["CAST"]) / np.mean(runs["DICE-RL"])
    print(f"Environment-step reduction to threshold: {100 * gain:.2f}%")
