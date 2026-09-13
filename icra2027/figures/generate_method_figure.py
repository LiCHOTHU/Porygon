"""CAST's field update as a four-stage visual roadmap.

This is an explicitly constructed schematic, not an experimental result.
An analytic Gaussian base and a translated current actor share noise samples.
The critic, restoring, and clipping displacements are computed from the paper's
equations, with illustrative parameters chosen to make all operations visible.
The final panel shows regression targets, not a guaranteed post-fit policy.

python icra2027/figures/generate_method_figure.py --device cuda
python icra2027/figures/generate_method_figure.py --render-only
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
BLUE, ORANGE, TEAL = "#3667A6", "#D55E00", "#008577"
GRAY, INK = "#82909D", "#283746"


def clip(v, cap):
    return v * (cap / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)


def make_data(device):
    config = dict(device=str(device), seed=8, particles=25, dimension=2,
                  eta_q=1.0, eta_0=0.05, eta_anc=1.0,
                  rho_rms=0.10, q_cap=0.25, total_cap=0.12,
                  base_mean=[-0.52, 0.10], base_std=[0.13, 0.045],
                  current_residual=[0.60, 0.12], normalized_q_gradient=[0.45, 0.65])
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(device)
    torch.manual_seed(config["seed"])
    z = torch.randn(config["particles"], 2, device=device)
    z[0] = 0  # Follow the center particle consistently in all four panels.
    tensor = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    base = tensor(config["base_mean"]) + z * tensor(config["base_std"])
    residual = tensor(config["current_residual"]).expand_as(base)
    current = base + residual
    g = tensor(config["normalized_q_gradient"]).expand_as(base)
    dq = config["eta_q"] * clip(g, config["q_cap"])
    rms = residual.norm(dim=-1, keepdim=True) / np.sqrt(2)
    restoring = -config["eta_0"] * residual - config["eta_anc"] * (
        1 - config["rho_rms"] / rms.clamp_min(1e-12)
    ).clamp(min=0) * residual
    combined = dq + restoring
    delta = clip(combined, config["total_cap"])
    target_residual = (residual + delta).detach()
    arrays = dict(base=base, current=current, raw_q=current + g,
                  q_proposal=current + dq, combined=current + combined,
                  target=base + target_residual, dq=dq, dr=restoring, delta=delta)
    assert torch.isfinite(torch.stack(list(arrays.values()))).all()
    assert dq.norm(dim=-1).max() <= config["eta_q"] * config["q_cap"] + 1e-6
    assert delta.norm(dim=-1).max() <= config["total_cap"] + 1e-6
    assert (restoring * residual).sum(-1).max() <= 0
    # This soft update need not put the target inside the base-centered radius.
    assert (target_residual.norm(dim=-1) / np.sqrt(2) > config["rho_rms"]).all()
    return {key: value.cpu().numpy() for key, value in arrays.items()}, config


def draw_cloud(ax, mean, std, color, *, alpha=1, linestyle="-"):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x, y = np.meshgrid(np.linspace(xmin, xmax, 240), np.linspace(ymin, ymax, 180))
    # Smooth density shading with exact Gaussian 50% and 95% mass contours.
    radius_sq = ((x - mean[0]) / std[0])**2 + ((y - mean[1]) / std[1])**2
    density = np.exp(-0.5 * radius_sq)
    cmap = LinearSegmentedColormap.from_list("cloud", [to_rgba(color, 0), to_rgba(color, 0.30 * alpha)])
    ax.imshow(density, extent=(xmin, xmax, ymin, ymax), origin="lower", cmap=cmap,
                vmin=0, vmax=1, interpolation="bilinear", aspect="equal", zorder=1)
    ax.contour(x, y, density, levels=[0.05, 0.50], colors=[to_rgba(color, 0.8 * alpha)],
                linewidths=[0.55, 0.8], linestyles=linestyle, zorder=2)


def arrow(ax, start, end, color, *, dashed=False, lw=1.35):
    # Size heads relative to the visible shaft; fixed large heads reversed the
    # apparent direction of short arrows in the previous small-panel rendering.
    ax.apply_aspect()
    length = np.linalg.norm(ax.transData.transform(end) - ax.transData.transform(start))
    length *= 72 / ax.figure.dpi
    head = min(9, max(3.5, length * 0.40))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, linestyle="--" if dashed else "-",
                                 shrinkA=min(0.6, length * 0.04),
                                 shrinkB=min(1.0, length * 0.06), zorder=8))


def point(ax, xy, color, *, marker="o", size=11, hollow=False):
    ax.scatter(*xy, s=size, marker=marker, facecolors="white" if hollow else color,
                edgecolors=color if hollow else "white", linewidths=0.4, zorder=7)


def render(data, config):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "text.color": INK, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(3.45, 2.90))
    # Serpentine reading order mirrors the reference roadmap: 1 -> 2 -> 3 -> 4.
    positions = [(0.02, 0.54), (0.54, 0.54), (0.54, 0.03), (0.02, 0.03)]
    titles = ["1  Sample", "2  Critic step", "3  Restore + clip", "4  Fit the actor"]
    # Every panel uses exactly the same action coordinates and reference clouds.
    # The long raw critic request is shown as an off-panel continuation in step 2.
    common_view = ((-0.90, 0.59), (-0.16, 0.68))
    formulas = [r"$a_i=a_{0,i}+\bar r_i$",
                r"$d_i^Q=\eta_Q C_{\delta_Q}(g_i)$",
                r"$\Delta_i=C_{\delta_{\rm tot}}(d_i^Q+d_i^R)$" + "\n" + r"$\tilde r_i=\mathrm{sg}[\bar r_i+\Delta_i]$",
                r"$\min_\theta\ \sum_i\|r_\theta-\tilde r_i\|^2$"]
    axes = []
    for n, (left, bottom) in enumerate(positions):
        fig.add_artist(FancyBboxPatch((left, bottom), 0.44, 0.43,
                                       boxstyle="round,pad=0.007,rounding_size=0.013",
                                       transform=fig.transFigure, lw=0.6,
                                       edgecolor="#CED6DE", facecolor="#FBFCFD", zorder=0))
        fig.text(left + 0.014, bottom + 0.389, titles[n], fontsize=7.1, fontweight="medium")
        ax = fig.add_axes([left + 0.012, bottom + 0.095, 0.414, 0.275])
        ax.set(xlim=common_view[0], ylim=common_view[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        axes.append(ax)
        fig.add_artist(FancyBboxPatch((left + 0.011, bottom + 0.011), 0.418, 0.073,
                                       boxstyle="round,pad=0.003,rounding_size=0.007",
                                       transform=fig.transFigure, facecolor="#F2F5F7",
                                       edgecolor="none", zorder=0))
        fig.text(left + 0.22, bottom + 0.047, formulas[n], ha="center", va="center",
                  fontsize=5.9 if n == 2 else 6.4, linespacing=1.35)
    for start, end in (((0.474, 0.77), (0.526, 0.77)),
                       ((0.76, 0.522), (0.76, 0.478)),
                       ((0.526, 0.26), (0.474, 0.26))):
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                        arrowstyle="-|>", mutation_scale=8,
                                         color="#7D8B97", lw=0.9))
    fig.text(0.025, 0.499, "Blue: frozen base   ·   Gray: current actor",
              fontsize=5.8, color=GRAY)

    std = np.asarray(config["base_std"])
    base, current, proposed = (data[key][0] for key in ("base", "current", "q_proposal"))
    combined, target, raw = (data[key][0] for key in ("combined", "target", "raw_q"))
    for ax in axes:
        draw_cloud(ax, base, std, BLUE, alpha=0.8, linestyle="--")
        draw_cloud(ax, current, std, GRAY, alpha=0.65)
        ax.scatter(*data["base"][1:7].T, color=BLUE, s=3, alpha=0.5)
        ax.scatter(*data["current"][1:7].T, color=GRAY, s=3, alpha=0.5)
        point(ax, base, BLUE, size=11)
        point(ax, current, INK)
        ax.text(base[0] - 0.02, -0.06, r"$a_0$", color=BLUE, fontsize=6.5, ha="center")
        ax.text(current[0] + 0.10, current[1] - 0.15, r"$a_i$", fontsize=6.5, color=GRAY)

    # Intermediate distributions persist: proposal 2 -> 3, fixed target 3 -> 4.
    for ax in (axes[1], axes[2]):
        draw_cloud(ax, proposed, std, ORANGE, alpha=0.80)
        point(ax, proposed, ORANGE, size=7)
    for ax in (axes[2], axes[3]):
        draw_cloud(ax, target, std, TEAL, alpha=0.9)
        point(ax, target, TEAL, marker="D", size=13)

    # 1: paired samples from a frozen base and its current residual policy.
    ax = axes[0]
    arrow(ax, base, current, GRAY)
    ax.text(-0.52, 0.35, "frozen base", fontsize=6.2, color=BLUE, ha="center")
    ax.text(0.08, 0.54, "current actor", fontsize=6.2, color=GRAY, ha="center")

    # 2: the raw critic request and the clipped reward displacement.
    ax = axes[1]
    # Solid accepted displacement followed by the discarded dashed extension.
    # Display the raw ray only up to the shared view boundary; do not move or
    # rescale any distribution just to accommodate its far-away endpoint.
    direction = raw - proposed
    fraction = min(1.0, (common_view[1][1] - 0.03 - proposed[1]) / direction[1])
    raw_visible = proposed + fraction * direction
    arrow(ax, proposed, raw_visible, ORANGE, dashed=True, lw=0.8)
    arrow(ax, current, proposed, ORANGE, lw=1.6)
    ax.text(0.13, 0.61, "raw…", color=ORANGE, fontsize=6.0, ha="center")
    ax.text(0.30, 0.27, r"$d_i^Q$", color=ORANGE, fontsize=6.5)

    # 3: translate the restoring vector head-to-tail, then cap the total step.
    # dR is evaluated at the CURRENT residual, not at the orange proposal.
    ax = axes[2]
    ax.add_patch(Circle(current, config["total_cap"], fill=False, color=GRAY,
                        linestyle=":", lw=0.65, zorder=3))
    arrow(ax, current, proposed, ORANGE, lw=1.3)
    arrow(ax, proposed, combined, TEAL, lw=1.5)
    arrow(ax, combined, target, GRAY, dashed=True, lw=0.8)
    point(ax, combined, GRAY, hollow=True, size=7)
    ax.text(-0.14, 0.57, r"$d_i^R$: restore", color=TEAL, fontsize=6.3, ha="center")
    ax.annotate("corrected target", xy=target, xytext=(0.16, -0.10), fontsize=6.0,
                 color=TEAL, ha="center", arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.5})

    # 4: a cloud of detached targets supervises the actor. Arrows depict the
    # regression objective, not a measured or guaranteed realized policy move.
    ax = axes[3]
    for idx in (2, 6, 8):
        arrow(ax, data["current"][idx], data["target"][idx], TEAL, lw=1.0)
        point(ax, data["target"][idx], TEAL, size=5)
    ax.text(-0.08, 0.55, "fit to fixed targets", fontsize=6.2, color=TEAL, ha="center")
    # Guard against another accidental per-panel zoom or aspect change.
    for ax in axes:
        ax.apply_aspect()
    linear = axes[0].transData.get_affine().get_matrix()[:2, :2]
    assert all(np.allclose(ax.transData.get_affine().get_matrix()[:2, :2], linear)
               for ax in axes), "Panel coordinate scales must remain identical"
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        with (HERE / "method_update.json").open() as f:
            config = json.load(f)["configuration"]
        with np.load(HERE / "method_update.npz") as saved:
            data = dict(saved)
    else:
        data, config = make_data(torch.device(args.device))
        np.savez_compressed(HERE / "method_update.npz", **data)
        report = dict(kind="Constructed field-update schematic, not experimental evidence",
                      configuration=config,
                      first_particle={key: value[0].tolist() for key, value in data.items()},
                      max_target_step=float(np.linalg.norm(data["delta"], axis=-1).max()))
        (HERE / "method_update.json").write_text(json.dumps(report, indent=2) + "\n")
    fig = render(data, config)
    fig.savefig(HERE / "method_update.pdf", dpi=300)
    fig.savefig(HERE / "method_update.png", dpi=240)
    plt.close(fig)
    print(f"Wrote method_update.pdf and .png (numerical construction: {config['device']})")


if __name__ == "__main__":
    main()
