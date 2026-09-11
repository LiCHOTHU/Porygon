"""Reproduce the ICRA problem figure using a fitted, frozen toy critic.

Run from any directory:
    python icra2027/figures/generate_problem_figure.py --device cuda

The reward, demonstration distribution, and critic architecture follow
scripts/plot_fig1_manifold.py. This is a fresh synthetic diagnostic, not a
robot RL experiment or the full CAST update. The starting action is the
demonstration closest to x=0 (the middle of the demonstration interval),
chosen without consulting the fitted critic or the resulting trajectories.
Both arms use the same critic, starting action, learning rate, and step cap;
the second additionally projects onto a ball around the fixed start.

The distribution view additionally transports paired samples of an analytic
curved-Gaussian base sampler. These are direct particle updates, not fitted
neural policies. Fixed-bandwidth KDE contours visualize their distributions.

Save computed arrays and configuration alongside the PDF for inspection.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ORANGE = "#D55E00"
TEAL = "#008577"
BLUE = "#3667A6"
INK = "#243343"
GRAY = "#66717B"


def band(x):
    return 0.55 * x**2 - 0.25


def reward(actions):
    x, y = actions[..., 0], actions[..., 1]
    return np.exp(-((x - 0.45) ** 2) / 0.30) * np.exp(
        -((y - band(x)) ** 2) / 0.030
    )


def predict(net, actions, device):
    flat = np.asarray(actions).reshape(-1, 2)
    # Small batches keep the diagnostic usable alongside existing GPU jobs.
    with torch.no_grad():
        values = [
            net(torch.as_tensor(a, dtype=torch.float32, device=device))
            .squeeze(-1).cpu().numpy()
            for a in np.array_split(flat, max(1, (len(flat) + 1023) // 1024))
        ]
    return np.concatenate(values).reshape(np.asarray(actions).shape[:-1])


def rollout(net, start, device, *, anchored, steps, lr, cap, radius):
    point = torch.as_tensor(start, dtype=torch.float32, device=device)
    reference = point.clone()
    path = [point.detach().cpu().numpy().copy()]
    for _ in range(steps):
        point = point.detach().requires_grad_(True)
        grad = torch.autograd.grad(net(point[None]).sum(), point)[0]
        with torch.no_grad():
            delta = lr * grad
            delta *= (cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
            point = point + delta
            if anchored:
                offset = point - reference
                point = reference + offset * (
                    radius / offset.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                ).clamp(max=1.0)
        path.append(point.cpu().numpy().copy())
    return np.asarray(path)


def density_on_grid(particles, grid, bandwidth, device):
    """Normalized Gaussian KDE; use the same bandwidth for every snapshot."""
    samples = torch.as_tensor(particles, dtype=torch.float32, device=device)
    flat = grid.reshape(-1, 2)
    pieces = []
    with torch.no_grad():
        for block in np.array_split(flat, (len(flat) + 1023) // 1024):
            points = torch.as_tensor(block, dtype=torch.float32, device=device)
            sq_dist = (points[:, None] - samples[None]).square().sum(-1)
            pdf = torch.exp(-sq_dist / (2 * bandwidth**2)).mean(-1)
            pieces.append((pdf / (2 * np.pi * bandwidth**2)).cpu().numpy())
    return np.concatenate(pieces).reshape(grid.shape[:-1])


def mass_levels(density):
    """Density thresholds for 95%, 80%, and 50% highest-density regions."""
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    cumulative /= cumulative[-1]
    thresholds = [ordered[np.searchsorted(cumulative, mass)] for mass in (0.95, 0.80, 0.50)]
    assert np.all(np.diff(thresholds) > 0), "KDE contours must have distinct thresholds"
    return np.asarray(thresholds)


def plot(grid_x, grid_y, true_grid, critic_grid, demo, paths, values, config, distribution):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # Designed at its final single-column size, not shrunk from a wide figure.
    fig = plt.figure(figsize=(3.45, 2.65))
    ax = fig.add_axes([0.025, 0.10, 0.95, 0.82])
    # This region is about critic error, not action probability. Keep it neutral
    # and dashed so it cannot be mistaken for any of the three KDEs.
    overestimated = ((critic_grid > 1.0) & (true_grid < 0.01)).astype(float)
    ax.contourf(grid_x, grid_y, overestimated, levels=[0.5, 1.5], colors=["#F0F1F3"], zorder=0)
    ax.contour(grid_x, grid_y, overestimated, levels=[0.5], colors=["#A7ADB5"],
                linewidths=0.65, linestyles="--", zorder=1)
    # Direction-only arrows from finite differences of the GPU-evaluated critic
    # grid. They depict the critic field, not a claimed action-step magnitude.
    gy, gx = np.gradient(critic_grid, grid_y[:, 0], grid_x[0])
    rows, cols = np.meshgrid(np.arange(16, len(grid_y) - 16, 28),
                             np.arange(16, grid_x.shape[1] - 16, 28), indexing="ij")
    selected = overestimated[rows, cols] > 0.5
    rows, cols = rows[selected], cols[selected]
    norm = np.hypot(gx[rows, cols], gy[rows, cols]).clip(1e-8)
    ax.quiver(grid_x[rows, cols], grid_y[rows, cols],
               0.14 * gx[rows, cols] / norm, 0.14 * gy[rows, cols] / norm,
               angles="xy", scale_units="xy", scale=1, color="#B7BEC7",
               width=0.003, headwidth=3.2, headlength=4, alpha=0.75, zorder=1)
    colors = {"base": BLUE, "vanilla": ORANGE, "anchored": TEAL}
    for name in ("base", "vanilla", "anchored"):
        density = distribution[f"density_{name}"]
        levels = mass_levels(density)
        ax.contourf(grid_x, grid_y, density,
                     levels=np.r_[levels, density.max() * 1.001],
                     colors=[to_rgba(colors[name], a) for a in (0.055, 0.12, 0.22)], zorder=2)
        ax.contour(grid_x, grid_y, density, levels=levels, colors=[colors[name]],
                    linewidths=[0.65, 0.8, 1.0],
                    linestyles="--" if name == "anchored" else "-", zorder=3)
        particles = distribution[f"particles_{name}"]
        visible = np.linspace(0, len(particles) - 1, 32, dtype=int)
        ax.scatter(*particles[visible].T, s=5, color=colors[name], alpha=0.7,
                    edgecolors="white", linewidths=0.2, zorder=5)

    # Choose paths by initial x-quantile, independently of the update outcomes.
    order = np.argsort(distribution["particles_base"][:, 0])
    chosen = order[(np.linspace(0.15, 0.85, 7) * (len(order) - 1)).astype(int)]
    for name in ("vanilla", "anchored"):
        cloud_paths = distribution[f"{name}_particle_paths"]
        for idx in chosen:
            path = cloud_paths[:, idx]
            ax.plot(*path.T, color=colors[name], lw=0.55, alpha=0.30, zorder=4)
        path = cloud_paths[:, chosen[len(chosen) // 2]]
        ax.plot(*path.T, color=colors[name], lw=1.5, zorder=6,
                 path_effects=[pe.Stroke(linewidth=2.3, foreground="white"), pe.Normal()])
        t = min(16, len(path) - 3) if name == "vanilla" else min(1, len(path) - 3)
        ax.annotate("", xy=path[t + 2], xytext=path[t],
                     arrowprops={"arrowstyle": "->", "color": colors[name], "lw": 1.4}, zorder=7)

    means = {name: distribution[f"particles_{name}"].mean(0) for name in colors}
    callouts = [
        (means["base"], (0.03, 0.91), BLUE, r"Base policy $p_0$",
         f"reward {distribution['reward_base'].mean():.2f}"),
        (means["anchored"], (0.61, 0.91), TEAL, r"Ours: anchored",
         f"reward {distribution['reward_anchored'].mean():.2f}"),
        (means["vanilla"], (0.04, 0.16), ORANGE, r"Vanilla update",
         f"critic {distribution['q_vanilla'].mean():.2f}\nreward {distribution['reward_vanilla'].mean():.2f}"),
    ]
    for point, position, color, title, subtitle in callouts:
        heading = TextArea(title, textprops={"color": color, "fontsize": 8,
                                            "fontweight": "semibold"})
        detail = TextArea(subtitle, textprops={"color": GRAY, "fontsize": 7,
                                              "linespacing": 1.25})
        label = VPacker(children=[heading, detail], align="left", pad=0, sep=3)
        ax.add_artist(AnnotationBbox(label, point, xybox=position,
                                     xycoords="data", boxcoords=fig.transFigure,
                                     box_alignment=(0, 0.5),
                                     bboxprops={"fc": "white", "ec": "none", "alpha": 0.9,
                                                "boxstyle": "square,pad=0.2"},
                                     arrowprops={"arrowstyle": "-", "color": color,
                                                 "lw": 0.75, "shrinkB": 5}, zorder=8))
    ax.set(xlim=(grid_x.min(), grid_x.max()), ylim=(grid_y.min(), grid_y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(0.06, 0.43, "Critic overestimation\n" + r"$\hat Q>1$, reward $<0.01$",
             transform=ax.transAxes, color=GRAY, fontsize=6.3, va="center",
             bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1.5})
    fig.text(0.5, 0.025, "Contours: 50 / 80 / 95% probability mass", ha="center", fontsize=6.5, color=GRAY)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--render-only", action="store_true",
                        help="Redraw saved diagnostic arrays without refitting the critic")
    args = parser.parse_args()
    if args.render_only:
        with (args.out_dir / "problem_diagnostic.json").open() as f:
            config = json.load(f)["config"]
        with np.load(args.out_dir / "problem_diagnostic.npz") as data:
            paths = {name: data[f"{name}_path"] for name in ("clip_only", "anchored")}
            values = {name: {"q": data[f"{name}_q"], "reward": data[f"{name}_reward"]}
                       for name in paths}
            if "particles_base" not in data:
                raise RuntimeError("Generate the particle diagnostic with --device cuda before --render-only")
            distribution = {key: data[key] for key in data.files if key.startswith(
                ("particles_", "density_", "reward_", "q_")) or key.endswith("_particle_paths")}
            fig = plot(data["grid_x"], data["grid_y"], data["true_grid"], data["critic_grid"],
                        data["demo"], paths, values, config, distribution)
        fig.savefig(args.out_dir / "problem_motivation.pdf", dpi=300)
        fig.savefig(args.out_dir / "problem_motivation.png", dpi=200)
        plt.close(fig)
        print(f"Redrew saved {config['device']} diagnostic in {args.out_dir}")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly pass --device cpu if needed")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    config = dict(seed=args.seed, n_demo=25, train_steps=3000, train_lr=0.003,
                   steps=60, action_lr=0.09, cap=0.06, radius=0.15,
                   n_particles=384, base_std_x=0.18, base_std_y=0.035, kde_bandwidth=0.06,
                  start_rule="training action closest to x=0", device=str(device),
                  torch_version=torch.__version__, numpy_version=np.__version__)
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(device)
    x = rng.uniform(-1, 1, config["n_demo"])
    demo = np.stack([x, band(x) + 0.04 * rng.standard_normal(len(x))], axis=1)
    net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1)).to(device)
    inputs = torch.as_tensor(demo, dtype=torch.float32, device=device)
    targets = torch.as_tensor(reward(demo), dtype=torch.float32, device=device)[:, None]
    optimizer = torch.optim.Adam(net.parameters(), lr=config["train_lr"])
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = (net(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    start = demo[np.argmin(np.abs(demo[:, 0]))]
    paths = {name: rollout(net, start, device, anchored=anchored,
                           steps=config["steps"], lr=config["action_lr"],
                           cap=config["cap"], radius=config["radius"])
             for name, anchored in (("clip_only", False), ("anchored", True))}
    values = {name: {"q": predict(net, path, device), "reward": reward(path)}
               for name, path in paths.items()}
    # Analytic noise-to-action base centered near the same reference action.
    # Pair each sampled base action with its own permanently fixed anchor.
    z = rng.standard_normal((config["n_particles"], 2))
    cloud_x = start[0] + config["base_std_x"] * z[:, 0]
    cloud_y = band(cloud_x) + start[1] - band(start[0]) + config["base_std_y"] * z[:, 1]
    base_particles = np.stack([cloud_x, cloud_y], -1).astype(np.float32)
    cloud_paths = {name: rollout(net, base_particles, device, anchored=anchored,
                                 steps=config["steps"], lr=config["action_lr"],
                                 cap=config["cap"], radius=config["radius"])
                   for name, anchored in (("vanilla", False), ("anchored", True))}
    distribution = {"particles_base": base_particles}
    for name, path in cloud_paths.items():
        assert np.isfinite(path).all()
        assert np.linalg.norm(np.diff(path, axis=0), axis=-1).max() <= config["cap"] + 1e-5
        if name == "anchored":
            assert np.linalg.norm(path - path[0], axis=-1).max() <= config["radius"] + 1e-5
        distribution[f"{name}_particle_paths"] = path
        distribution[f"particles_{name}"] = path[-1]
    # Expand the view to show the complete trajectory, rather than clipping a failure.
    all_actions = np.concatenate([demo, *paths.values(), *[p.reshape(-1, 2) for p in cloud_paths.values()]])
    padding = 4 * config["kde_bandwidth"]
    lo, hi = all_actions.min(0) - padding, all_actions.max(0) + padding
    grid_x, grid_y = np.meshgrid(np.linspace(lo[0], hi[0], 240),
                                np.linspace(lo[1], hi[1], 240))
    grid = np.stack([grid_x, grid_y], -1)
    true_grid = reward(grid)
    critic_grid = predict(net, grid, device)
    report = {"config": config,
              "diagnostic": "Synthetic fixed-action optimization, not a robot training run or full CAST",
              "training_mse": float(np.mean((predict(net, demo, device) - reward(demo))**2)),
               "initial_reward": float(reward(start)), "arms": {}}
    report["particle_distributions"] = {
        "base": "Analytic curved Gaussian around the reference; not a learned robot policy",
        "n_particles": config["n_particles"], "summaries": {},
        "contour_masses": [0.50, 0.80, 0.95], "kde_bandwidth": config["kde_bandwidth"],
    }
    area = (grid_x[0, 1] - grid_x[0, 0]) * (grid_y[1, 0] - grid_y[0, 0])
    for name in ("base", "vanilla", "anchored"):
        points = distribution[f"particles_{name}"]
        density = density_on_grid(points, grid, config["kde_bandwidth"], device)
        mass = float(density.sum() * area)
        assert 0.98 < mass < 1.02, f"Incomplete KDE grid for {name}: mass={mass}"
        distribution[f"density_{name}"] = density
        distribution[f"reward_{name}"] = reward(points)
        distribution[f"q_{name}"] = predict(net, points, device)
        levels = mass_levels(density)
        report["particle_distributions"]["summaries"][name] = {
            "mean_reward": float(distribution[f"reward_{name}"].mean()),
            "mean_prediction": float(distribution[f"q_{name}"].mean()),
            "mean_distance_from_base": float(np.linalg.norm(points - base_particles, axis=-1).mean()),
            "kde_grid_mass": mass,
            "contour_enclosed_mass": [float(density[density >= level].sum() / density.sum())
                                       for level in levels],
        }
    for name, path in paths.items():
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        distance = np.linalg.norm(path - path[0], axis=1)
        assert np.isfinite(path).all() and np.isfinite(values[name]["q"]).all()
        assert steps.max() <= config["cap"] + 1e-5
        if name == "anchored":
            assert distance.max() <= config["radius"] + 1e-5
        report["arms"][name] = dict(final_reward=float(values[name]["reward"][-1]),
                                     final_prediction=float(values[name]["q"][-1]),
                                     max_step=float(steps.max()),
                                     final_distance=float(distance[-1]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(demo=demo, grid_x=grid_x, grid_y=grid_y,
                  true_grid=true_grid, critic_grid=critic_grid)
    for name in paths:
        arrays.update({f"{name}_path": paths[name],
                       f"{name}_reward": values[name]["reward"],
                       f"{name}_q": values[name]["q"]})
    np.savez_compressed(args.out_dir / "problem_diagnostic.npz", **arrays, **distribution)
    (args.out_dir / "problem_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    fig = plot(grid_x, grid_y, true_grid, critic_grid, demo, paths, values, config, distribution)
    fig.savefig(args.out_dir / "problem_motivation.pdf", dpi=300)
    fig.savefig(args.out_dir / "problem_motivation.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Wrote figure and diagnostic data to {args.out_dir}")


if __name__ == "__main__":
    main()
