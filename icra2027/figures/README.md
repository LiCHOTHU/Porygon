# Problem-motivation figure

`problem_motivation.pdf` is the paper figure; the PNG is a review preview.

## Reproduce

From the repository root, with NumPy, Matplotlib, and CUDA-enabled PyTorch:

```bash
python icra2027/figures/generate_problem_figure.py --device cuda
```

For layout edits only, reuse the exact saved observations:

```bash
python icra2027/figures/generate_problem_figure.py --render-only
```

After regeneration, compile `root.tex` from `icra2027/`.

## Visual references

The design follows distribution/transport visualization conventions inspected in:

- **Generative Modeling via Drifting**, Figs. 2–3:
  https://arxiv.org/html/2602.04770v2#S3.F2 and
  https://arxiv.org/html/2602.04770v2#S4.F3.
  These combine colored density contours, particle samples, and movement
  arrows, and distinguish the reference distribution from the evolving one.
- **Flow Matching for Generative Modeling**, Figs. 2–3:
  https://arxiv.org/html/2210.02747v2#S4.F2 and
  https://arxiv.org/html/2210.02747v2#S4.F3.
  These visualize vector fields and paired sample trajectories against
  density backgrounds.

Our figure is independently generated from the diagnostic below. The sources
inform the graphical conventions; none of their artwork or results is reused.

## What is measured

- **Three particle distributions, one compact action-space view.** Blue is
  the analytic base sampler, orange is the distribution after unanchored
  critic ascent, and green is the distribution after per-particle base
  projection. Dots show 32 particles per distribution. The seven paths are
  selected by fixed initial x-quantiles (15%–85%), without inspecting outcomes;
  the middle one is highlighted. Labels summarize all 384 particles.
- **Actual density contours.** All three clouds use the same isotropic
  Gaussian KDE bandwidth, 0.06. Contour thresholds enclose approximately
  50%, 80%, and 95% of each estimated distribution's probability mass.
  Grid integration and enclosed mass are recorded in the diagnostic JSON.
  Unlike the earlier reward heatmap, the colored regions now represent
  probability density.
- **The gray region is critic error, not density.** It is computed from
  `critic > 1` and `true reward < 0.01`. Gray arrows show increasing critic
  value, using finite differences of the GPU-evaluated grid and normalized
  display lengths. They do not represent actual step magnitudes.
- **Analytic base sampler.** Let the existing reference action be `(x0,y0)`
  and `b(x)=0.55*x^2-0.25`. Draw independent standard-normal `z1,z2` and set
  `x=x0+0.18*z1`, `y=b(x)+y0-b(x0)+0.035*z2`. This frozen curved-Gaussian
  sampler gives 384 initial actions, shared between the two update arms.
  Each particle's anchor is its own initial action, not the cloud centroid.
  It is an illustrative noise-to-action policy, not a learned robot policy.
- **A synthetic diagnostic.** A two-hidden-layer ReLU critic
  (256 units per layer) is fitted to 25 action–reward pairs on a curved band.
  Reward is continuous and bounded between 0 and 1. The frozen critic is then
  used for 60 direct action updates. Both arms share the starting particles,
  critic, learning rate (0.09), and per-step L2 cap (0.06). The anchored arm
  additionally projects onto a radius-0.15 ball around each fixed start.
  It is a projection diagnostic, not full CAST or a robot RL run.
- The starting point is the demonstration closest to the center of the
  sampled x interval (x=0), selected without consulting the critic or the
  update outcomes. The illustrated run uses seed 0, not an aggregate over
  training seeds. It demonstrates a possible failure, not its frequency.
- The reward function, data distribution, and architecture are adapted from
  `scripts/plot_fig1_manifold.py`. These are newly computed trajectories;
  values from the older figure are not reused.
- **Supporting robot data.** `constraint_factorial.json` records
  the five-task means from `tab:constraint` in the ICLR draft, with source
  attribution. Averaging the same five task values gives the
  pretrained base (67.28%), clip-only update (0%), and CAST (70.34%). These
  are the constraint-ablation results, not the separate hard-8 headline
  comparison. These are not new robot evaluations. This supporting file is
  retained for the experiment section; Figure 1 now focuses on the toy
  mechanism. Raw per-seed observations are unavailable in this checkout.

`problem_diagnostic.json` records the configuration, device, software versions,
fit error, per-distribution mean rewards and predictions, and density-mass
checks. `problem_diagnostic.npz` contains the training actions, reward and
critic grids, all particle paths and snapshots, KDEs, and per-particle scores.
The earlier single-action trajectories are retained for provenance but are not
the plotted distribution means. The generator verifies finite results, the
per-particle step caps, the anchored distance bounds, and integrated KDE mass.

The current diagnostic was generated on the local NVIDIA GeForce RTX 5090.
Floating-point differences across devices can change a fitted critic outside
its training data; use the saved arrays for exact figure reproduction.
