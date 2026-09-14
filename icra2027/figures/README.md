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

- **Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning**, Figs. 1–2:
  https://arxiv.org/html/2606.11087v1#S0.F1 and
  https://arxiv.org/html/2606.11087v1#S4.F2.
  The teaser repeats the reference policy across method panels and highlights
  the change in guidance. The trajectory comparison uses matched starts,
  common axes, and explicit method labels. Our current figure adopts this
  matched-panel structure: vanilla above, anchored below, with the same base
  and critic field in both. QGF's test-time denoising guidance and our
  training-time residual update are distinct methods; the numerical data and
  anchoring diagnostic here are our own.
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

- **Two matched, single-column panels.** The same analytic base distribution
  (blue, dashed contours) appears in both panels. The top panel adds the
  vanilla-updated distribution (orange); the bottom adds the anchored
  distribution (green). Both display the same critic-error region and field.
  Dots show 32 particles per distribution. The seven paths are
  selected by fixed initial x-quantiles (15%–85%), without inspecting outcomes;
  the middle one is highlighted. Labels summarize all 384 particles.
- **Shared coordinates.** A single rigid rotation aligns the mean vanilla
  departure left-to-right so two landscape panels fit inside one column.
  It is applied to both panels, including densities and critic arrows.
  Orthogonality and positive determinant are checked; there is no distance
  rescaling. Both panels have equal aspect and identical limits covering all
  displayed density contours, dots, and representative paths. The plotted
  horizontal axis is a spatial coordinate, not denoising time or RL iteration.
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

## Method-update roadmap (Figure 2)

`method_update.pdf` and `method_update.png` explain one field update in four
linked panels: sample, critic displacement, restore and clip, and regression.
The clockwise layout is inspired by **Figure 2 of HardFlow v3**:
https://arxiv.org/html/2511.08425v3#S5.F2
(PDF page 6: https://arxiv.org/pdf/2511.08425v3).
That figure is a derivation roadmap with geometric sketches and objectives.
Our independently drawn roadmap uses CAST's field-update equations; it does
not import HardFlow's optimal-control formulation or hard-feasibility claims.

```bash
python icra2027/figures/generate_method_figure.py --device cuda
python icra2027/figures/generate_method_figure.py --render-only
```

The diagram is a **constructed schematic**, not an experimental result. It
uses an analytic Gaussian base and a translated current actor, with paired
noise samples. The base standard deviations `(0.13,0.045)` make the density
elliptical; the illustrative normalized critic gradient is `(0.45,0.65)` so
the opposing vector directions are visually distinguishable. These are
schematic construction choices, not tuned experimental settings. In particular,
the field operations match the method equations: clip the gradient before
applying the reward step, add the weak pull and RMS-gated restoring field
evaluated at the current residual, clip the total displacement, then detach.

All four panels now use **identical coordinates, equal aspect, and reference
distributions**. The blue base and gray pre-update actor have exactly the same
means, covariance, sample points, colors, opacity, and contour styles in every
panel. The orange proposal is repeated identically in panels 2 and 3; the green
target cloud is repeated identically in panels 3 and 4. Rendering checks that
the coordinate-transform scale is equal across panels.

Smooth Gaussian shading and 50%/95% mass outlines make the distributions
visible. Arrowheads scale with the displayed shaft length to keep short
corrections legible. In panel 2 the dashed ray is the discarded extension of
the raw critic request beyond the accepted solid step. The ray is truncated
near the common view's upper boundary and labeled with an ellipsis: its full
endpoint is outside the panel, rather than a reason to change that panel's
zoom. The caption states this explicitly.

The restoring vector in panel 3 is drawn head-to-tail after the critic
proposal to visualize vector addition. It is still evaluated at the current
residual. The dotted circle is the **per-step target cap around the current
action**, not a hard ball around the base. The gray dashed arrow from the
combined proposal to the final target depicts total-step clipping. The target remains outside the
base dead zone in this example, which is checked numerically. In panel 4,
the green distribution is the regression target; the diagram does not assert
exact interpolation by the neural actor.

The numbers, device, and illustrative settings are saved in
`method_update.json`; all sampled actions, proposals, restoring vectors, and
targets are saved in `method_update.npz`. The numerical construction ran on
the RTX 5090 and verifies finite values, the reward and total step bounds,
and the inward direction of the restoring field. Density contours are
analytic Gaussian 50% and 95% mass contours.

## Physical task illustration (Figure 3)

Run `python icra2027/figures/generate_real_robot_figure.py` from the repository
root. It selects eight committed frames using the ICLR real-robot manifest:
initial (chest), grasp (left wrist), align (chest), and release (left wrist) for
each task. It needs no access to the original external videos. The output PDF is
self-contained for LaTeX; its JSON manifest records the source frame paths,
hashes, common crop, and nominal times. The task titles and stage labels use
separate header bands. These are teleoperation demonstrations, not autonomous
evaluation outcomes. A fifth panel in each row plots that task's autonomous
success rates for Base FM, success-only BC, DICE-RL, and DICE-RL + CAST.
These charts replace Table V and use the experimenter's counts recorded in
`icra2027/real_robot_results.json`. Both charts share a percentage axis and
method colors; labels report successes out of 20, with the largest count
bolded. Chart values and source measurements are also saved in the figure
manifest. The generator reads the results file directly.
