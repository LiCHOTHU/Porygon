# CAST policy-optimization animations

**Start with [the animation player](index.html)** for full-size playback, or [the timed slide preview](../preview.html) to watch the clips in context. The [two-minute method video](../cast_method_preview.mp4) is a silent, 1080p review export. The PowerPoint embeds the five clips; its PDF shows their poster frames.

## The explanation

1. **See the policy change.** A distribution over actions moves toward higher predicted value. A synthetic critic overestimates actions far from the useful region. With the same initial policy and step cap, clipping alone accumulates drift; CAST retains a restoring reference.
2. **Identify that reference.** The frozen generator and learned residual share the observation and noise. Each current action has a specific paired base action.
3. **Build one target.** Add critic guidance, weak restoration, and radial restoration; then cap the combined step from the current action.
4. **Show the learning.** Fixed targets are squares; actor outputs form a white curve. Gradient descent fits the residual to those targets. Targets are refreshed only after fitting.
5. **Connect to evidence.** Equal-status bars show the actual five-task LIBERO component results, including both zero-success configurations.

The visual teaching reference is Sergey Levine’s [CS 285, Fall 2023, Lecture 5: Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/static/slides/lec-5.pdf), PDF pages 11–12: Gaussian policies and making better actions more likely. Our artwork and code are original. The lecture’s REINFORCE gradient is **not** used as a substitute for CAST’s critic-guided regression update.

## Completed clips

All clips are silent H.264 MP4, **1280 × 720, 30 fps**. They fit inside the existing slide slots without changing the 120-second method / 60-second robot split.

| Clip | Duration | Content |
|---|---:|---|
| [A01](A01_accumulated_drift.mp4) | 20 s | Matched actor optimization: clipping alone versus CAST |
| [A02](A02_paired_residuals.mp4) | 15 s | Shared noise, frozen base, and learned residual |
| [A03](A03_cast_update.mp4) | 25 s | Critic proposal, the two restoring terms, and combined cap |
| [A04](A04_regress_refresh.mp4) | 22 s | Actual actor regression to fixed targets, then refresh |
| [A05](A05_measured_learning_progress.mp4) | 28 s | Measured LIBERO component comparison |

A05 retains its original reserved filename, but displays **aggregate success**, not a training curve. Full success-versus-training logs are not available locally. The accompanying slide retains the measured robomimic steps-to-90%-success comparison.

## What is actually computed

A01 and A04 use a small, trainable residual actor, not manually drawn optimization paths:

```text
z ~ Normal(0, 1)
frozen base:  a0(z) = 0.27 z
residual:     r_theta(z) = b + w z
actor:        a_theta(z) = a0(z) + r_theta(z)
critic:       Q_hat(a) = 3 - 0.22 (a - 3.4)^2
```

The synthetic critic is held fixed to isolate the actor update. The shaded useful region `[-0.25, 1.15]` is an assumption of the toy example, not a measured robot reward. The example demonstrates how the mechanisms behave when the critic favors distant actions; it does not establish a general performance guarantee.

Both configurations start at `b = w = 0` and share a fixed batch of 512 standardized Gaussian noise samples, seed 285. Each outer iteration constructs detached targets and runs four SGD steps with learning rate 0.35. There are 80 outer iterations. CAST uses:

```text
eta_Q = 0.18       delta_Q = 2.0
eta_0 = 0.12       eta_anc = 0.48
rho = 0.4         delta_total = 0.08
```

The clipping-only comparison sets both restoring coefficients to zero and changes nothing else. These are teaching-example settings, not experimental hyperparameters.

`toy_trace.json` stores both parameter trajectories, settings, and validation results. A01 renders the corresponding Gaussian action distributions. **Each curve is normalized to the same peak height**, preserving its mean and width; heights are not comparable probability densities. Frames interpolate recorded parameter updates for readable motion. A04 displays three consecutive CAST fitting rounds, using their actual frozen regression targets and fitting errors. A02 reveals the learned residual at a recorded checkpoint.

A03 is a separate two-dimensional geometric example. Its vectors are computed from the same update, with `r=(0.85, 0.28)`, normalized critic gradient `g=(1.8, 0.95)`, `eta_Q=0.55`, `delta_Q=1.2`, `eta_0=0.08`, `eta_anc=0.28`, `rho=0.38`, and `delta_total=0.30`. The radial threshold is `sqrt(2)*rho`. When restoring arrows are drawn head-to-tail, both fields were still evaluated at the current residual.

## Exact CAST target construction

For action dimension `D = H*d`, paired base action `a0`, detached residual `r`, and current action `a = a0+r`:

```text
q_scale = max(epsilon, mean(abs(Q_bar(current_candidates))))
g = grad_a Q_bar(s,z,a) / q_scale
clip(v,delta) = v * min(1, delta / norm(v))  # clip(0,delta)=0

dQ = eta_Q * clip(g, delta_Q)               # cap BEFORE scaling
m = norm(r) / sqrt(D)
dR = -eta_0*r - eta_anc*max(1-rho/max(m,epsilon),0)*r
r_target = stop_gradient(r + clip(dQ+dR, delta_total))
loss = mean_over_states_candidates_coordinates((r_theta-r_target)**2)
```

The base radius is a **soft activation threshold**. The final cap is centered on the **current action**, and constrains the constructed target step. Actor fitting does not impose a hard bound on all policy outputs.

The restoring field is the negative gradient of

```text
U(r) = eta_0/2 * ||r||^2 + eta_anc/2 * max(||r|| - sqrt(D)*rho, 0)^2.
```

The local regression-gradient identity used in the slide narration applies at target-construction parameters, with both caps inactive and the critic, samples, normalization, and targets held fixed. The paper’s distributional bound concerns candidate actions before value-guided selection.

## Benchmark and robot footage

A **6–9 second montage of actual evaluation rollouts** would help show the range of tasks. Use approximately 2–3 seconds per benchmark, inside slide 6’s existing 28-second allocation, then show the measured comparison. The main method explanation should remain the focus.

- **LIBERO:** image-based manipulation with sparse task-success rewards. Choose a task actually refined from the LIBERO-90 base.
- **robomimic:** state-based manipulation with sparse task-success rewards. A square/nut-assembly rollout connects directly to the steps-to-success result. A rendered camera view does not mean this policy takes images as input.
- **DMC:** continuous control. Choose an evaluated task and label its actual dense or sparse reward setting.

Show saved policy evaluation rollouts, not the optimizer “running inside” the environment. A base / CAST comparison should use the same task, camera, playback speed, and preferably the same initial state or evaluation seed. For training progression, label actual saved checkpoints and use matching evaluation logs; do not infer a success curve from video snippets.

No local rollout video or model checkpoint was found during this build. `benchmark_footage_plan.json` records the desired inputs and proposed timing. Environment animations have not been fabricated.

- **V01:** autonomous placement, showing approach, insertion, and withdrawal.
- **V02:** autonomous stacking, including gripper opening and the stable final stack.

The current robot photographs are explicitly labeled **teleoperated demonstration stills**. They illustrate the setup; they are not autonomous CAST executions. These two video slots remain reserved.

## Rebuild

From the repository root, with the Python dependencies in `../requirements.txt`, FFmpeg, and Poppler installed:

```bash
python icra_video/scripts/build_animations.py
python icra_video/scripts/build_slides.py
python icra_video/scripts/build_preview.py
python icra_video/scripts/render_video_preview.py
python icra_video/scripts/validate_deck.py
python icra_video/scripts/validate_animations.py
```

Use `build_animations.py --posters-only` for quick layout review, or `--clips A03` to rerender one clip. `render_video_preview.py --full` exports a three-minute silent draft with the robot placeholders. The default export is the first two minutes.

All files stay in `icra_video`; the submitted paper is unchanged.
