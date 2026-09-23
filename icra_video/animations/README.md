# CAST policy-optimization animations

**Start with [the animation player](index.html)** for full-size playback, or [the timed slide preview](../preview.html) to watch the clips in context. The [96-second method video](../cast_method_preview.mp4) is a silent, 1080p review export. The PowerPoint embeds the five clips; its PDF shows their poster frames.

## The explanation

1. **See the policy change.** A distribution over actions moves toward higher predicted value. A synthetic critic overestimates actions far from the useful region. With the same initial policy and step cap, clipping alone accumulates drift; CAST retains a restoring reference.
2. **Identify that reference.** The frozen generator and learned residual share the observation and noise. Each current action has a specific paired base action.
3. **Build one target.** Add critic guidance, weak restoration, and radial restoration; then cap the combined step from the current action.
4. **Show the learning.** Fixed targets are squares; actor outputs form a white curve. Gradient descent fits the residual to those targets. Targets are refreshed only after fitting.
5. **Connect to evidence.** Equal-status bars show the actual five-task LIBERO component results, including both zero-success configurations.

The visual teaching reference is Sergey Levine’s [CS 285, Fall 2023, Lecture 5: Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/static/slides/lec-5.pdf), PDF pages 11–12: Gaussian policies and making better actions more likely. Our artwork and code are original. The lecture’s REINFORCE gradient is **not** used as a substitute for CAST’s critic-guided regression update.

## Completed clips

All clips are silent H.264 MP4, **1280 × 720, 30 fps**. They fit inside the existing slide slots with a faster 96-second method/evidence segment and 60-second robot segment. The 156-second draft leaves 24 seconds for incoming simulation footage within the three-minute limit.

| Clip | Duration | Content |
|---|---:|---|
| [A01](A01_accumulated_drift.mp4) | 12 s | Matched actor optimization: clipping alone versus CAST |
| [A02](A02_paired_residuals.mp4) | 8 s | Shared noise, frozen base, and learned residual |
| [A03](A03_cast_update.mp4) | 30 s | Animated paper Figure 2, with synchronized equations and both clipping operations |
| [A04](A04_regress_refresh.mp4) | 10 s | Actual actor regression to fixed targets, then refresh |
| [A05](A05_measured_learning_progress.mp4) | 30 s | Measured LIBERO component comparison |

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

## Animated Figure 2 — the main method explanation

[A03](A03_cast_update.mp4) now uses the **submitted Figure 2’s exact numerical construction**, loaded from `_ICRA_2027__CAST/figures/method_update.npz` and its JSON settings. It preserves the clockwise four-panel overview, focuses on one active panel at a time, and returns to the complete figure. No submitted paper asset is modified.

| Time | Visual operation | Equation shown |
|---|---|---|
| 0–5 s | Clockwise overview; paired samples | `a0 = pi0(s,z)`, `a = a0 + stop_gradient(r_theta)` |
| 5–11 s | Raw critic request visibly contracts | `dQ = eta_Q * clip(g, delta_Q)`; `||dQ|| <= eta_Q*delta_Q` |
| 11–18 s | Weak pull, then additional radial pull | Both terms of the paper’s restoring field; RMS departure `m` |
| 18–24 s | Combined proposal contracts onto the current-centered cap | `r_target = sg[r + clip(dQ+dR, delta_total)]`; target-step inequality |
| 24–30 s | Actor fits fixed targets; return to overview | The paper’s actor MSE, target action, and stop-gradient definition |

Math is typeset with Matplotlib’s STIX math renderer. Equations appear beside their corresponding operation. Motion completes within about 1–2 seconds, leaving brief reading time rather than slow transitions.

The original schematic uses `D=2`, `eta_Q=1`, `eta_0=0.05`, `eta_anc=1`, `rho=0.10`, `delta_Q=0.25`, and `delta_total=0.12`. Its restoration vectors are evaluated at the current residual and drawn head-to-tail only to illustrate addition. The last stage performs eight gradient-descent steps on a translation residual with learning rate 0.35, using the figure’s fixed targets. This is an illustrative fit, not benchmark training. `figure2_trace.json` records the cue timings, fitting parameters, and loss values.

The surrounding explanations are shorter: drift **20 → 12 s**, shared-noise pairing **15 → 8 s**, and fitting/refresh **22 → 10 s**. The main Figure 2 sequence receives 30 seconds because it now carries the complete mathematical update.

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

A **nine-second task-demonstration montage** is now available in `../footage/benchmark_task_montage.mp4`. It uses recorded LIBERO and robomimic human demonstrations and a scripted DMC controller. These clips establish the task setting; they are not CAST evaluation rollouts. Incoming server footage can replace them or use the remaining 24-second budget. Use approximately 2–3 seconds per benchmark, inside slide 6’s 30-second allocation, then show the measured comparison. The main method explanation should remain the focus.

- **LIBERO:** image-based manipulation with sparse task-success rewards. Choose a task actually refined from the LIBERO-90 base.
- **robomimic:** state-based manipulation with sparse task-success rewards. A square/nut-assembly rollout connects directly to the steps-to-success result. A rendered camera view does not mean this policy takes images as input.
- **DMC:** continuous control. Choose an evaluated task and label its actual dense or sparse reward setting.

Show saved policy evaluation rollouts, not the optimizer “running inside” the environment. A base / CAST comparison should use the same task, camera, playback speed, and preferably the same initial state or evaluation seed. For training progression, label actual saved checkpoints and use matching evaluation logs; do not infer a success curve from video snippets.

High-resolution task demonstrations were rendered from local LIBERO simulator states, official robomimic simulator states, and the actual DMC environment. `../footage/README.md` records provenance and reproduction. `benchmark_footage_plan.json` distinguishes these task illustrations from incoming policy evaluation footage.

- **V01:** autonomous placement, showing approach, insertion, and withdrawal.
- **V02:** autonomous stacking, including gripper opening and the stable final stack.

The current robot photographs are explicitly labeled **teleoperated demonstration stills**. They illustrate the setup; they are not autonomous CAST executions. These two video slots remain reserved.

## Rebuild

From the repository root, with the Python dependencies in `../requirements.txt`, FFmpeg, and Poppler installed:

```bash
python icra_video/scripts/build_animations.py
python icra_video/scripts/build_environment_montage.py
python icra_video/scripts/build_slides.py
python icra_video/scripts/build_preview.py
python icra_video/scripts/render_video_preview.py
python icra_video/scripts/validate_deck.py
python icra_video/scripts/validate_animations.py
```

Use `build_animations.py --posters-only` for quick layout review, or `--clips A03` to rerender one clip. `render_video_preview.py --full` exports the complete 156-second silent draft with the robot placeholders. The default export is the 96-second method/evidence segment.

All files stay in `icra_video`; the submitted paper is unchanged.
