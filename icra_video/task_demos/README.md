# CAST — benchmark clips for the 3-minute video

Clips illustrating the tasks CAST is evaluated on (LIBERO, robomimic, DMC) plus,
for each benchmark, a **learning progression** showing the policy improving.
Every `.mp4` has a sibling `.json` describing what it is.

## Learning clips (one per benchmark) — the "how it learns" story

| File | Benchmark | Task | Progression |
|---|---|---|---|
| `libero_task32_learning.mp4` | LIBERO | task 32 | **BASE fails → CAST succeeds** (2-stage, same initial state; real policy rollouts) |
| `robomimic_square_learning.mp4` | robomimic | square | **EARLY fails → MID solved → FINAL ~94%** (CAST policy at 3 training checkpoints) |
| `dmc_walker_run_learning.mp4` | DMC | walker run | **EARLY → MID → FINAL** (return 77→113→127; walker goes from stumbling to running) |

These are **real policy rollouts** (not demonstrations): the robomimic and LIBERO
clips run the actual CAST-finetuned policy in the simulator; the DMC clip rolls
out saved training snapshots of the policy.

## Task clips — what the tasks look like

| File | Benchmark | Task |
|---|---|---|
| `libero_red_mug.mp4` | LIBERO | put the red mug on the left plate |
| `libero_ketchup.mp4` | LIBERO | put the ketchup in the top drawer |
| `robomimic_square.mp4` | robomimic | square nut-assembly (peg insertion) |
| `dmc_cartpole_swingup.mp4` | DMC | cartpole swingup (dense reward) |

The LIBERO/robomimic task clips are successful **demonstration replays** (they show
the task, not a CAST rollout — `cast_evaluation: false` in their JSON). The DMC
task clip is a competent trained-policy rollout.

## How they were made

All rendered in the actual task simulators on a local GPU (EGL):

- **LIBERO** (`porygon` env) — `render_libero_learning.py` loads the frozen drift
  base + trained residual (`residual_libero_drift_t32`), rolls out task 32 in
  `mode="base"` vs `mode="policy"`, and captures the agentview camera.
- **robomimic** (`dice-rl` env) — `render_robomimic_learning.py` rebuilds the CAST
  agent (`newbase_CASTv2_s42`) from each checkpoint's config and records one eval
  episode; `stitch_learning.py` captions and concatenates early/mid/final.
- **DMC** (`dice-rl` env) — `render_dmc.py` rolls out the trained SAC policy's saved
  snapshots (walker-run) and the competent final policy (cartpole-swingup).

Note: robomimic clips render at the sim's native 256×256 and LIBERO at 128×128
(upscaled to 384 in the stitched clip); DMC at 640×480; demo clips at 1280×720.
