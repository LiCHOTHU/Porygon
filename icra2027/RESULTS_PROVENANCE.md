# Eight-page main-paper experiment draft

Simulation source snapshot: `800cfe6` (ICLR manuscript and real-robot setup).
The Jigglypuff and stacking evaluations additionally use counts reported directly by the
experimenter, recorded in `real_robot_results.json`.

## Main result displays

| ICRA label / file | Source | Transformation |
|---|---|---|
| `tab:robomimic`, `sections/tables/05_robomimic.tex` | `tab:matched` and `tab:baselines` | Seed ranges for drifting can/square; paired diffusion/FM square rows; separate external-reference block |
| `tab:libero-main`, `sections/tables/05_libero.tex` | `tab:libero-single` | All eight tasks, both bases, both shared CAST reward steps; provisional GRPO cells not averaged into this table |
| `tab:ablations`, `sections/tables/05_ablations.tex` | `tab:constraint`, `tab:exp1`, reward-field part of `tab:locus` | Three labeled study blocks, each retaining its own reference and aggregation |
| `tab:dmc`, `sections/tables/05_dmc.tex` | `tab:dmc-scope` | All six tasks, with every incomplete/high-variance/tuned marker preserved |
| `fig:real-robot`, `figures/real_robot_tasks.pdf` | ICLR task setup + user-reported evaluations | Fifth-column bar charts replace Table V: four methods per task, 20 trials each, shared percentage scale and count labels |

## Physical evaluation updates

The experimenter reports 20 rollouts per method with Jigglypuff's pose
randomly changed on the stairs: Base FM 11/20 (55%), BC refinement 16/20
(80%), DICE-RL 18/20 (90%), and CAST 19/20 (95%). BC refinement maps to
the existing success-only BC condition; CAST maps to DICE-RL + CAST.
The counts are not derived from the demonstration images. The CAST--DICE
difference is one successful trial. Exact pose ranges and paired reset
identifiers were not supplied, so the manuscript does not claim a paired
trial design or particular translation/rotation ranges.

For stacking, the experimenter reports Base FM 7/20 (35%), BC refinement
13/20 (65%), DICE-RL 12/20 (60%), and CAST 14/20 (70%). CAST exceeds
DICE-RL by two trials and BC by one; BC exceeds DICE-RL on this task.
Stacking pose randomization and the post-release observation interval were
not supplied. Jigglypuff's randomized-pose description is not extended to
stacking without confirmation.

## Arithmetic and protocol decisions

- The external-baseline source caption/body says that the stricter harness
  evaluates the shared square checkpoint at **0.348**, although its old table
  printed 0.600 and subtracted that value. The ICRA external block uses the
  reported *same-harness* baseline, 0.348, and recomputes deltas:
  DIPO -0.192; QSM/DQL -0.348; IDQL -0.021; AWR +0.045.
  This changes arithmetic/labeling, not the reported fine-tuned measurements.
- DIPO/QSM/DQL/IDQL/AWR use about 24M environment steps. The old blanket
  “125x our budget” statement refers to a 192K reference run, not all paired
  rows in the current table. The ICRA prose reports the absolute external
  budget and does not apply that ratio to the 496K diffusion comparison.
- The diffusion square pair is quoted at matched 496K environment steps.
  Its steps-to-0.90 ratio is mean(224,240,256)/mean(128,160,160) = 1.607.
- The FM square pair uses a value-only CAST field. Its precise budget is not
  specified with that row in the source, so no matched-budget or speed claim
  is made from the FM row. The differing-base published-pipeline numbers and
  older diffusion drop-in results are not merged into this pair.
- Drifting can DICE seeds 42 and 43 are below the 20K matched budget; the
  range is marked accordingly. The ICRA text does not claim a complete
  three-seed matched-budget win on can.
- LIBERO means use one shared eta per column. The eta=2 gains are about
  +9.4 points on the drifting base and +2.3 points on the FM base. The source's
  “four FM tasks measured” and eta=1-only “wash” summaries are not carried over.

## Keep separate until run-level reconciliation

The source gives different full-CAST mug/ketchup references across studies:
factorial 0.600/0.627; newer constraint comparison 0.78/0.68; older recipe
0.781/0.684; headline per-task values differ again. These are kept as
explicitly separate study blocks, never merged into a single control row.
The exact checkpoint/config/seed mapping behind those differences remains a
source-level audit item before final submission.

The new `tab:exp1` is the intended three-training-seed constraint comparison.
Its square hinge entry is **0.28** (the adjacent prose rounds it as 0.29).
The older **0.991 provisional single-seed plateau** and older hard-projection
cells are not substituted into this table. The current reward-field block
does show tilted > gradient on red mug (0.697 vs 0.600); the ICRA text
therefore does not repeat the old claim of universal gradient superiority.

## Status and scope

- DMC marks below-budget cells with a circle. The quadruped CAST row is
  both below budget and tuned to eta_Q=1. No blanket untuned-transfer,
  matched-endpoint superiority, or all-completed claim is made.
- The transport loss (CAST 0.594 vs backprop 0.912 at 2.24M steps) and the
  best-of-16 multitask ordering (DICE 0.722 vs CAST 0.692) are stated in the
  main discussion. They are not hidden by the positive-result table merge.
- Real-robot images are teleoperated demonstrations, not autonomous results.
  The compact figure uses committed frames and writes its own selection and
  hash manifest. Both rows show initial/grasp/align/release using
  chest/left-wrist/chest/left-wrist views. The align panels use source frames
  129 (Jigglypuff) and 140 (stacking), rather than the release frames.
  Both tasks have the 20-rollout measurements above; the stacking
  post-release scoring window is not specified in the available record.
- The target-bound and gradient identities are mathematical statements about
  the defined update. They do not assert policy-output bounds, guaranteed
  improvement, or containment in the demonstration support.
