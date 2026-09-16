# Main-paper experiment evidence and provenance

Simulation source snapshot: `800cfe6` (ICLR manuscript and initial real-robot setup).
The implementation and physical-training protocols were additionally checked against
the current repository on 2026-09-14. The Jigglypuff and stacking evaluations use
counts reported directly by the experimenter, recorded in `real_robot_results.json`;
that file does not identify evaluated checkpoint paths or hashes.

## Main result displays

| ICRA label / file | Source | Transformation |
|---|---|---|
| `tab:robomimic`, `sections/tables/05_robomimic.tex` | `tab:matched` and `tab:baselines` | Seed ranges for drifting can/square; paired diffusion/FM square rows; separate external-reference block |
| `tab:libero-main`, `sections/tables/05_libero.tex` | `tab:libero-single` | All eight tasks, both bases, both shared CAST reward steps; provisional GRPO cells not averaged into this table |
| `tab:ablations`, `sections/tables/05_ablations.tex` | `tab:constraint` | Complete five-task component comparison and means, retaining the original campaign's reference; exact factorial interpretation requires run-config reconciliation |
| `tab:update-design`, `sections/tables/05_update_design.tex` | `tab:exp1`, reward-field part of `tab:locus` | Separate constraint and guidance blocks, each retaining its own reference and aggregation |
| `tab:dmc`, `sections/tables/05_dmc.tex` | `tab:dmc-scope` | All six tasks, with every incomplete/high-variance/tuned marker preserved |
| `fig:real-robot`, `figures/real_robot_tasks.pdf` | ICLR task setup + user-reported evaluations | Fifth-column bar charts replace Table V: four methods per task, 20 trials each, shared percentage scale and count labels |

## Physical evaluation updates

The experimenter reports 20 rollouts per method with Jigglypuff's pose
randomly changed on the stairs: Base FM 11/20 (55%), BC refinement 16/20
(80%), DICE-RL 18/20 (90%), and CAST 19/20 (95%). CAST maps to DICE-RL + CAST. The evaluation record labels the BC condition
"Success-only BC", whereas the newer Jigglypuff training protocol uses supplementary
expert demonstrations. The BC label and evaluated checkpoint linkage remain to be
reconciled; neither the reported count nor its source is changed here.
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
component study 0.600/0.627; newer constraint comparison 0.78/0.68; older recipe
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

- The experimental narrative follows three hypotheses: improvement (H1),
  the contribution of restoration beyond step clipping (H2), and the balance
  of preservation and adaptation (H3), followed by separate scope tests. The current
  tables are manuscript-source checked, not independently rederived from
  unavailable cluster evaluations. Interpretation limits from the code audit
  are recorded below.
- The current physical-training documents supersede the ICLR setup's planned
  40/80 base demonstrations, 40 positive BC trajectories, and 40-attempt RL
  pools as descriptions of the implemented training campaigns. They document
  offline refinement and task-specific datasets, as detailed below. The
  supplied autonomous counts remain unchanged; their association with these
  campaigns' final checkpoints is awaiting experimenter confirmation.
- Containment endpoints and critic diagnostics are not promoted to the main
  tables. The containment script uses older success references, can overwrite
  iterations across matching runs, and reads global RMS rather than the
  caption's mean per-particle RMS. Shared-critic sensitivity captions also
  contain inconsistent percentages. These require source-log reconciliation.
- The shared-critic best-of-16 DICE row links to the published-recipe
  single-draw mean (.688), not the matched-recipe mean (.660). The .722/.692
  result is reported as a system-level counterexample, not an actor-only
  intervention. Its source +.054 delta requires unrounded verification
  against displayed .722 and .669; the ICRA text uses endpoints only.
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

## Implementation reconciliation (2026-09-14)

The source tables contain measured outcomes, while launchers establish intended
configurations. Neither launchers nor development notes replace each measured
run's saved configuration. No local cluster run manifests were found for these
simulation table cells, so discrepancies below narrow the claims; they do not
change measurements or silently infer which configuration produced a cell.

### Shared DICE training recipe

- `dice_train.py:94` loads `config/dice_train.yaml`, whose lines 69--71 set
  `use_n_step: true` and `n_step: 3`. In the current
  `scripts/field_single_task.sbatch`, arm A inherits these values and arm B
  repeats them explicitly. The current defaults therefore agree. However,
  `DEVLOG.md:1005--1022` describes three-step returns as part of an earlier
  recipe change; historical A/B run configurations remain unverified.
- The matched comparison concerns the **actor-update package** within the
  shared residual architecture and training procedure. DICE applies expert-data
  Q masking and Q-filtered BC (`distill_rl.py:651--710`); the field loss receives
  no data-source mask (`distill_rl.py:833--846,1178--1203`). Identical training
  procedures do not imply identical learned critic weights or collected replay.
  Do not attribute the result solely to regression or stop-gradient: the local
  gradient identity in `sections/04_analysis.tex` rules out that interpretation.

### Restoration and clipping components

- Full B enables the weak pointwise pull (`bc_step_size=0.05`), dead-zone
  restoration (`restore_step_size=1`), and both clipping stages
  (`scripts/field_single_task.sbatch:23--26`). B_CLIP and B_NONE disable both
  restoring terms (lines 55--59 and 65--69). B_ANC enables dead-zone restoration
  but sets the weak pull to zero as well as disabling both clips (lines 60--64).
- Thus the committed launcher does **not** implement an exact 2x2 in which both
  restoring terms stay fixed when clipping is toggled. Bare B_ANC launch commands
  in `DEVLOG.md:1425--1428` and `EXPERIMENT_PLAN.md:118--124` do not document a
  correcting override; the launcher permits further overrides, so saved run
  configurations are needed to resolve what was executed.
- Defensible table interpretation: the two arms without restoration have zero
  reported success; restoration variants retain nonzero success; the full package
  exceeds its base on all five tasks, while restoration alone wins on two.
  The full-versus-restoration-only contrast cannot isolate clipping's contribution
  until the weak-pull setting is reconciled. Success outcomes alone do not
  establish reduced critic error or bounds on learned policy outputs.

### Constraint implementation and reward fields

- The LIBERO hinge launcher uses DICE's residual actor with `bc_hinge_rho=0.05`
  (`scripts/exp1_arm.sbatch:37--38`). The penalty remains inside DICE's filtering
  and weighting logic (`distill_rl.py:665--711`). The projection launcher removes
  both restoring terms **and both clipping stages**, then projects the residual
  target to a base-centered RMS ball (launcher lines 39--44;
  `distill_rl.py:998--1009`). This is a comparison of implemented alternatives,
  not a single-variable test of constraint location or soft versus hard control.
  Square launch overrides differ from the LIBERO launcher
  (`scripts/guard/resubmit_paper_jobs.sh:129--137`); its external training configs
  are not available here for full reconstruction.
- The field-loss API can vary `q_source` independently. Its gradient field uses
  normalized critic derivatives; top-k transport uses the highest-valued current
  particles as attractors; tilted transport weights the same cloud by softmax of
  per-state standardized Q values (`distill_rl.py:890--917`). These are not
  equivalent to deploying a top-k action selector.
- The available older hard-8 launchers also change the auxiliary base field:
  gradient arms use `field_pointwise`, while top-k and tilted arms use
  `field_distributional` (`scripts/field_hard8_drift.sbatch:58--70`). Development
  notes additionally describe radius 0 for the winning square gradient recipe
  and radius 0.05 for the value-only variants (`DEVLOG.md:751--754,768--773`).
  Their exact mapping to the reported single-task reward-field table remains
  unverified. Describe the measured ordering as a comparison of field variants;
  do not claim code-verified isolation of critic information or universal gradient
  superiority. Tilted exceeds the gradient reference on red mug.

### Current physical-training campaigns

- **Jigglypuff:** `real_robot/FOUR_WAY_EXPERIMENT.md:5--24` documents the original
  epoch-17 base; BC fine-tunes on 30 supplementary expert demonstrations; both
  RL arms use those demonstrations plus 18 reviewed saved attempts (11 successes,
  seven failures), collected by a stronger continued-BC checkpoint. Thus the
  implemented BC condition is supplementary demonstration refinement, not
  success-only imitation of the same mixed-outcome practice pool.
- **Stacking:** `real_robot/STACK_FOUR_WAY.md:3--38` documents one frozen v2 base;
  BC uses the 12 successful saved attempts; both RL arms use all 24 attempts
  (12 successful, 12 failed) plus 30 demonstrations from the base's training split.
  BC fine-tunes the base/encoder; both RL arms freeze them and train a residual.
- Both documents report completed 40-epoch BC runs and 6,000-update RL runs.
  The RL recipe uses 500 critic-only warmup steps, actor updates every two steps,
  batch 32 with 50% expert replay, five critics, 16 particles, and shared optimizer
  settings. `real_robot/offline_rl.py:33--52` precomputes identical replay-index
  schedules from the same seed, independently of actor RNG use; lines 88--98
  switch only the actor-loss branch. Stacking's completion record reports that
  all 6,000 minibatches and frozen-base tensors were checked.
- These records support a matched **training** design for DICE versus CAST.
  `icra2027/real_robot_results.json` contains only experimenter-reported counts,
  without checkpoint IDs. Confirmation that the 20-trial evaluations used these
  fixed-budget final checkpoints is still needed before claiming an audited
  matched-data physical **evaluation**. BC versus RL also changes available
  outcomes, supervision, and trainable parameters, so it does not isolate an
  objective change or establish equal collection effort.
- The robot adapter restricts residual corrections and critic inputs to the
  nominal eight-action executed prefix of a 30-action FM chunk
  (`real_robot/residual_policy.py:10--30`); actor targets retain same-noise pairing.
  The critic is noise-independent. Legacy image alignment is approximate and
  exact inference-chunk boundaries were not logged (both protocol documents).
  Offline imitation errors are diagnostics and never substitute for autonomous
  success counts.

### Hypothesis-to-evidence map

| Hypothesis | Evidence that can address it | Limit on interpretation |
|---|---|---|
| H1: CAST can improve a competent base and its refinement efficiency | Within-comparison gains, square threshold-crossing steps, visual/base-family comparisons, and physical counts | Matched actor-package contrasts where documented; different baselines/harnesses remain separate; robot checkpoint linkage unresolved |
| H2: A reference to the base contributes beyond a per-step cap | Zero success in both no-restoration arms versus nonzero restoration variants | Not an audited exact 2x2; no direct proof of critic-error reduction or output containment |
| H3: Practical constraint and field choices affect refinement | Hinge/projection/CAST ordering, field variants, and exploratory radius/reward-step sweeps | Implemented packages can change multiple factors; preserve campaign-specific references and counterexamples |
| Scope: Benefits depend on the refinement setting | DMC dense/sparse tasks, transport loss, and shared-critic best-of-16 reversal | Incomplete or tuned DMC cells preclude a blanket matched-endpoint claim; these tests do not isolate data coverage as a causal mechanism |

## Expanded analysis and percentage presentation (2026-09-14)

- The experimenter reports that the main observed stacking failure is failure
  to release the gripper after otherwise correct reaching, grasping, and cube
  alignment. This is a qualitative observation supplied in the conversation;
  no per-method failure-category counts or stage-wise critic diagnostics were
  supplied. It is not inferred from the teleoperated figure frames.
- The physical analysis uses this observation to motivate a credit-assignment
  explanation for CAST 70%, BC 65%, and DICE-RL 60%. Terminal failure receives
  zero reward; earlier transitions bootstrap future values
  (`real_robot/prepare_offline_replay.py:63`, `real_robot/offline_rl.py:81--85`).
  The code does not label every earlier action incorrect. Low predicted return
  can be correct under a failed continuation; reliability of the local action
  gradient is a separate issue. Neither critic error nor release-specific
  improvement by CAST is established by the success counts.
- Success-only stacking BC supplies action supervision including successful
  release. CAST's restoration and clipping apply across action dimensions, with
  no special release-stage mechanism or extra gripper weight. The proposed
  preservation of earlier useful motion is an interpretation, not a measured
  per-stage intervention effect. Failed episodes remain valid learning data.
- Success tables and corresponding prose now use percentages; success changes
  use percentage points (pp). All original values, seed ranges, and run markers
  are preserved. DMC remains episode return on its original 0--1000 scale.
- The user explicitly permits exceeding eight pages to improve the explanation.
  No class, geometry, or font-size reduction is introduced to enforce the old
  working page budget.

## Final manuscript writing pass (2026-09-14)

- The abstract, introduction, and conclusion use the same claim scope: a
  persistent base reference can support refinement; neither target clipping nor
  residual proximity guarantees improved learned behavior.
- The constraint comparison now states the taskwise finding explicitly.
  CAST improves on all three bases, while projection degrades ketchup; it does
  not claim a higher mean than projection (80.0% versus 81.0%).
- The orphaned GRPO paragraph was omitted from the ICRA narrative because its
  provisional cells have no corresponding ICRA result display or full protocol.
  The source measurements remain in the ICLR draft; no table cells were changed.
- “Backprop” in the DMC table is labeled DICE-RL, matching the source baseline
  definition and the surrounding prose. “+ CAST” is standardized to CAST in
  the robomimic table; the setup explains the relationship to DICE-RL.
- The physical explanation retains the gripper-release observation, small-sample
  interpretation, and unresolved evaluation/checkpoint linkage. BC--RL
  differences in data, objectives, and trainable parameters are explicit.
