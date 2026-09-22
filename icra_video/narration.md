# CAST — three-minute narration and edit plan

The first 120 seconds explain the idea, method, and simulation evidence. The final 60 seconds cover the real robot and takeaway. Timings are embedded in the PowerPoint. These are draft readings, not a recorded voiceover.

| Slide | Time | Duration | Topic | Media |
|---|---|---:|---|---|
| 01 | 00:00–00:10 | 10 s | Improve the skill. Keep what works. | Static |
| 02 | 00:10–00:30 | 20 s | Small steps can still drift. | A01 |
| 03 | 00:30–00:45 | 15 s | Freeze the base. Learn a correction. | A02 |
| 04 | 00:45–01:10 | 25 s | Propose. Restore. Limit the step. | A03 |
| 05 | 01:10–01:32 | 22 s | A soft reference, learned through fixed targets. | A04 |
| 06 | 01:32–02:00 | 28 s | Reach useful performance with less experience. | A05 |
| 07 | 02:00–02:16 | 16 s | From recorded experience to robot execution. | Static |
| 08 | 02:16–02:32 | 16 s | Place accurately across changing heights. | V01 |
| 09 | 02:32–02:52 | 20 s | Preserve the approach. Improve the finish. | V02 |
| 10 | 02:52–03:00 | 8 s | Improve the skill. Keep a persistent reference. | Static |

## Narration and sources

### Slide 01 — 00:00–00:10

A robot can position an object correctly, yet fail at release. CAST improves the skill while preserving behavior that already works.

Draft pace: 21 words / approximately 126 words per minute.

Sources: `_ICRA_2027__CAST/sections/01_introduction.tex`.

### Slide 02 — 00:10–00:30

A critic predicts which actions are valuable. Here, its predictions favor actions far from the useful region. Watch the action distributions shift during optimization. Clipping keeps each correction small, but repeated corrections still drift. CAST restores toward the frozen base while allowing a useful adjustment. This is an illustrative toy example.

Draft pace: 51 words / approximately 153 words per minute.

Sources: `_ICRA_2027__CAST/sections/03_problem_statement.tex`, `_ICRA_2027__CAST/sections/04_analysis.tex`.

**A01 cue:** Animate matched synthetic actions with clipping alone versus the exact CAST restoration and caps. Same base, critic, and step cap.

### Slide 03 — 00:30–00:45

We freeze the generative policy and learn an additive residual. The same observation and noise pair each current action with its base action, giving corrections a consistent reference without policy likelihoods.

Draft pace: 31 words / approximately 124 words per minute.

Sources: `_ICRA_2027__CAST/sections/03_problem_statement.tex`, `_ICRA_2027__CAST/sections/04_method.tex`.

**A02 cue:** Reveal base actions and learned residual arrows for the same noise samples. Keep the base fixed.

### Slide 04 — 00:45–01:10

First, the critic proposes an improvement. We normalize and cap its gradient, then scale the proposal. A weak pull always points toward the paired base action. Beyond a soft reference radius, an additional radial pull becomes active. We add these fields and cap the final step from the current action. The resulting target teaches the residual policy.

Draft pace: 57 words / approximately 137 words per minute.

Sources: `_ICRA_2027__CAST/sections/04_method.tex`.

**A03 cue:** Animate the normalized critic proposal, paired weak/radial restoration, and combined displacement cap. Restoration is evaluated at the current action.

### Slide 05 — 01:10–01:32

The squares are fixed targets. The white actor learns to match them by regression; its fitting error decreases. Then we construct new targets and repeat. Restoration corresponds to a penalty on departure from the base. With clipping inactive, the first regression gradient follows regularized critic optimization. The reference stays soft, leaving room to improve.

Draft pace: 54 words / approximately 147 words per minute.

Sources: `_ICRA_2027__CAST/sections/04_method.tex`, `_ICRA_2027__CAST/sections/04_analysis.tex`.

**A04 cue:** Show fixed targets during regression; move the actor toward them, then recompute. Illustrate the soft restoration potential without claiming a hard policy-output bound.

### Slide 06 — 01:32–02:00

On robomimic square, CAST reaches ninety percent success with thirty-eight percent fewer environment steps than DICE-RL, while final success is similar. In a separate five-task LIBERO study, both updates without restoration lose all task success, including clipping alone. Full CAST improves every task over its base, although radial restoration alone is better on two tasks.

Draft pace: 55 words / approximately 118 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_experiments.tex`, `_ICRA_2027__CAST/sections/tables/05_ablations.tex`, `_ICRA_2027__CAST/figures/experiment_chart_data.json`.

**A05 cue:** Animate equal-status bars for the base and four component configurations using the five-task mean success from the submitted paper. This is an aggregate comparison, not a learning curve.

### Slide 07 — 02:00–02:16

We evaluate OpenArm with chest and wrist cameras, joint state, and a parallel gripper. A flow-matching policy predicts action chunks. Refinement trains offline on recorded successes and failures, then we evaluate placement and stacking autonomously on a staircase scene.

Draft pace: 39 words / approximately 146 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/sections/05_experimental_supplement.tex`.

### Slide 08 — 02:16–02:32

Placement requires putting the toy fully inside the case. CAST raises success from fifty-five to ninety-five percent, compared with eighty percent for BC refinement and ninety percent for DICE-RL, all starting from the same base.

Draft pace: 35 words / approximately 131 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/real_robot_results.json`.

**V01 cue:** Insert an actual autonomous placement rollout. Current still is a labeled teleoperated illustration, not a CAST evaluation.

### Slide 09 — 02:32–02:52

Stacking requires alignment and a clean release. A common observed failure is reaching the correct position without releasing the cube. The failed outcome does not mean every preceding action was wrong. CAST reaches seventy percent success, versus thirty-five for the base, sixty-five for BC refinement, and sixty for DICE-RL.

Draft pace: 49 words / approximately 147 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/real_robot_results.json`.

**V02 cue:** Insert an actual autonomous stacking rollout showing release and stable final stack. Current still is teleoperated; no per-method clip is available here.

### Slide 10 — 02:52–03:00

CAST combines critic guidance, soft restoration, and bounded target steps: improve the skill while keeping a persistent reference.

Draft pace: 18 words / approximately 135 words per minute.

Sources: `_ICRA_2027__CAST/sections/06_conclusion.tex`.

## Protocol counts awaiting author confirmation

The main physical section says 80 stacking pretraining demonstrations and 30 evaluation attempts. The bundled supplement and result figure retain approximately 60 and 20. This deck deliberately omits those counts. It preserves the reported success percentages (55/80/90/95 for placement; 35/65/60/70 for stacking), rather than inventing new 30-trial counts.

Demonstration stills illustrate the scene only. Replace V01/V02 with correctly attributed autonomous footage before final video export.
