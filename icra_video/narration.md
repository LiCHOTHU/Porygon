# CAST — three-minute narration and edit plan

The first 96 seconds explain the idea, method, and simulation evidence. The final 60 seconds cover the real robot and takeaway. The current 156-second draft leaves 24 seconds for incoming simulation footage within the three-minute limit. Timings are embedded in the PowerPoint. These are draft readings, not a recorded voiceover.

| Slide | Time | Duration | Topic | Media |
|---|---|---:|---|---|
| 01 | 00:00–00:06 | 6 s | Improve the skill. Keep what works. | Static |
| 02 | 00:06–00:18 | 12 s | Small steps can still drift. | A01 |
| 03 | 00:18–00:26 | 8 s | Freeze the base. Learn a correction. | A02 |
| 04 | 00:26–00:56 | 30 s | CAST in four steps: animated Figure 2. | A03 |
| 05 | 00:56–01:06 | 10 s | A soft reference, learned through fixed targets. | A04 |
| 06 | 01:06–01:36 | 30 s | Reach useful performance with less experience. | A05 |
| 07 | 01:36–01:52 | 16 s | From recorded experience to robot execution. | Static |
| 08 | 01:52–02:08 | 16 s | Place accurately across changing heights. | V01 |
| 09 | 02:08–02:28 | 20 s | Preserve the approach. Improve the finish. | V02 |
| 10 | 02:28–02:36 | 8 s | Improve the skill. Keep a persistent reference. | Static |

## Narration and sources

### Slide 01 — 00:00–00:06

We introduce CAST, which improves robot skills while keeping the original policy as a reference.

Draft pace: 15 words / approximately 150 words per minute.

Sources: `_ICRA_2027__CAST/sections/01_introduction.tex`.

### Slide 02 — 00:06–00:18

This matters because the critic can misjudge which actions are better. As orange shows, even small updates can drift away. CAST pulls back, as green shows.

Draft pace: 26 words / approximately 130 words per minute.

Sources: `_ICRA_2027__CAST/sections/03_problem_statement.tex`, `_ICRA_2027__CAST/sections/04_analysis.tex`.

**A01 cue:** Animate matched synthetic actions with clipping alone versus the exact CAST restoration and caps. Same base, critic, and step cap.

### Slide 03 — 00:18–00:26

So we keep the original policy fixed and learn a correction. Each action keeps its own reference.

Draft pace: 17 words / approximately 128 words per minute.

Sources: `_ICRA_2027__CAST/sections/03_problem_statement.tex`, `_ICRA_2027__CAST/sections/04_method.tex`.

**A02 cue:** Reveal base actions and learned residual arrows for the same noise samples. Keep the base fixed.

### Slide 04 — 00:26–00:56

Here’s one learning step. We first sample paired actions. Next, the critic proposes an improvement. We normalize, cap, and scale its gradient. We also pull toward the base, with a stronger pull beyond the reference radius. We combine these arrows and cap the target step around the current action. We then fit the correction to these fixed targets, and repeat the process.

Draft pace: 62 words / approximately 124 words per minute.

Sources: `_ICRA_2027__CAST/sections/04_method.tex`, `_ICRA_2027__CAST/sections/04_method_figure.tex`, `_ICRA_2027__CAST/figures/method_update.npz`, `_ICRA_2027__CAST/figures/method_update.json`.

**A03 cue:** Reuse the submitted figure samples and vectors. Reveal the matching equations and zoom into each clockwise stage. Show the raw critic request contracting to the critic cap, then restoration and total clipping around the current action.

### Slide 05 — 00:56–01:06

The policy now learns to match the fixed green targets. We refresh them and repeat, while the soft pull leaves room to improve.

Draft pace: 23 words / approximately 138 words per minute.

Sources: `_ICRA_2027__CAST/sections/04_method.tex`, `_ICRA_2027__CAST/sections/04_analysis.tex`.

**A04 cue:** Show fixed targets during regression; move the actor toward them, then recompute. Illustrate the soft restoration potential without claiming a hard policy-output bound.

### Slide 06 — 01:06–01:36

We test this approach in LIBERO, robomimic, and DMC. These clips introduce the tasks. On robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL. On five LIBERO tasks, updates with no controls or clipping alone have zero success. CAST improves every task over the base, although radial restoration alone is better on two.

Draft pace: 59 words / approximately 118 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_experiments.tex`, `_ICRA_2027__CAST/sections/tables/05_ablations.tex`, `_ICRA_2027__CAST/figures/experiment_chart_data.json`.

**A05 cue:** First nine seconds: LIBERO and robomimic human demonstration replays, then DMC with a scripted controller. Remaining twenty-one seconds: the unchanged measured five-task LIBERO component comparison. Task footage illustrates the environments.

### Slide 07 — 01:36–01:52

We then bring CAST to OpenArm, using chest and wrist cameras. Its flow-matching policy predicts short action sequences. We refine that policy offline, using recorded successes and failures, before testing autonomous placement and stacking.

Draft pace: 34 words / approximately 128 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/sections/05_experimental_supplement.tex`.

### Slide 08 — 01:52–02:08

In placement, the robot must put the toy fully inside the case. CAST increases success from fifty-five percent to ninety-five percent. That’s higher than behavior cloning and DICE-RL refinement.

Draft pace: 29 words / approximately 109 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/real_robot_results.json`.

**V01 cue:** Insert an actual autonomous placement rollout. Current still is a labeled teleoperated illustration, not a CAST evaluation.

### Slide 09 — 02:08–02:28

Stacking adds another challenge: a correct approach can still end in a failed release. With sparse rewards, that final failure makes the earlier actions harder to judge. Here, CAST increases success from thirty-five to seventy percent, again exceeding both refinement baselines.

Draft pace: 41 words / approximately 123 words per minute.

Sources: `_ICRA_2027__CAST/sections/05_real_robot.tex`, `_ICRA_2027__CAST/real_robot_results.json`.

**V02 cue:** Insert an actual autonomous stacking rollout showing release and stable final stack. Current still is teleoperated; no per-method clip is available here.

### Slide 10 — 02:28–02:36

In short, we guide improvement, limit each target step, and keep the original skill as a reference.

Draft pace: 17 words / approximately 128 words per minute.

Sources: `_ICRA_2027__CAST/sections/06_conclusion.tex`.

## Protocol counts awaiting author confirmation

The main physical section says 80 stacking pretraining demonstrations and 30 evaluation attempts. The bundled supplement and result figure retain approximately 60 and 20. This deck deliberately omits those counts. It preserves the reported success percentages (55/80/90/95 for placement; 35/65/60/70 for stacking), rather than inventing new 30-trial counts.

Demonstration stills illustrate the scene only. Replace V01/V02 with correctly attributed autonomous footage before final video export.
