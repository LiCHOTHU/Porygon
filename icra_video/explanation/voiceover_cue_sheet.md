# Recording cues — paper explanation

**1:44 · narration audio not yet recorded.** Read [voiceover_script.md](voiceover_script.md) against the captioned video. The seven paragraphs follow its seven scenes. Cue times are delivery targets for your recording.

| Time | Scene / on-screen event | Spoken text |
|---|---|---|
| 00:00–00:06 | Title and OpenArm scene; introduce the problem. | We introduce CAST, which improves robot skills while keeping the original policy as a reference. |
| 00:06–00:11 | A01: the toy critic curve predicts which actions are valuable. | This matters because the critic can misjudge which actions are better. |
| 00:11–00:15 | Orange: repeated clipped updates drift away from the blue base. | As orange shows, even small updates can drift away. |
| 00:15–00:18 | Green: restoration keeps the updated policy near the base. | CAST pulls back, as green shows. |
| 00:18–00:23 | A02: frozen base plus learned residual; point to the sum. | So we keep the original policy fixed and learn a correction. |
| 00:23–00:26 | The paired samples share the same observation and noise. | Each action keeps its own reference. |
| 00:26–00:28 | A03: clockwise overview of Figure 2. | Here’s one learning step. |
| 00:28–00:31 | Sample panel: base/current pairs and residual arrows. | We first sample paired actions. |
| 00:31–00:37 | Critic panel: normalized gradient appears, then contracts to its cap. Scaling follows the cap. | Next, the critic proposes an improvement. We normalize, cap, and scale its gradient. |
| 00:37–00:40 | Restoration panel: weak pull toward the paired base action. | We also pull toward the base, |
| 00:40–00:44 | Additional radial restoration appears beyond the reference radius. | with a stronger pull beyond the reference radius. |
| 00:44–00:50 | Combine the vectors and clip around the CURRENT action. The circle bounds the target step. | We combine these arrows and cap the target step around the current action. |
| 00:50–00:54 | Fit panel: regression to fixed residual targets; squared-error equation. | We then fit the correction to these fixed targets, |
| 00:54–00:56 | Return to the complete clockwise figure; leave a short breath. | and repeat the process. |
| 00:56–01:01 | A04: the actor moves toward the fixed green squares; fitting error falls. | The policy now learns to match the fixed green targets. |
| 01:01–01:03 | The animation constructs fresh targets for another fitting round. | We refresh them and repeat, |
| 01:03–01:06 | Right-hand curve: increasing restoration penalty, with a soft threshold. | while the soft pull leaves room to improve. |
| 01:06–01:15 | Three benchmark settings: LIBERO images/sparse rewards; robomimic state/sparse rewards; DMC continuous control/dense rewards. | We test this approach in LIBERO, robomimic, and DMC, covering manipulation and continuous control. |
| 01:15–01:24 | Read the right-hand robomimic square efficiency chart; the left-hand LIBERO bars finish appearing. | On robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL. |
| 01:24–01:30 | Left-hand LIBERO chart: no controls and clipping only both show zero. These are the five tested tasks. | On five LIBERO tasks, updates with no controls or clipping alone have zero success. |
| 01:30–01:36 | CAST improves over the base on every task; the displayed bars aggregate five tasks. Radial-only beats CAST on two individual tasks. | CAST improves every task over the base, although radial restoration alone is better on two. |
| 01:36–01:40 | Point through the three cards: critic guidance, soft restoration, bounded target steps. | In short, we guide improvement, limit each target step, |
| 01:40–01:44 | Hold the closing title and project address. | and keep the original skill as a reference. |
