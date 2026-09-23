# Recording cues — CAST — supplementary video

**02:59 · narration audio not yet recorded.** Read [voiceover_script.md](voiceover_script.md) against the captioned video. The 14 paragraphs follow its 14 scenes. Cue times are delivery targets for your recording.

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
| 01:06–01:12 | Both policies begin from the same initial state. | Now let’s look at the experiments. In LIBERO, both policies start from the same state. |
| 01:12–01:17 | The right-hand CAST rollout places the bottle in the drawer; the left-hand base fails. | CAST places the bottle in the drawer; the base fails. |
| 01:17–01:21 | Compare the three supplied CAST checkpoints side by side. | To see how learning progresses, robomimic compares three CAST checkpoints. |
| 01:21–01:27 | The middle and final policies place the nut onto the square peg; their last frames remain visible. | The early policy struggles, while later checkpoints complete the nut insertion. |
| 01:27–01:38 | CAST training checkpoints show a walker progressing toward running. | Beyond manipulation, DMC tests locomotion with dense rewards. As CAST learns, the walker progresses from unsteady movement toward running. |
| 01:38–01:41 | Transition from the task demonstrations to the measured simulation results. | The measured results show the same pattern. |
| 01:41–01:50 | Read the right-hand robomimic square chart: steps to the first evaluation reaching 90% success. | In robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL. |
| 01:50–01:55 | The left-hand LIBERO chart summarizes five tasks; no controls and clipping alone yield zero success. | Across five LIBERO tasks, no controls or clipping alone gives zero success, |
| 01:55–01:59 | CAST improves over the base on every task; the displayed bars aggregate the five tasks. | while CAST improves every task over the base. |
| 01:59–02:04 | Chest and wrist views from a recorded robot evaluation. | We next move from simulation to OpenArm, using chest and wrist cameras. |
| 02:04–02:10 | Offline CAST refinement uses recorded successful and failed rollouts. | CAST learns offline from recorded successes and failures before deployment. |
| 02:10–02:15 | Introduce the successful CAST placement and stacking rollouts. | We now show the CAST policies completing placement and stacking. |
| 02:15–02:20 | The robot approaches and grasps the toy on a stair. | For placement, the robot reaches down and grasps Jigglypuff. |
| 02:20–02:23 | The gripper transports the toy over the case. | It carries the toy over the case, |
| 02:23–02:26 | The fingers open; the toy falls fully inside the case. | then releases it fully inside. |
| 02:26–02:30 | The gripper approaches and grasps the red cube. | For stacking, it grasps the red cube, |
| 02:30–02:34 | The held red cube moves above the purple support cube. | then aligns it above the purple cube. |
| 02:34–02:37 | Wrist view shows release and the completed stack. | The gripper opens, completing the stack. |
| 02:37–02:44 | Placement chart from the paper: base 55%, CAST 95%. | Looking at the overall results, CAST raises placement success from 55 to 95 percent. |
| 02:44–02:51 | Stacking chart from the paper: base 35%, CAST 70%. | Stacking improves from 35 to 70 percent, again exceeding both refinement baselines. |
| 02:51–02:55 | Point through the three cards: critic guidance, soft restoration, bounded target steps. | In short, we guide improvement, limit each target step, |
| 02:55–02:59 | Hold the closing title and project address. | and keep the original skill as a reference. |

## Recording against this cut

- Read the [continuous script](voiceover_script.md); its 14 paragraphs match the 14 scenes in the timeline. Use the [phrase-by-phrase cue sheet](voiceover_cue_sheet.md) to time each sentence. Only the script text is spoken.
- Record one scene at a time if helpful. Place each recording at that scene's start; leave natural pauses within its allotted duration. The file `voiceover_timing.csv` gives precise phrase boundaries for editing.
- Figure 2 remains at **00:26–00:56**. The critic gradient is capped at 00:34–00:37; the combined target step is capped around the current action at 00:44–00:50.
- Robot action cues follow the visible grasp, transport/alignment, and release. Both robot clips use continuous 11-second excerpts at normal playback speed.
- Simulation stages play side by side. Their speed labels refer to the supplied videos. Shorter stages hold their completed final frame while longer stages finish playing. Exact source intervals, speed factors, and hold durations are in `source_timing.csv`.
- All times are absolute from the first video frame. The video uses constant **30 fps** and non-drop-frame `HH:MM:SS:FF` timecode. CSV frame ranges are zero-based and end-exclusive: a scene ending at frame 180 is immediately followed by the next scene starting at frame 180.
- The master and captioned rehearsal have identical timing. Narration has not yet been recorded; subtitles are delivery cues, not measured speech timings. The cut is 2:59, leaving 1 second within the three-minute limit.

Exact scene table: [timeline.md](supplementary/timeline.md).

Video: [captioned cut](supplementary/cast_supplementary_video_captioned.mp4). This edit is defined in `supplementary/edit_manifest.json`; CSV timing files are in `supplementary/`.
