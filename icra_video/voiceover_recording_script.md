# CAST — timestamped reading script

**Final cut: 02:59.** Read only the paragraphs aloud. Headings give the exact scene times; pause naturally within each interval. For sentence-level timing, use [the cue sheet](voiceover_cue_sheet.md).

## 00:00–00:06 · The idea

We introduce CAST, which improves robot skills while keeping the original policy as a reference.

## 00:06–00:18 · Why anchoring matters

This matters because the critic can misjudge which actions are better. As orange shows, even small updates can drift away. CAST pulls back, as green shows.

## 00:18–00:26 · Policy architecture

So we keep the original policy fixed and learn a correction. Each action keeps its own reference.

## 00:26–00:56 · Method and equations

Here’s one learning step. We first sample paired actions. Next, the critic proposes an improvement. We normalize, cap, and scale its gradient. We also pull toward the base, with a stronger pull beyond the reference radius. We combine these arrows and cap the target step around the current action. We then fit the correction to these fixed targets, and repeat the process.

## 00:56–01:06 · Learn and repeat

The policy now learns to match the fixed green targets. We refresh them and repeat, while the soft pull leaves room to improve.

## 01:06–01:17 · LIBERO: place the bottle in the drawer

Now let’s look at the experiments. In LIBERO, both policies start from the same state. CAST places the bottle in the drawer; the base fails.

## 01:17–01:27 · robomimic: learning square-nut insertion

To see how learning progresses, robomimic compares three CAST checkpoints. The early policy struggles, while later checkpoints complete the nut insertion.

## 01:27–01:38 · DMC walker-run: locomotion learning

Beyond manipulation, DMC tests locomotion with dense rewards. As CAST learns, the walker progresses from unsteady movement toward running.

## 01:38–01:59 · Simulation evidence

The measured results show the same pattern. In robomimic square, CAST reaches 90 percent success using 38 percent fewer environment steps than DICE-RL. Across five LIBERO tasks, no controls or clipping alone gives zero success, while CAST improves every task over the base.

## 01:59–02:15 · Real robot setup

We next move from simulation to OpenArm, using chest and wrist cameras. CAST learns offline from recorded successes and failures before deployment. We now show the CAST policies completing placement and stacking.

## 02:15–02:26 · Successful placement example

For placement, the robot reaches down and grasps Jigglypuff. It carries the toy over the case, then releases it fully inside.

## 02:26–02:37 · Successful stacking example

For stacking, it grasps the red cube, then aligns it above the purple cube. The gripper opens, completing the stack.

## 02:37–02:51 · Reported robot results

Looking at the overall results, CAST raises placement success from 55 to 95 percent. Stacking improves from 35 to 70 percent, again exceeding both refinement baselines.

## 02:51–02:59 · Takeaway

In short, we guide improvement, limit each target step, and keep the original skill as a reference.
