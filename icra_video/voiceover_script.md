# CAST — supplementary video — voiceover

We introduce CAST, which improves robot skills while keeping the original policy as a reference.

This matters because the critic can misjudge which actions are better. As orange shows, even small updates can drift away. CAST pulls back, as green shows.

So we keep the original policy fixed and learn a correction. Each action keeps its own reference.

Here’s one learning step. We first sample paired actions. Next, the critic proposes an improvement. We normalize, cap, and scale its gradient. We also pull toward the base, with a stronger pull beyond the reference radius. We combine these arrows and cap the target step around the current action. We then fit the correction to these fixed targets, and repeat the process.

The policy now learns to match the fixed green targets. We refresh them and repeat, while the soft pull leaves room to improve.

Now let’s look at the experiments. In LIBERO, both policies start from the same state. CAST places the bottle in the drawer; the base fails.

To see how learning progresses, robomimic compares three CAST checkpoints. The early policy struggles, while later checkpoints complete the nut insertion.

Beyond manipulation, DMC tests locomotion with dense rewards. As CAST learns, the walker progresses from unsteady movement toward running.

The measured results show the same pattern. In robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL. Across five LIBERO tasks, no controls or clipping alone gives zero success, while CAST improves every task over the base.

We next move from simulation to OpenArm, using chest and wrist cameras. CAST learns offline from recorded successes and failures before deployment. We now show the CAST policies completing placement and stacking.

For placement, the robot reaches down and grasps Jigglypuff. It carries the toy over the case, then releases it fully inside.

For stacking, it grasps the red cube, then aligns it above the purple cube. The gripper opens, completing the stack.

Looking at the overall results, CAST raises placement success from 55 to 95 percent. Stacking improves from 35 to 70 percent, again exceeding both refinement baselines.

In short, we guide improvement, limit each target step, and keep the original skill as a reference.

