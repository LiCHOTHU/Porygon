# CAST — paper explanation voiceover

We introduce CAST, which improves robot skills while keeping the original policy as a reference.

This matters because the critic can misjudge which actions are better. As orange shows, even small updates can drift away. CAST pulls back, as green shows.

So we keep the original policy fixed and learn a correction. Each action keeps its own reference.

Here’s one learning step. We first sample paired actions. Next, the critic proposes an improvement. We normalize, cap, and scale its gradient. We also pull toward the base, with a stronger pull beyond the reference radius. We combine these arrows and cap the target step around the current action. We then fit the correction to these fixed targets, and repeat the process.

The policy now learns to match the fixed green targets. We refresh them and repeat, while the soft pull leaves room to improve.

We test this approach in LIBERO, robomimic, and DMC, covering manipulation and continuous control. On robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL. On five LIBERO tasks, updates with no controls or clipping alone have zero success. CAST improves every task over the base, although radial restoration alone is better on two.

In short, we guide improvement, limit each target step, and keep the original skill as a reference.

