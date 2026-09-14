# Stacking refinements from a fixed v2 base

The base is a frozen copy of `stack_continued_v2/best.pt`, additional epoch 40.
It is copied to `runs/stack_four_way/base.pt` before any refinement and identified
by SHA-256 in `experiment.json`. No method starts from the original stacking
checkpoint or from another refinement's output.

## Data and objectives

The saved `policy_evaluations/stack` pool contains 24 usable operator-reviewed
attempts: 12 successful and 12 failed. All logged behavior checkpoint epochs are
40 and their condition is stacking v2. Pending/discarded takes are not included.

- **BC refinement:** only the 12 successful saved evaluations (2,330 samples).
  Input proprioception is actual follower state; targets are future commands
  from `policy_commands.csv`, sampled with zero-order hold. Stationary leader
  CSVs are not used as policy-action targets. Full FM/encoder weights are
  fine-tuned, starting from the frozen base snapshot, with its normalization and
  prompt retained. Budget: 40 additional epochs at LR 1e-5.
- **DICE-RL:** all 24 evaluations, plus a seed-42 subset of 30 demonstrations
  from the base's TRAINING split as expert replay. Base and encoder are frozen;
  residual actor and ensemble critic are trained with the existing Porygon loss.
- **DICE-RL + CAST:** the identical replay, initialization seed, batch schedule,
  critic and budget, replacing only the actor objective with `field_pointwise`
  critic-gradient targets, clipping and base restoration.

Both RL runs use the same 6,000-update recipe as the earlier Jigglypuff study:
500 critic-only warmup steps, actor every two updates, batch 32, 50% expert
replay, five critics with 256×256 hidden layers, 16 action particles, actor and
critic LR 1e-4, CQL weight 0.1, gamma 0.99, and Polyak coefficient 0.01.
CAST settings: reward step 1, weak restoration 0.05, dead-zone restoration 1,
radius 0.05 RMS, Q-field clip 1 and combined displacement clip 0.15.

This is offline refinement from existing data, not new online robot interaction.
BC and RL deliberately use different refinement data (success-only versus both
outcomes with expert replay); the clean matched actor comparison is DICE versus
CAST. Expert demonstrations are treated as successful per the curated training
set, while the 24 online labels are explicitly supplied by the operator.

## Replay and limitations

Replay has 1,344 macro transitions from 54 episodes: 625 expert transitions and
719 rollout transitions. There are 42 positive terminal rewards (30 expert +
12 reviewed successes), and 12 failed terminals. No transition crosses episodes.
Eight actual logged commands form a macro action; the last full window can
overlap its predecessor to retain terminal labels without padded fake commands.
Discount exponents use elapsed time in nominal 15 Hz steps.

Terminal next-state fields are explicit absorbing placeholders that are NEVER
bootstrapped (`done=1`). This preserves terminal rewards when the last sensor
sample occurs a few milliseconds before the final command; it does not invent
a measured post-command observation. Nonterminal transitions still require
observed state coverage.

The FM base retains its full 30-step noise/action representation. The residual
acts on the nominal eight-step prefix and the critic ignores the unused suffix.
Deployment returns the full base chunk plus prefix corrections, with the common
GUI's asynchronous execution and latency compensation. This is the same
real-robot adaptation used for Jigglypuff, not an exact reproduction of every
published online DICE-RL setting.

Videos still lack per-frame capture timestamp sidecars, so alignment is
end-anchored and approximate. Behavior latent samples and exact inference-chunk
boundaries were not recorded; the critic is noise-independent. These limitations
apply to both RL arms and must be disclosed in any reported experiment.

## Outputs and evaluation

`real_robot/runs/stack_four_way/` contains the frozen data manifest, base snapshot,
BC cache config, replay/report, per-method logs/checkpoints, `pipeline_status.json`
and final `offline_comparison.json`. The 27 base-validation demonstrations are
held out from all refinements; only base-training demos enter expert replay.
Offline diagnostics compare the first eight actions with fixed noise and common
normalization. They are not real-robot success rates and do not fill paper cells.

For fixed-budget comparison use `bc_refinement/final.pt`, `dice_rl/final.pt`, and
`cast/final.pt`. BC's separate `best.pt` is validation-selected and can differ.

The runner exports inference-only snapshots into the shared Docker workspace:
`/openarm_ws/policy_checkpoints/stack_four_way/`. Optimizers and critics are
omitted from these deployment files; the trained base/residual weights and all
inference configuration remain. The training checkpoints retain full state.
The current container exposes the export through
`/porygon/real_robot/runs/stack_four_way/`. GUI entries identify all four stacking
comparison conditions and store that condition with operator Success/Fail labels.

Commands from the Porygon root:

```bash
.venv/bin/python -m real_robot.prepare_stack_comparison
.venv/bin/python -m real_robot.prepare_offline_replay --experiment-dir real_robot/runs/stack_four_way
.venv/bin/python -m real_robot.run_refinements --experiment-dir real_robot/runs/stack_four_way --background
```

The supervisor runs BC, then DICE, then CAST and stops on any failure. Existing
experiment manifests and unfinished runs are protected against silent overwrite.

## Completed run

BC completed 40 additional epochs. DICE-RL and CAST each completed 6,000 updates
sequentially. Both RL checkpoints preserve every frozen-base tensor exactly,
and their 6,000×32 minibatch-index arrays are identical. Inference-only exports
were checked against the full training checkpoint weights and loaded through
the Docker GUI.

Held-out eight-step normalized action MSE on the same 3,467 demonstration samples:

| Condition | Offline action MSE |
|---|---:|
| Frozen stacking v2 base | 0.010094 |
| Success-only BC refinement, epoch 40 | 0.009940 |
| DICE-RL, update 6,000 | 0.010009 |
| DICE-RL + CAST, update 6,000 | 0.009808 |

The differences are small and these are imitation diagnostics, not measured
stacking success rates. The GUI provides four explicitly labeled stacking
comparison conditions for subsequent physical trials with operator outcome labels.
