# Jigglypuff Real-Robot Flow Policy

This pipeline trains Porygon's DiT conditional flow-matching policy on the
OpenArm demonstrations in `data_collection_jigglypuff`.

Inputs:

- upright chest RGB
- upright left-wrist RGB
- current left leader joints 1-7 and continuous gripper position
- CLIP embedding of `grasp and drop the jigglypuff toy in the box`

Targets are 30 future absolute left-arm joint/gripper positions sampled at
15 Hz. The policy uses the same conditional path, flipped Beta time sampler,
velocity MSE, and Euler integration as `imitation/algos/fm_policy.py`.

## Train

From the Porygon repository with its Python environment active:

```bash
python -m real_robot.prepare_data
python -m real_robot.prompt
python -m real_robot.train
```

Checkpoints and metrics are written to `real_robot/runs/jigglypuff/`.

## Stacking policy

The same model and training objective are configured in `config_stack.yaml`.
Prompt: `stack the blocks`. Inputs and action layout match the jigglypuff model:
two upright RGB views and current left-arm joints/gripper, predicting 30 future
absolute left-arm joint/gripper targets at 15 Hz.

```bash
.venv/bin/python -m real_robot.prepare_data --config real_robot/config_stack.yaml
.venv/bin/python -m real_robot.prompt --prompt "stack the blocks" \
  --output real_robot/cache/stack/prompt_embedding.npy
.venv/bin/python -m real_robot.train --config real_robot/config_stack.yaml
.venv/bin/python -m real_robot.plot_loss --run-dir real_robot/runs/stack
.venv/bin/python -m real_robot.evaluate_offline real_robot/runs/stack/best.pt
```

The completed 20-epoch run used 28 training takes (3,858 samples) and seven
validation takes (938 samples), split by episode with seed 42. Take
`20260910_171754` was excluded because both RGB videos are missing; exclusions
and the exact split are saved in the cache manifest. The original recordings
still require approximate end-anchored video alignment.

Artifacts in `real_robot/runs/stack/`:
- `best.pt`: epoch 11, sampled validation flow loss 0.37945.
- `latest.pt`: epoch 20, training loss 0.10656 and validation loss 0.41754.
- `loss_curve.png`, `metrics.jsonl`, `config.yaml`, `offline_evaluation.json`.

The train/validation gap indicates limited generalization. Single-seed offline
first-step MAE is 0.0034–0.1213 rad across the seven joints and 0.0062 in gripper
position units. First-step error is worse than the simple hold-current-state
baseline (small next-step changes make that a strong baseline). Full-chunk
improvements are mixed. No real-robot stacking success rate has been measured.

The best checkpoint was also copied to the current Docker container at
`/porygon/real_robot/runs/stack/best.pt` and passed CUDA loading/action sampling.
Select that path in the policy inference GUI to evaluate this model.

## Continued jigglypuff training

`config_jigglypuff_continued.yaml` combines the original and supplementary
demonstrations. It preserves the original eight validation takes and adds
twelve fixed supplementary holdouts: 79 training takes / 20 validation takes,
11,182 / 2,769 samples. Supplementary take `20260911_145513` has an empty joint
CSV and is excluded in the manifest.

The run starts from original `runs/jigglypuff/best.pt` (epoch 17). Neural weights
and prompt are loaded from it; normalization expands to combined **training**
data ranges because some new demonstrations exceed the old action bounds.
AdamW restarts at a lower 3e-5 peak learning rate with warmup and cosine decay.
This is fine-tuning from learned weights with a fresh optimizer/schedule.

```bash
.venv/bin/python -m real_robot.prepare_data --config real_robot/config_jigglypuff_continued.yaml
.venv/bin/python -m real_robot.continue_train --config real_robot/config_jigglypuff_continued.yaml --background
```

Artifacts under `real_robot/runs/jigglypuff_continued/`:
- `train.log`, `train.pid`, `status.json`: background process and progress.
- `metrics.jsonl`: per-epoch flow loss, separate original/supplementary scores.
- `baseline.json`: original checkpoint on the same held-out data.
- `best.pt`: lowest fixed-seed held-out normalized **action** MSE, evaluated
  after epoch 1 and every five epochs. Original best remains incumbent until
  beaten. `latest.pt` is updated atomically each epoch.
- `loss_curve.png`: flow-loss curves and action error against the original baseline.
- `normalization.json`, `data_manifest.json`, `config.yaml`: reproducibility.

The run caps at 200 additional epochs and stops after 40 epochs with no best
action-score improvement. Epoch numbers in these checkpoints count additional
epochs (`parent_epoch` records the original checkpoint epoch). Fixed validation
noise makes comparisons repeatable; action errors use the same combined-training
scale for both old and new models. Old-normalizer flow loss is not directly
comparable to new-normalizer flow loss. These are offline imitation metrics,
not measured real-robot success rates. Keep robot policy execution stopped while
training occupies the shared GPU. An existing run directory is never overwritten.

## Continued stacking training

`config_stack_continued.yaml` starts from the original stacking best checkpoint
(epoch 11) and combines `data_collection_stack` with the user-named
`data_collection_stack_supplementsary` folder. The frozen split contains 35
usable original and 70 supplementary takes: 84 train / 21 validation,
11,935 / 2,744 samples. The original seven validation takes are preserved;
fourteen supplementary takes are held out with seed 42.

The run uses a 3e-5 peak learning rate, expanded training-only normalization,
up to 200 additional epochs, and patience 40. Best-checkpoint selection uses
fixed-seed normalized MSE on the first eight actions, matching the nominal
execution prefix, with separate original/supplementary diagnostics. Full-chunk
and first-step MAE are also recorded. No real-robot success rate is inferred
from those offline errors.

```bash
.venv/bin/python -m real_robot.continue_train \
  --config real_robot/config_stack_continued.yaml --background
```

The active run is in `real_robot/runs/stack_continued/`: `status.json`,
`train.log`, `metrics.jsonl`, `loss_curve.png`, `best.pt`, and `latest.pt`.
An existing run directory is protected against accidental overwriting.

Each improved best checkpoint is also atomically exported to the host's
`openarm_ws/policy_checkpoints/stack_continued/best.pt`, visible in Docker at
`/openarm_ws/policy_checkpoints/stack_continued/best.pt`. The current container
has `/porygon/real_robot/runs/stack_continued` linked to that shared folder, so
the GUI's continued-stacking preset tracks the exported file. Loading remains
explicit: an already running policy is never replaced in memory automatically.
Keep policy execution stopped while training uses the shared GPU.

## Stacking continuation v2

`config_stack_continued_v2.yaml` initializes from `stack_continued/best.pt`
(additional epoch 80, itself descended from original stacking epoch 11).
It adds `/openarm_ws/data_collection_stack_supplement` to the previous original
and `data_collection_stack_supplementsary` datasets. Thirty new takes are usable;
`20260913_213222` and `20260913_213312` have empty joint CSVs and are excluded.
The old 21 validation takes stay held out, together with six new validation
takes. Total: 108 training / 27 validation takes; 14,879 / 3,467 samples.

This run uses a lower 1e-5 peak learning rate, up to 200 additional epochs and
patience 40, selecting by fixed-seed held-out eight-step action error. Results
and logs are under `real_robot/runs/stack_continued_v2/`. Improved checkpoints
are atomically exported to `/openarm_ws/policy_checkpoints/stack_continued_v2/best.pt`
in Docker, also reachable through the current container's
`/porygon/real_robot/runs/stack_continued_v2/best.pt` alias. The GUI has a separate
v2 choice; the original and first continued stacking models remain selectable.
Real stacking success must be assessed with new robot trials, not inferred from
the offline action-error metric. Keep Play stopped while the GPU trains.

## Alignment limitation

The recording process did not save camera timestamps. Each camera stream is
therefore aligned by anchoring its final frame to `ended_at_epoch_s` and using
the declared 15 FPS. This is deterministic but approximate. Future recordings
should preserve a timestamp per video frame.

## Deployment safety

`JigglypuffPolicy.predict_chunk()` returns absolute leader-space targets in
this order: left joints 1-7, then left gripper. Validate limits, controller
mapping, rate, workspace constraints, and emergency-stop behavior before
commanding hardware. These demonstrations record leader state rather than
verified follower commands.
