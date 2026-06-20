# Imitation

A modular imitation-learning framework and a jumping-off point for IL/RL research.
It pairs swappable observation encoders with action decoders (ACT, diffusion,
flow-matching, …) and adds RL fine-tuning of a pretrained flow-matching policy on
LIBERO.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone + sync the environment
git clone https://github.com/yourusername/imitation.git
cd imitation
uv sync
```

Point-cloud support additionally needs DGL (see the [DGL install guide](https://www.dgl.ai/pages/start.html)):

```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

### Optional simulators

<details>
<summary><b>LIBERO</b> (required for the RL experiments below)</summary>

```bash
cd .. && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd imitation
cp imitation/envs/libero/pyproject.toml ../LIBERO/   # LIBERO predates pyproject
uv pip install -e ../LIBERO
```
</details>

<details>
<summary><b>MimicGen</b></summary>

```bash
cd .. && git clone https://github.com/NVlabs/mimicgen.git && cd imitation
cp imitation/envs/mimicgen/pyproject.toml ../mimicgen/
uv pip install -e ../mimicgen
```
</details>

<details>
<summary><b>DexMimicGen</b> (separate conda env — incompatible robosuite version)</summary>

DexMimicGen needs a different robosuite, so install it in its own env. First comment out
every line in the cloned repo's `requirements.txt` (do not skip this), then:

```bash
cd .. && git clone https://github.com/NVlabs/dexmimicgen.git
conda create -n imitation-dmg python=3.10 -y && conda activate imitation-dmg
git clone -b v1.5.1 https://github.com/ARISE-Initiative/robosuite.git && pip install -e robosuite/
git clone https://github.com/ARISE-Initiative/robosuite_models.git && pip install -e robosuite_models/
pip install -e dexmimicgen/
pip install mink==0.0.10 'qpsolvers[quadprog]'
```
</details>

## Hydra

This repo leans heavily on [Hydra](https://hydra.cc/docs/intro/). Every object is built
through `hydra.utils.instantiate`, so a checkpoint stores the full config needed to rebuild
the policy — read the Hydra docs before making substantial changes.

## Data

### LIBERO

```bash
uv run scripts/download_libero.py                          # → data/libero/libero_90_unprocessed
uv run scripts/process_libero_data.py task=libero_90_data  # → repo training format
```

The code assumes data lives under `data/` in the repo root; symlink if you store it elsewhere.

### MimicGen

Download per the [MimicGen instructions](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html)
into `data/mimicgen/core`, then add absolute actions / depth / calibration (slow, hours):

```bash
uv run scripts/process_mimicgen.py \
  --hdf5_path data/mimicgen/core/task_dx.hdf5 \
  --output_dir data/mimicgen/core_depth/task_dx.hdf5 --depth
```

## Training (BC)

```bash
uv run train.py --config-name=train.yaml \
  task=libero algo=diffusion_policy algo/encoder=rgb algo.chunk_size=8
```

Swap `algo` (`act`, `baku`, `fm_policy`, …) and `algo/encoder` (`rgb`, `rgbd`, `dp3`, `idp3`, …)
to change the policy / observation stack. Use `--config-name=train_debug.yaml` for a debug run
(TQDM on, WandB off, unique run dir).

`exp_name` / `variant_name` organize runs: `exp_name` groups an experiment, `variant_name`
identifies one parameter configuration (e.g. group seeds of one config under a shared
`variant_name`). Training auto-checkpoints each epoch (resume by rerunning the same command)
and keeps a non-overwritten checkpoint every 10 epochs.

## Evaluation

Policy build params are saved in the checkpoint, so eval only needs the task and the path:

```bash
uv run evaluate.py task=mimicgen task.task_name=coffee_d1 checkpoint_path=[path]
```

Use `export_videos.py` instead of `evaluate.py` to dump rollout videos. Override saved params
with `+overrides.<key>=<val>` (e.g. `+overrides.temporal_agg=false`). For generalization tests:
`task.robot={UR5e,Kinova3,IIWA}` (unseen embodiment) or `task.cam_shift={small,medium,large}`
(unseen viewpoint).

## RL fine-tuning

Two trainers fine-tune a frozen-encoder flow-matching (or 1-step drift) BC checkpoint on
LIBERO's sparse binary success reward. Both load a BC `cold_start_checkpoint` and freeze the
encoder. The drifting policy is treated as a 1-step flow (`algo.num_inference_steps=1`).

**DICE-RL** — residual `a = a_teacher + actor(s, z)` with an ensemble critic, n-step returns,
Q-normalization, optional RLPD expert replay, and a BC trust-region anchor:

```bash
uv run dice_train.py cold_start_checkpoint=[bc.pth] 'task_indices=[32]'
```

**GRPO / SimpleVLA-RL** — critic-free, group-relative advantage over G rollouts from a shared
init state, with per-denoising-step PPO clipping (a stochastic noised-Euler sampler supplies the
log-probs):

```bash
uv run rl_train.py cold_start_checkpoint=[bc.pth] 'task_indices=[32]'
```

Configs: `config/dice_train.yaml`, `config/rl_train.yaml`. See `DEVLOG.md` for experiment notes.

## Design notes

**Hydra everywhere.** `instantiate` makes every object rebuildable from a dict: all params show
up in WandB, and eval only needs the checkpoint path.

**Modular policies.** A policy = observation encoder + action decoder. The encoder turns
observations into perception/proprioception tokens (declaring count and dim); any decoder
(ACT, DP, FM, …) conditions on arbitrary token sequences, so encoders and decoders mix freely.

**Action representations.** Action keys and dims are declared in the shape meta (unimanual
order: pos, rot, gripper). A `preprocess_actions` hook runs *before* normalization-statistic
computation, so arbitrary action transforms get correct stats for free. Postprocessing is split
into pre- and post-temporal-aggregation stages (e.g. EECF→world before aggregation; axis-angle
conversion after) to keep aggregation in a consistent, continuous frame.

## TODOs

- [ ] Better logging
- [ ] Unify HDF5 loading across LIBERO, MimicGen, DexMimicGen
- [ ] Zarr datasets

## License

MIT — see [LICENSE](LICENSE).
