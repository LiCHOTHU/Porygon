# RL Fine-Tuning of Flow-Matching Policies on LIBERO

A small-policy analog of [SimpleVLA-RL](https://arxiv.org/abs/2509.09674): fine-tune a
pretrained **flow-matching** (or 1-step **drift**) behavior-cloning policy on LIBERO with
sparse-reward RL. The base policy is ~18M params and runs on a single GPU, so the RL loop is
cheap enough to iterate on. Two fine-tuners are provided:

- **DICE-RL** — residual `a = a_teacher + actor(s, z)` over a frozen base, with an ensemble
  critic, n-step returns, Q-normalization, RLPD expert replay, and a BC trust-region anchor.
- **GRPO** (SimpleVLA-RL flavor) — critic-free, group-relative advantage from rollouts that
  share an init state, with per-denoising-step PPO clipping.

Built on the [`imitation`](#framework) IL framework (ACT / DP / flow-matching / BAKU encoders,
Hydra-configured). See [`DEVLOG.md`](DEVLOG.md) for experiment notes and results.

## Quickstart

```bash
# install (uv) + LIBERO; see Setup for details
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# fine-tune a flow-matching BC checkpoint on LIBERO task 32 with DICE-RL
uv run dice_train.py cold_start_checkpoint=/path/to/bc.pth 'task_indices=[32]'

# ...or with GRPO
uv run rl_train.py cold_start_checkpoint=/path/to/bc.pth 'task_indices=[32]'
```

Both load a frozen-encoder BC checkpoint (`cold_start_checkpoint`) — train one with the base
framework (see [Framework](#framework)) or supply your own. The reward is LIBERO's sparse binary
terminal success.

## Method

Both fine-tuners start from a frozen-encoder BC policy and update only the action decoder.

**Stochastic sampler (for GRPO).** A deterministic flow sampler has no log-prob, so the Euler
integrator is made stochastic (ReinFlow/DPPO style): at each of the `K` denoising steps,
`x_{k+1} = μ_k + σ·√dt·ε`, giving a tractable per-step Gaussian log-prob. The per-step PPO
log-ratio uses the squared-error difference `(‖x_{k+1}−μ_old‖² − ‖x_{k+1}−μ_new‖²)/(2σ²dt)`
(the Gaussian constant cancels), clipped for stability.

**GRPO advantage.** A *group* is `G ≥ 8` rollouts from the **same** init state; the advantage is
the centered (optionally std-normalized) binary return — so an all-success or all-fail group
contributes no signal, and mixed groups drive the update. Critic-free.

**DICE-RL.** A zero-initialized residual head (exact BC recovery at init) is added to the frozen
teacher action; an ensemble critic with n-step returns and Q-normalization scores actions, a soft
Q-filter gates self-imitation, and a `bc_loss_weight` anchor keeps the residual on-manifold.
Optional RLPD mixes the run's BC demos into replay. (Ported from
[real-stanford/dice-rl](https://github.com/real-stanford/dice-rl).)

**Drift = 1-step flow.** The drifting policy is the `num_inference_steps=1` case of the
flow-matching policy; pass `algo.num_inference_steps=1` to either trainer to fine-tune it.

## RL fine-tuning

| | DICE-RL | GRPO |
|---|---|---|
| entrypoint | `dice_train.py` | `rl_train.py` |
| config | `config/dice_train.yaml` | `config/rl_train.yaml` |
| algo | `imitation/algos/dice/` | `imitation/algos/rl/` |
| critic | ensemble Q | none (group-relative) |

Common overrides (Hydra):

```bash
uv run dice_train.py \
  cold_start_checkpoint=/path/to/bc.pth \
  'task_indices=[32]' \            # one or more LIBERO-90 task indices
  algo.num_inference_steps=1 \     # 1 = drift policy, 10 = flow-matching
  dice.use_rlpd=true \             # mix this run's BC demos into replay
  dice.bc_loss_weight=10 \         # residual trust-region anchor
  dice.n_iters=50

uv run rl_train.py \
  cold_start_checkpoint=/path/to/bc.pth \
  'task_indices=[32]' \
  rl.group_size=8 rl.inits_per_iter=4 \  # GRPO group / #groups per iter
  rl.target_kl=0.03 rl.lr=1e-6 \
  rl.n_iters=100
```

Checkpoints from both trainers share the BC checkpoint schema (config baked in), so they load
back through the standard eval path with no extra flags.

## Setup

Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/yourusername/imitation.git && cd imitation
uv sync
```

LIBERO is required for the RL experiments (it predates `pyproject.toml`, so copy ours in):

```bash
cd .. && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd imitation
cp imitation/envs/libero/pyproject.toml ../LIBERO/
uv pip install -e ../LIBERO
```

<details>
<summary>Optional: point clouds (DGL), MimicGen, DexMimicGen</summary>

```bash
# point-cloud encoders (DP3 / iDP3)
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# MimicGen
cd .. && git clone https://github.com/NVlabs/mimicgen.git && cd imitation
cp imitation/envs/mimicgen/pyproject.toml ../mimicgen/ && uv pip install -e ../mimicgen
```

DexMimicGen needs a separate conda env (incompatible robosuite); comment out every line in its
`requirements.txt` first, then install robosuite v1.5.1 / robosuite_models / dexmimicgen from
source plus `mink==0.0.10` and `qpsolvers[quadprog]`.
</details>

## Data

```bash
uv run scripts/download_libero.py                          # → data/libero/libero_90_unprocessed
uv run scripts/process_libero_data.py task=libero_90_data  # → repo training format
```

Data is assumed to live under `data/` in the repo root; symlink if you store it elsewhere.

## Results

See [`DEVLOG.md`](DEVLOG.md) for the full experiment log. Headlines:

- **RL gain is inverted-U in demo count** — it peaks in the partial-success band (BC ≈ 30–60%)
  and vanishes at both the few-demo extreme (no successful groups → no GRPO signal) and the
  saturated extreme. "RL replaces demos" does not hold uniformly.
- **GRPO beats DICE-RL on the 1-step drift policy**, which is over-dispersed per state; DICE-RL
  is the stronger choice on the multi-step (K=10) flow-matching policy.
- Multi-task BC on LIBERO-90 saturates near **~91%** for both policies, so the hard-task subset
  is where the RL methods actually separate.

These are directional unless stated as powered evals (≥100 rollouts × 3 seeds); see the log.

## Framework

This repo extends a modular IL framework: a policy = swappable observation encoder + action
decoder, all built through Hydra `instantiate` (so a checkpoint stores everything needed to
rebuild and evaluate it). Train a BC base policy and evaluate it with:

```bash
# train a flow-matching BC policy (the RL cold-start)
uv run train.py --config-name=train.yaml task=libero algo=fm_policy_S algo.chunk_size=16

# evaluate any checkpoint (build params are restored from the .pth)
uv run evaluate.py task=libero checkpoint_path=/path/to/ckpt.pth
```

Swap `algo` (`act`, `baku`, `diffusion_policy`, `fm_policy`, …) and `algo/encoder`
(`rgb`, `rgbd`, `dp3`, `idp3`, …) to change the policy / observation stack; use
`export_videos.py` for rollout videos and `--config-name=train_debug.yaml` for a debug run.
Read the [Hydra docs](https://hydra.cc/docs/intro/) before making substantial changes.

## License

MIT — see [LICENSE](LICENSE).
