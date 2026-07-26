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

## Experiment plan — hard-8 baseline comparison

Full comparison on the 8 hardest LIBERO-90 tasks `[8,21,32,53,65,73,75,81]`, multitask bases
(FM BC 0.744 / drift BC 0.637), official DICE budget (100 iters × 32 ep × 800 grad steps),
scored by powered eval (100 rollouts × 3 eval seeds). Launcher:
`scripts/rl_hard8_dice_tier1.sbatch <FM|DRIFT> <hard|guarded> <SEED>`.

Note on filter naming: the paper's Eq. 6 filter (**guarded**: drop the BC anchor where the
critic endorses the edit *and* is not overestimating vs. the MC return) maps to
`use_soft_q_filtering=true`; our **hard** variant keeps only the advantage condition; the
officially *released* configs ship the filter disabled entirely — which is the no-filter
baseline row, and the setting all pre-2026-07 runs unknowingly used.

**Tier 1 — method arms (running):**

| arm | base × seeds | status |
|---|---|---|
| DICE, hard filter | FM+drift × {10000,10001,10002} | seed 10000 mid-run; +2 seeds launched 2026-07-06 |
| DICE, guarded filter (paper Eq. 6) | FM+drift × {10000,10001,10002} | launched 2026-07-06 |

**Tier 2 — baselines:** BC (done); no-filter DICE = released-default (pre-fix RLPD runs, needs
matched re-eval); GRPO FM+drift (checkpoints exist, matched re-eval); plain residual RL
(ResFit/act_sim-style, to launch); DSRL (not implemented — decide vs. citing their LIBERO curve).

**Tier 3 — analysis:** K ∈ {1,4,16} on drift; best-of-N off; finetunability metrics
(GoodCov/BadCov/BadEnt) for FM vs drift bases (local GPU).

## Results

See [`DEVLOG.md`](DEVLOG.md) for the full experiment log. Headlines (as of 2026-07-26):

- **The field-target-regression update on the drifting base ("B": V_Q = clipped analytic ∇ₐQ +
  restore anchor, applied as a regression target instead of backprop-through-critic) beats both
  plain DICE-RL on the same base and the FM + DICE-RL baseline on robomimic**: can 0.99
  (base 0.877, FM-DICE 0.957), square 0.92–0.94 across 3 seeds at the full 20K budget
  (FM-DICE best ≈ 0.86–0.90). First complete-budget square runs ever (paid-QOS wave).
- **Q-source ablation (ABCT)**: with residual health verified, the value-only field variants —
  zeroth-order top-k (C) and exp(Q/τ)-tilted transport (T) — reach only ~0.57–0.61 on square
  vs B's ~0.93. The analytic gradient wins wherever the critic's ∇ₐQ is trustworthy; the
  value-only transports are competitive only in the weak-critic regime (LIBERO hard-8, where
  C-guarded ties B at 0.738; two-regime toy demo shows the mechanism).
- LIBERO hard-8: field arms at 0.738 vs FM-DICE 0.757 — competitive, not yet ahead; tuned
  chains still running.
- Earlier findings (GRPO study): RL gain is inverted-U in demo count; GRPO > DICE-RL on the
  over-dispersed 1-step drift policy, DICE-RL stronger on K=10 flow matching; LIBERO-90
  multi-task BC saturates ~91%, so hard tasks are where methods separate.

Robomimic numbers are 300-episode evals (last-3 average); hard-8 numbers are 20-episode
checkpoint evals (±11pp) pending powered evals (100 rollouts × 3 seeds); see the log.

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

### Repository layout

```
train.py / dice_train.py / rl_train.py   entry points: BC pretraining, DICE/field RL, GRPO RL
evaluate.py / eval_libero_bc.py          checkpoint evaluation (robomimic-style / LIBERO BC)
imitation/                               the framework: algos (dice/, rl/), encoders, envs, datasets
  algos/dice/distill_rl.py              residual DICE-RL trainer (actor modes: residual, dfp, field_*)
  algos/dice/drift_field.py             V_Q / V_BC field construction for the field actor update
config/                                  Hydra configs (config/dice_train.yaml = RL fine-tuning)
scripts/                                 active launchers, evals, tests, and utilities
  field_hard8_drift.sbatch              ABCT hard-8 chains (resume-capable)
  eval_hard8_tier1.sbatch + powered_eval.py   powered evaluation
  test_*.py, smoke_*.py, verify_*.py    field/trainer unit and smoke tests
  toy_field_vs_gradient*.py             two-regime toy demo (paper Fig. 1)
  archive/                              one-off scripts from concluded studies (untracked)
```

Data, checkpoints, and logs live outside the repo (scratch/cedar); `data/` holds symlinks only.

## License

MIT — see [LICENSE](LICENSE).
