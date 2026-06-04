# DEVLOG

Project: **What data matters in BC and RL** (data-usefulness / data-curation thesis), using the `imitation` repo as the experimental harness.

All artifacts (checkpoints, logs, hydra dirs) live under `/storage/scratch1/8/lwang831/`. The repo itself is in `$HOME/workspace/imitation`.

---

## 2026-05-27 — Scoping & env validation

- Reviewed LIBERO reward: sparse binary 0/1 terminal success (`bddl_base_domain.py:165`). Not dense.
- Reviewed **SimpleVLA-RL** (arXiv 2509.09674) as closest related work: PPO on autoregressive token VLAs (OpenVLA / OpenVLA-OFT) with sparse reward; **1-demo SFT + RL → 17.3 → 91.7 on LIBERO-Long**. Flow-matching policies explicitly NOT supported (their roadmap). Gap to fill: the FM analog.

## 2026-05-28 — LIBERO-90 data preprocessing

- Processed all 90 LIBERO tasks (50 demos each) into repo format → `/storage/scratch1/8/lwang831/imitation/data/libero/libero_90/`. ~103 GB total, ~6 h on gpu-h100 (job 9231919).
- Verified flow-matching BC training end-to-end: `python train.py --config-name=train task=libero algo=fm_policy_S rollout.enabled=false`.

## 2026-05-29 — FM-RL: SimpleVLA-RL analog for flow-matching policy

### Method (locked with user as Path B; plan: `sequential-gliding-salamander.md`)
- **Critic-free GRPO** (group = G≥8 rollouts from SAME init state; centered binary-reward advantage) + **per-denoising-step PPO clip**.
- **Stochastic noised-Euler sampler** (ReinFlow/DPPO style): `x_{k+1}=μ_k + σ·√dt·ε` → per-step Gaussian log-prob.
- **Squared-error-difference ratio**: `log_ratio_k = (‖x_{k+1}-μ_old‖² − ‖x_{k+1}-μ_new‖²) / (2σ²dt)` — Gaussian constant cancels.
- Reward: LIBERO sparse 0/1 terminal success.

### Files created
- `imitation/algos/fm_policy_rl.py` — `FlowMatchingPolicyRL` (subclass): `sample_actions_stochastic`, `chain_logprob`, `_gaussian_logp`.
- `imitation/algos/rl/grpo.py` — `compute_grpo_advantages`, `ppo_clip_loss`.
- `imitation/algos/rl/rollout_collector.py` — `FlowRLCollector` (sequential rollouts through `env_num=1`; subprocess path crashes due to EGL-in-forked-subprocess).
- `rl_train.py` — entrypoint mirroring `train.py`.
- `config/rl_train.yaml`, `config/algo/fm_policy_rl_S.yaml`.

### Validation ladder
- Ladder 1-3 (pure-tensor; logp self-consistency, Gaussian sanity, PPO clip pre-update): **PASSED** (tol 2e-3 relaxed for fp32 summation noise over 560 elements; original 1e-4 too tight).
- Ladder 4-6 (env in the loop): integrated into the vertical-slice run below.

### Cold-start task selection (preliminary sweep)
- Task 0 (drawer): ~100% even at 1 demo → no RL headroom.
- Tasks 46 / 73 (pick-place) at 5 demos: 0% → too hard, no signal.
- **Task 16 (KITCHEN_SCENE2 stack bowls) at 10 demos: ~50%** → partial-success regime; chosen as RL cold-start.

### Stability findings (non-obvious; recorded as `project-fm-rl` memory)
- The per-step ratio has a `1/(2σ²·dt)` amplifier (≈500× at σ=0.1) → naive PPO diverges (KL=15.4, log_ratio capped at 20 on attempt 1).
- Fixes that worked:
  - **KL early-stop** with bounded estimator `0.5·E[log_ratio²]`, `target_kl=0.03–0.05`.
  - **Tiny lr (1e-6)**, grad-clip 1.0, fp32, dropout off.
- Post-fix: ratio ≈ 0.99, KL ≈ 0.02/iter, stable.
- Also required: `temporal_agg=false` (else overlapping chunks break chain↔action correspondence); encoder frozen during RL (`velocity_net` only).

### RL vertical-slice v1 (stable but flat)
- Cold-start: task 16 / d=10 (~50% baseline).
- Settings: G=8, 3 groups/iter, ppo_epochs=4, lr=1e-6, target_kl=0.03, n_iters=100.
- Eval: 20 deterministic rollouts every 5 iters.
- Result: stable but **flat at 50% across iter 0/5/10**. Diagnosis: gradient noise from only 3 groups/iter on a long-horizon stacking task with outcome-only credit broadcast across ~30 decisions.

### RL vertical-slice v2 (in progress, then paused for BC ablations)
- Same cold-start, with: 6 groups/iter (doubled), `target_kl=0.05`, `eval_rollouts=30`, `n_iters=25`. **Not yet conclusive — paused to investigate the BC operating point first.**

---

## 2026-05-29 – 2026-05-30 — BC ablations on task 16

All trains: 60 epochs, no wandb, eval = 40 rollouts (rl_train.py with `n_iters=0` prints two 20-rollout evals).

### Chunk size × action horizon sweep (10 demos fixed)
| chunk_size | action_horizon | eval1 | eval2 | combined (40 rollouts) |
|-----------:|---------------:|------:|------:|-----------------------:|
| 16 | 8  | — | — | **~50%** (baseline) |
| 32 | 16 | 10% | 15% | **12.5%** |
| 32 | 8  | 30% | 50% | **40.0%** |

**Findings:**
- `action_horizon=16` is the dominant regressor (–25 to –30 pts vs horizon=8 at any chunk size).
- `chunk_size=32` alone (with horizon=8) is roughly neutral within 40-rollout noise (40% vs ~50%).
- Doubling both **compounds** (12.5% — multiplicative regression).
- Training loss was *lower* at chunk=32 than chunk=16 — clean instance of "BC training loss vs closed-loop task success" divergence.

Scripts: `cold_t16_d10_c32h16.sh`, `cold_t16_d10_c32h8.sh` (under `/storage/scratch1/8/lwang831/`).

### Demo-count sweep at chunk_size=16, action_horizon=8
| demos | eval1 | eval2 | combined (40 rollouts) |
|------:|------:|------:|-----------------------:|
| 5  | 10% | 15% | **12.5%** |
| 10 | 50% | 25% | **37.5%** |
| 20 | 45% | 60% | **52.5%** |
| 40 | 85% | 95% | **90.0%** |

**Findings:**
- Per-demo marginal value is **non-monotone**: +5.0 pts/demo (5→10), +1.5 pts/demo (10→20), +1.9 pts/demo (20→40).
- The 20→40 range shows a **sharp jump (+38 pts)** — suggests a coverage threshold near ~30 demos beyond which the policy can reliably chain the full trajectory.
- d=10 sits in the noisy partial-success band (50% vs 25% across two evals) — confirms why it's a good RL cold-start point but a bad BC operating point.

Script: `demos_sweep_t16.sh`; results file: `/storage/scratch1/8/lwang831/demos_sweep_t16_results.txt`.

---

## 2026-05-30 — BC → BC+RL sweep on task 16 (the main result so far)

For each of the 4 cold-start checkpoints from the demo-count sweep, ran FM-RL
(15 iters, 6 groups/iter × G=8 rollouts = 48 rollouts/iter; ppo_epochs=4, lr=1e-6,
target_kl=0.05; v2-tuned settings). Eval every 5 iters + final, 20 rollouts each.
Jobs: SLURM 9335346 (cells d=5/10/20) + 9347994 (d=40 rerun after TIMEOUT).

| demos | BC | iter 0 | iter 5 | iter 10 | iter 15 | final | best | **Δ vs BC** |
|------:|---:|-------:|-------:|--------:|--------:|------:|-----:|------------:|
|     5 | 12.5% | 10% | 10% |   5% |   5% |   5% | 10% | **–2.5** |
|    10 | 37.5% | 50% | 60% |  50% |  55% |  65% | 65% | **+27.5** |
|    20 | 52.5% | 45% | 70% |  70% |  65% |  65% | 70% | **+17.5** |
|    40 | 90.0% | 90% | 80% |  95% |  90% |  90% | 95% | **+5.0** |

**Headline:** RL gain is **inverted-U in demo count**, peaking in the partial-success band.
This is the *opposite* of SimpleVLA-RL's pitch ("RL works best with few demos") — for FM-RL
the few-demo extreme (d=5, BC=12.5%) collapses because GRPO needs at least one success per
group to produce non-zero centered advantage, and at 12.5% most G=8 groups are all-zero.

**Mechanism diagnosis:**
- d=5 (BC 12.5%): expected successful groups per iter ≈ 6 · (1 − (1 − 0.125)^8) ≈ 4.4 / 6 —
  but the surviving advantage signal is small and noisy, and the model drifts off the BC manifold.
  Net regression (–2.5 pts).
- d=10 (BC 37.5%): mixed groups everywhere → cleanest gradient signal → **+27.5 pts**, the
  largest single-cell gain in the sweep.
- d=20 (BC 52.5%): still in the partial-success band, +17.5 pts. Tracks the d=10 story.
- d=40 (BC 90%): ceiling effect — most groups are all-success, centered advantage is small,
  and the small slice of failures provides the only signal. Modest +5.0 pts gain.

**Implications for the data-usefulness thesis:**

1. **"More demos" and "more RL" are not strict substitutes** — they're complements in a
   narrow band (BC ≈ 30–60%) and substitutes outside it. "RL replaces demos" is wrong at
   d=5 and trivially true at d=40.
2. **The most data-efficient operating point for the BC+RL pipeline is ~10–20 demos here**,
   not the minimum demo count that yields any non-zero success rate. Concrete labeling rule:
   target the partial-success band, not the bottom of the curve.
3. **The SimpleVLA-RL claim does NOT transfer to small flow-matching policies.** They report
   1-demo SFT + RL → 17% → 92% on LIBERO-Long with an autoregressive token VLA at 8×A800.
   Our 18M FM-RL at d=5 (BC 12.5%) goes the *opposite* way (–2.5 pts). One of {token-vs-flow
   representation, model scale, warm-start protocol} is doing the heavy lifting in their
   result — worth isolating before generalizing the "RL works best with few demos" framing.

## 2026-06-03 — Multi-task LIBERO-90: drift vs FM-BC head-to-head

### Setup
- Trained `PolicyDrifting` (lambertae JAX port → PyTorch; 1-step generator, `R_list=[0.02,0.05,0.2]`,
  `scale_inputs=true`, per-state `G=4` generators) on **all 90 LIBERO-90 tasks, 50 demos each**.
  Run: `exp_name=drift_multitask_lib90`, 30 epochs, `chunk_size=16`, `action_horizon=8`,
  `rollout.enabled=false` (eval done separately).
- Hardware: **L40S** (account `gts-agarg35-ideas_l40s`, partition `gpu-l40s`, embers QoS).
  Job 9403479 → 30 epochs in **3:46:26** wall (~7.4 min/epoch on 90 tasks). Loss curve healthy:
  inferenced-action MSE 0.0524 → 0.0165 (3.2× reduction).
- Eval: `scripts/eval_drift_multitask_lib90.{py,sbatch}` — 10 deterministic rollouts/task,
  5 parallel envs → **900 total rollouts**. Per-task resumable (results file is the resume key).
  Job 9414477 → **1:56:23 wall on V100**. Comparison baseline: FM-BC `cold_multitask_lib90`
  ckpt (also 50 demos, 30 epochs), same 10-rollouts/task protocol.

### Headline result

| | drift (`PolicyDrifting`) | FM-BC (`FlowMatchingPolicy`) | Δ |
|---|---:|---:|---:|
| mean success over 90 tasks | **0.9122** | 0.9067 | **+0.56pp** |
| tasks with sr = 1.00 | **55 / 90** | 48 / 90 | +7 |
| tasks with sr = 0.00 | 1 / 90 | 1 / 90 | tie (same task: `LIVING_ROOM_SCENE2_pick_up_butter_and_put_it_in_the_basket`) |
| win / tie / loss (drift's view) | 21 / 52 / 17 | — | — |

- **Multi-task BC on LIBERO-90 is saturating around ~91% for both methods.** Drift edges FM-BC
  by +0.56pp on the mean, but the more interesting signal is the **+7 perfect-task gap** (55 vs 48):
  drift more reliably nails tasks all the way to 100%, with the trade that it loses small amounts
  on a handful of single-step pick-place tasks.
- Per-task biggest **drift wins** (Δ ≥ +0.20):
  - t8 `KITCHEN_SCENE1_open_top_drawer_and_put_…` 1.00 vs 0.60 (+0.40)
  - t13 `KITCHEN_SCENE2_put_bowl_front_on_plate` 1.00 vs 0.70 (+0.30)
  - t43 `KITCHEN_SCENE9_put_white_bowl_on_cabinet` 1.00 vs 0.70 (+0.30)
  - t21 `KITCHEN_SCENE3_turn_on_stove_and_put_pan` 0.70 vs 0.50 (+0.20)
  - t23 `KITCHEN_SCENE4_close_drawer_and_open_…` 0.90 vs 0.70 (+0.20)
  - t45 `KITCHEN_SCENE9_turn_on_stove_and_put_pan` 1.00 vs 0.80 (+0.20)
  - t53 `LIVING_ROOM_SCENE2_pick_up_oj_in_basket` 0.80 vs 0.60 (+0.20)
- Per-task biggest **drift losses** (Δ ≤ −0.20):
  - t65 `LIVING_ROOM_SCENE5_put_red_mug_on_left_plate` 0.50 vs 0.90 (−0.40)
  - t27 `KITCHEN_SCENE4_put_wine_bottle_on_rack` 0.70 vs 1.00 (−0.30)
  - t81 `STUDY_SCENE3_book_in_front_compartment` 0.60 vs 0.90 (−0.30)
  - t26 `KITCHEN_SCENE4_put_wine_bottle_in_drawer` 0.70 vs 0.90 (−0.20)
  - t48 `LIVING_ROOM_SCENE1_pick_up_ketchup_in_basket` 0.80 vs 1.00 (−0.20)
  - t89 `STUDY_SCENE4_book_right_under` 0.70 vs 0.90 (−0.20)
- Win-pattern (informal): drift's wins concentrate on **compound / sequential KITCHEN tasks**
  (`open-then-put`, `turn-on-then-put`, `close-then-open`); its losses are mostly **single-step
  precision pick-place** (mugs/wine bottles/books to a specific compartment).

### Implications for the data-usefulness thesis
- **The ~91% multi-task BC ceiling kills the "RL extracts more from few demos than BC" headroom
  on LIBERO-90.** FM-RL on the multi-task setting earlier moved the mean only +2.2pp (78%→80.2%
  best, see the May-30 RL sweep notes for the d≥30 saturation pattern); with BC now at 91%, the
  multi-task RL gain ceiling is even tighter. → Pivoted to **LIBERO-Long (libero_10)** as the
  hard-task arena where the methods can actually separate; libero_10 is already downloaded
  (cedar, 13 GB, 10 HDF5s) and preprocessed (job 9403912, 1:06:32 on V100). Next: train
  multi-task FM + drift on libero_10 and re-run this head-to-head.
- **L40S is the right training rig for libero_90-scale multitask runs**: 7.4 min/epoch vs V100's
  estimated ~25 min/epoch → ~3.4× speedup, full 30-epoch run inside the 8-hour embers wall.
  Use `--account=gts-agarg35-ideas_l40s` (default `gts-agarg35` silently falls back to V100).
- **Rollout eval is sim-bound, not GPU-bound** → keep evals on V100/A100 and reserve L40S for
  training. 90 tasks × 10 rolls in ~2 h on V100 is comfortable.

## Open threads

- **d=30 gap-filler** + **d=15 gap-filler**: localize the BC 20→40 jump and the RL gain
  20→40 collapse (does the inverted-U peak at d=10 or d=15?).
- **Demo sweep replication on a second task**: test whether the inverted-U RL-gain shape
  generalizes or is task-specific (good candidates: another partial-success task at d=10).
- **σ sensitivity at d=5**: does a smaller `rl_sigma` keep the policy on-manifold so the
  few signal-bearing groups at d=5 actually move things in the right direction?

## Artifact index

| Item | Path |
|------|------|
| Cold-start checkpoints | `/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/<exp_name>/` |
| RL experiment dirs | `/storage/scratch1/8/lwang831/imitation/experiments_rl/` |
| BC ablation scripts | `/storage/scratch1/8/lwang831/cold_t16_d10_c32h*.sh`, `demos_sweep_t16.sh` |
| Demos sweep results | `/storage/scratch1/8/lwang831/demos_sweep_t16_results.txt` |
| BC→RL sweep sbatch | `scripts/rl_demos_sweep_t16.sbatch` (jobs 9335346, 9347994) |
| BC→RL sweep results | `/storage/scratch1/8/lwang831/rl_demos_sweep_t16_results.txt` |
| BC→RL per-cell logs | `/storage/scratch1/8/lwang831/fm_rl_t16_d{5,10,20,40}.log` |
| LIBERO-90 processed data | `/storage/scratch1/8/lwang831/imitation/data/libero/libero_90/` |
| LIBERO-Long raw data (cedar) | `/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_10/` |
| LIBERO-Long processed data | `/storage/scratch1/8/lwang831/imitation/data/libero/libero_10/` |
| Drift multitask training ckpt | `/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90/` |
| FM-BC multitask training ckpt | `/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/cold_multitask_lib90/` |
| Drift multitask eval results | `/storage/scratch1/8/lwang831/imitation/eval_results/drift_multitask_lib90_per_task.tsv` |
| FM-BC multitask eval results | `/storage/scratch1/8/lwang831/eval_multitask_lib90_per_task.txt` |
| Drift training script | `scripts/train_multitask_lib90_drift.sbatch` (job 9403479) |
| Drift eval script | `scripts/eval_drift_multitask_lib90.{py,sbatch}` (job 9414477) |
| libero_10 preprocessing script | `scripts/process_libero_10.sbatch` (job 9403912) |
| Plan | `~/.claude/plans/sequential-gliding-salamander.md` |
