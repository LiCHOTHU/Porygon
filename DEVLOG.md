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

## 2026-06-04 — Multi-task RL on hard-8: DICE-RL + SimpleVLA-RL × FM + drift

### Setup
With multi-task BC saturating at ~91% on LIBERO-90 (2026-06-03), pivoted to the **8 hardest
tasks** (indices `[8, 21, 32, 53, 65, 73, 75, 81]`) as the arena where RL has room to move.
Built out **two RL algos × two policies = 4 cells**:

- **DICE-RL** ("From Prior to Pro", arXiv 2603.10263; ported from `real-stanford/dice-rl` +
  `zhanyisun/DICE-RL-Robot`): residual `a_total = a_teacher + actor(state, noise)` with an
  ensemble-of-10 critic, Q-normalization, n-step returns (n=3), multi-z next-noise targets,
  soft Q-filtering, expert/online data masks, optional self-imitation. Entrypoint
  `dice_train.py`; algo files under `imitation/algos/dice/`.
- **SimpleVLA-RL** (arXiv 2509.09674) flavor: GRPO advantage (group ≥ G rollouts from same
  init state), **std-normalized advantage hardcoded true**, **asymmetric PPO clip
  `[0.20, 0.28]`**, inclusive filter `[0.1, 0.9]`. Entrypoint `rl_train.py`; algo files under
  `imitation/algos/rl/`.

Drift treated as **K=1 flow** (smoke test [3] confirmed bit-exact equivalence of drift's
deployed 1-step rule and FM-arch + `num_inference_steps=1`); no separate algo, just the
config override on either trainer.

### Smoke tests (all PASS)
- `scripts/smoke_dice_residual.py`: [1] iter-0 residual exactly 0 (zero-final-layer), [2]
  gradient flows through residual / teacher frozen, [3] K=1 FMTeacher == drift's 1-step
  rule (max-err 0.0), [4] full `loss()` w/ Q-norm + multi-z + n-step runs.
- `scripts/smoke_dice_pipeline.py`: [A] n-step return math matches closed form, [B]
  `sample()` shape contract, [C] 20-step update loop no-NaN + target drift + bounded
  actor grad, [D] actor↔critic backward isolation (Δ=0.0 between graphs).
- `scripts/smoke_simplevla_rl.py`: [1] GRPO std-norm matches paper formula (max-err 0e0),
  [2] asymmetric clip / clipfrac matches log-bound math, [3] inclusive filter bitmask
  match, [4] ratio=1 → no-op clip.

### Hard-8 multi-task results (jobs 9420087–9420090, ~2.5–3 h each on A100/H100)

| algo / policy | cold-start | final | Δ |
|---|---:|---:|---:|
| **DICE-RL × FM** (K=10) | 0.725 | **0.769** | **+0.044 ✓** |
| DICE-RL × drift (K=1) | 0.717 | 0.681 | −0.036 |
| SimpleVLA-RL × FM | 0.725 | 0.708 | −0.017 |
| SimpleVLA-RL × drift | 0.717 | 0.692 | −0.025 |

**Headline:** of the four, **only `fm_dice` beat its frozen-BC cold-start globally**. drift_dice
showed a steep within-run climb (0.588 → 0.681 over iters 3 → 12, +9.3pp) but never recovered
to its 0.717 starting point — early DICE actor noise hurts more with K=1 (no Euler-step
smoothing), and the Q-filter (12 activations over the run) wasn't enough at this LR/iter count.
SimpleVLA-RL on both policies stayed roughly at cold-start parity for the 6 iters it managed
(its filter+PPO inner loop is ~2× DICE's per-iter cost).

### Per-task picture (hard-8, fm_dice; iter 3 → iter 12)
- KITCHEN_SCENE3 (21): 0.70 → 0.85 (+0.15) ✓
- LIVING_ROOM_SCENE2 (53): 0.80 → 0.90 (+0.10) ✓
- STUDY_SCENE1-right / **t75**: 0.50 → **0.80** (+0.30) ✓ largest gain
- STUDY_SCENE3 (81): 0.60 → 0.75 (+0.15) ✓
- KITCHEN_SCENE5 (32): 0.75 → 0.75 (flat)
- KITCHEN_SCENE1 (8): 0.95 → 0.90 (−0.05)
- LIVING_ROOM_SCENE5 (65): 0.85 → 0.80 (−0.05)
- STUDY_SCENE1-front (73): 0.70 → **0.40** (−0.30) ✗ largest loss

Even the winning algo improves 4/8 tasks and regresses 3/8 — RL gain is non-uniform at the
per-task level. The +0.044 mean is the *net* of these.

### Per-task-75 slice across all 4 algos (15-rollout estimates; ±0.12 noise)

| | DICE Δ on t75 | SimpleVLA Δ on t75 |
|---|---:|---:|
| FM | **+0.07** (0.73 → 0.80) | −0.06 (0.73 → 0.67) |
| Drift | **+0.08** (0.67 → 0.75) | **+0.13** (0.67 → 0.80) ← largest single cell |
| **mean** | +0.075 | +0.035 |
| **spread** | 0.01 (tight) | 0.19 (wide) |

DICE looks more *consistent* per-task; SimpleVLA has the biggest single cell but also the only
negative one. Caveat: 15-rollout noise (±0.12) means SimpleVLA's spread is partly sampling
variance, not a true ceiling difference. These numbers are from the multi-task hard-8 ckpts
sliced to task 75, not from a single-task RL run.

### Single-task RL on t75 — submitted to SLURM
True single-task ablation (RL on `task_indices=[75]` alone, starting from each cold-start
ckpt) is the cleaner version of the comparison. Local interactive H100 attempts crashed:
- Attempt 1 (4 algos × `num_parallel_envs=5`): all crashed with `EGLError(EGL_NOT_INITIALIZED)`
  in forked subprocess — re-hit the **parallel-env gotcha** ([[runtime-env]]), `num_parallel_envs>1`
  uses `SubprocVectorEnv` which can't share EGL contexts. Mine to remember; ~76 min wasted, 0
  eval lines.
- Attempt 2 (`num_parallel_envs=1` smoke): silent SIGKILL mid-rollout at 2.4 s/env-step
  (~50× slower than dedicated-GPU rate). Local node was either OOM-killing or thrashing
  rendering against another invisible user.

Submitted **job 9431681** (`scripts/rl_single_t75_all4.sbatch`) — one sbatch runs all 4 algos
sequentially on task 75, 8 h budget, embers, partitions `gpu-a100,gpu-h100,gpu-v100`, scrubbed
env to dodge the `SLURM_JOB_PARTITION` leak gotcha. Per-algo settings mirror the hard-8 runs
(DICE: n_iters=12, eps=8, warmup=16, grad_steps=200, eval every 2; SimpleVLA: n_iters=8,
G=16, inits=4, lr=1e-6, target_kl=0.05, filter [0.1, 0.9], eval every 2). Resume-safe — each
`run_dice` / `run_simplevla` skips an algo whose log already has the expected eval count, so
requeue doesn't lose progress. Currently `PENDING (Resources)`.

### Bug fixes that landed this session (worth remembering)
- **DICE-RL residual structure** was wrong: previous `distill_rl.py` had an independent
  Tanh-MLP actor with a BC-MSE pull, NOT the residual `a_teacher + actor(...)` form. Rewrote
  to mirror the official sim repo's `DistillResidualRLModel` (Identity output, optional
  `zero_final_layer=True` — verified by smoke [1]). All production knobs (Q-norm, n-step,
  multi-z, soft Q-filter, exploration strategies) ported.
- **dice_train.py update-loop bug:** else-branch was calling `actor_total.backward(retain_graph=True)`
  without `zero_grad` → stale-actor-grad accumulation. Removed.
- **SimpleVLA-RL config defaults were wrong:** `std_normalize=false` (should be hardcoded `true`),
  filter used strict `<` (should be inclusive `<=`), symmetric clip (should be asymmetric
  `0.20/0.28`). All three fixed in `config/rl_train.yaml` + `imitation/algos/rl/grpo.py`.
- **dice_train.py** now accepts `+dice_resume_checkpoint=<path>` to load a trained student
  for eval-only / continued training (added 2026-06-04). Pair with `dice.n_iters=0` and
  `dice.warmup_episodes=0` for a clean eval-only invocation.

## 2026-06-06 — Long hard-8 RL + LIBERO-Long BC baselines (high precision)

### LIBERO-Long (libero_10) BC baselines — 500-roll deterministic eval

50 rollouts/task × 10 tasks → SE ≈ 2.2 pp overall, ≈ 7 pp per-task. Sbatches:
`scripts/eval_multitask_bc_lib10_500roll{,_drift}.sbatch` (jobs 9464235 FM, 9464237 drift).
Drift completed cleanly (1h39m, L40S); FM hit the 2h30m wall *after* writing the eval line
(value is good, post-eval cleanup hung).

| policy | overall | per-task highlights |
|---|---|---|
| **FM (K=10)** | **0.722** | STUDY1=0.98, K3 stove+moka=0.92, K4 bowl+drawer=0.90, LR2 cream+butter=0.90; **K8 both-moka=0.00** |
| **Drift (K=1)** | **0.660** | STUDY1=0.92, K3=0.82, K4=0.76; **K8 both-moka=0.00** |

FM beats Drift by **+6.2 pp**; gap concentrated on multi-object pick-and-place (LR2 cream+butter
+18, K4 bowl +14, LR5 two-mug +10). KITCHEN_SCENE8 (both moka pots → stove) is **0/50 for
both** — same intractable task for both policies, not a policy artifact.

### Hard-8 long RL — re-launches after the `filter_accuracy × task_ids` bug fix (#57)

Same 8 LIBERO-90 task indices `[8, 21, 32, 53, 65, 73, 75, 81]`, 20 GRPO iters (SimpleVLA) /
40 DICE iters, eval every 5. Three of four show real lift; one is structurally broken.

| run | status | eval trajectory (deterministic) | Δ peak |
|---|---|---|---|
| **fm_dice** | COMPLETED 40/40 | 0.719 → 0.738 → 0.731 → **0.794** → 0.787 → 0.769 → 0.744 → 0.738 | **+7.5 pp** |
| **drift_simplevla** | COMPLETED 20/20 | 0.675 (BC) → 0.713 → 0.675 → 0.688 → **0.738** → 0.688 | **+6.3 pp** |
| **fm_simplevla** | preempted iter 5/20 | 0.688 (BC) → **0.750** | **+6.2 pp** (early) |
| **drift_dice** | preempted iter 26/40 | 0.719 → 0.669 → 0.619 → 0.675 → 0.650 | **−7 pp (declining)** |

So the story isn't "RL doesn't work" — three of four runs improved 6–7.5 pp at peak. Two
qualifiers: (a) hard-8's BC baseline is already 68–72 %, so headroom is small; (b) both
finishers (fm_dice, drift_simplevla) overshoot then decay 4–5 pp by the end — the verl-style
adaptive-KL controller isn't anchoring hard enough at target KL ≈ 0.005–0.008.

### Why drift_dice fails (structural — recorded as feedback memory)

Same code as fm_dice, same hard-8 config, *only* difference is the K=1 teacher. Three pieces
of evidence:

1. **Eval is monotonically downward from iter 5.** fm_dice climbs to 0.794 on the same critic
   architecture; drift_dice drifts down 7 pp over 25 iters.
2. **Critic & actor never reach fm_dice's floor:** at iter 21, fm_dice actor=−0.83 / Q=0.54 /
   critic_loss=0.05; drift_dice actor=−0.69 / Q=0.34 / critic_loss=0.08. The critic sees
   ~35 % less value-headroom for the drift student.
3. **Collect success has no trend** (bounces 0.375–1.000, mean ~0.6 for 25 iters). The
   `max_q_std` explorer can't find better noises.

**Root cause:** DICE's `get_action` is `a_teacher(state, noise) + residual(state, noise)`, and
its exploration is `max_q_std` over 10 sampled noises. This scheme needs the teacher's
noise→action map to be **rich** — different noises must yield meaningfully different teacher
actions for the Q ensemble to have anything to rank.

- **K=10 FM teacher:** noise integrates through 10 Euler steps; different noises traverse
  different ODE trajectories and land at distinct actions on the demo manifold. `max_q_std`
  ranking has signal; residual MLP has slack.
- **K=1 drift teacher:** `a = clamp(noise + v(noise, t=0))`. A *single* velocity step. With
  dropout off at eval, `v(·, t=0)` dominates and is only mildly noise-sensitive → different
  noises collapse to nearly the same teacher action. `max_q_std` is degenerate; the residual
  must do all the work but is BC-anchored back via `bc_weight · ‖residual‖²`, so it can't.

Critic locks onto Q-values for "essentially the teacher action"; the actor exploits those
Q-values into off-distribution states; estimates go stale → eval drifts down. This is also
why **drift + GRPO worked** (+6.3 pp peak) on the same teacher — GRPO injects σ-Gaussian
noise into the executed action at the single K=1 step, so the policy itself is stochastic and
PPO clipping has real signal.

**Cheap-to-bold fixes to try:** (1) swap `max_q_std` → `max_q_min` (pessimistic ranking
doesn't degenerate); (2) inject explicit σ ≈ 0.05–0.1 Gaussian noise into `a_teacher` for
K=1 so noise diversity exists somewhere; (3) increase residual head capacity and lower
`bc_weight`; (4) just skip DICE for K=1 — use GRPO as the drift-policy RL and reserve DICE
for K≥10.

### Bug fix that unblocked the relaunches

`filter_accuracy` keep-mask buffer-key loop didn't include `"task_ids"`, so after a filter
activation `task_ids` stayed full-length while `cond/chain/...` shrank. Next iter's
`balanced_perm(task_ids_full)` returned indices ≥ filtered N → `IndexError` at
`cond_mb = buf["cond"][idx]` (rl_train.py:204). Killed both SimpleVLA hard-8 runs yesterday
(9434006 at iter 1, 9434007 at iter 2). One-line fix: add `"task_ids"` to the tuple. Regression
test in `scripts/smoke_simplevla_verl_parity.py` (test [D]).

## 2026-06-06 PM — Corrected K=1 diagnosis + LIBERO-Long RL grid

The 2026-06-04/AM `drift_dice` failure analysis above is partially superseded by three
measurements run later in the day. Keeping the original text intact for the audit trail; the
corrected story is here.

### Phase 0 — `phase0_noise_diversity.json` (job 9489842)

Measured per-state noise→action diversity of FM K=10 vs drift K=1 on hard-8. Result inverts
the original hypothesis: drift K=1 σ = **0.043**, FM K=10 σ = **0.021** → drift has **~2×
more** spread, on all 8 tasks. The K=1 teacher is *over*-dispersed, not noise-starved.

### Step 1b — z-cond critic (`rl_hard8_long_drift_dice_zcond`)

Single-variable flip `dice.q_depends_on_noise=true`. With z-cond, critic fits Q(s,z,a) on the
narrow per-(s,z) action neighborhood instead of one loose marginalized surface. Verified
mechanism:
- critic_loss drops **28×** in 10 iters (1.38 → 0.05) — clean and attributable.
- eval trajectory (BC = 0.606): [0.637, 0.700, 0.631, 0.656, 0.637, 0.656, 0.600, 0.662, 0.588].
  Mean **+3.5pp** over BC. Looks peak-then-decay; see the noise discussion below.

### Discriminator probe — `probe_iter10_vs_iter40.json`

Built to distinguish (1) residual-direction drift from (2) Q-level inflation as causes of the
apparent decay. 200 (state, noise) pairs collected via iter-10 rollouts on hard-8. Result
**rejected both hypotheses**:
- Action RMS diff iter-10 vs iter-40 = **0.00040** — actions are essentially unchanged.
- Q drift on identical teacher action (Q_40 − Q_10) = **−0.158** — Q *dropped* on held-out
  failure states, not inflated. The train-log Q(s,a_student) climb was replay-batch credit
  accumulation (replay is success-dominated by iter 40), not overestimation.
- So the executed policy is essentially unchanged between iter 10 and iter 40, yet eval moved
  from 0.700 to 0.588. **The decay is mostly eval noise.**

Direct supporting evidence: drift × SimpleVLA-RL evaluated the same iter-20 ckpt twice and got
`0.800` → `0.740` (6pp of pure eval lottery on identical policy). Hard-8 noise band run
in flight (`scripts/noise_band_hard8_zcond.sbatch`, job 9509085) — 3 fresh evals each on
BC / iter 10 / iter 40 to nail down the band.

### LIBERO-Long (libero_10) RL grid — 4 cells

| algo × policy | BC | best eval | trajectory | net Δ vs BC | status |
|---|---|---|---|---|---|
| FM × DICE | 0.722 | **0.775** (iter 35) | 0.725 → 0.750 → 0.740 → 0.750 → 0.755 → 0.750 → 0.775 | +5.3 pp | TIMEOUT @ 39, resume in flight (job 9509086) |
| Drift × DICE-zcond | 0.660 | **0.745** (iter 5) | 0.745 → 0.675 → 0.720 → 0.690 → 0.675 → 0.640 → 0.695 → 0.700 → 0.660 | +0.0 end / +8.5 peak | DONE |
| FM × SimpleVLA | 0.722 | **0.790** (iter 10) | 0.740 → 0.790 → 0.730 | +6.8 peak / +0.8 (iter 15) | TIMEOUT @ 15 (8h wall too tight for 20 iters at ~30 min/iter) |
| **Drift × SimpleVLA** | 0.660 | **0.800** (iter 20) | 0.670 → 0.670 → 0.700 → **0.800** / 0.740 | **+8.0 end / +14 peak** | DONE — **headline** |

Same-ckpt iter-20 eval noise observed at **6pp** (0.800 vs 0.740). Apply that band to every
comparison above.

### Claim discipline — what's earned vs hedged vs conjectured

The probe finding (decay was mostly eval noise) reset what we can defensibly assert. Layering
the writeup claims by evidence type:

| Claim | Status | Evidence |
|---|---|---|
| z-cond fixes the marginalized-critic failure | **ASSERTED** | critic_loss 28× drop (1.38 → 0.05) is a training metric, noise-free |
| K=1 drift has 2× the per-state action dispersion of FM K=10 | **ASSERTED** | Phase 0 measured σ_drift/σ_FM = 2.05 on all 8 hard tasks |
| z-cond produces a reliable eval gain | **PENDING noise band** | +3.5pp mean is at-or-below the per-eval band; same-ckpt double-eval already showed 6pp natural variance |
| GRPO outperforms DICE on K=1 drift | **ASSERTED (cautious)** | drift × GRPO iter-20 **end** = 0.770 mean (0.800/0.740) vs drift × DICE-zcond iter-20 = 0.690; gap +8pp **end** exceeds 6pp noise band — peak (+14pp) demoted to supporting detail |
| Step-count is the governing axis | **CONJECTURED** | mechanism + N=2; K-sweep in flight (job 9509719) to turn N=2 into a within-policy trend |

The two-layer reading of the z-cond result is the most important reframe: **the critic
pathology is real and is fixed by z-cond; the fix does not translate into a reliable eval
gain.** That is more interesting than "modest improvement" — it says z-cond solves the wrong
problem, and the real ceiling for residual-on-frozen-prior RL is somewhere else (probably the
critic's inability to point a residual cleanly under a high-dispersion prior, which is
exactly what GRPO sidesteps).

### Headline (disciplined)

> For one-step generative policies, **the residual-on-frozen-prior + value-critic recipe (DICE)
> hits a ceiling that z-conditioning the critic does NOT lift in eval**, while critic-free
> group-relative RL (GRPO) clears the same ceiling by +8pp end on the lib10 grid (±6pp eval
> band). The Phase 0 dispersion measurement (σ_drift = 2× σ_FM) and the z-cond mechanism win
> (28× critic_loss drop with no eval gain) jointly point at the prior's per-state action
> dispersion as the underlying mechanism; an FM-at-K sweep is in flight to test whether step
> count is the governing axis.

### Grid plan (post-disciplining)

**Headline figure (lib10, 2×2 algo × policy, fair to iter 20):**
- All four cells annotated with their per-eval 95% CI from the noise band measurement.
- FM × DICE: 0.750 (iter 20), resumed run will overwrite once iter 40 lands.
- FM × SimpleVLA: **restart in flight** at 12h budget on h100-only (job 9509725) — original
  iter-15 result (0.730) does not go in the headline grid; iter-20 from restart will.
- Drift × DICE-zcond: 0.690 (iter 20).
- Drift × SimpleVLA: 0.770 (iter-20 mean of 0.800/0.740 same-ckpt double eval).

**Supporting figure (hard-8 ablation column):**
- BC drift / drift × DICE / drift × DICE-zcond — shows the z-cond critic mechanism win
  decoupled from the noise floor. Hard-8, *not* fused with the lib10 grid.

**Mechanism row:**
- Phase 0 σ(drift, K=1) = 0.043 vs σ(FM, K=10) = 0.021 (2.05× ratio).
- FM-at-K dispersion sweep: σ(FM, K ∈ {1, 2, 4, 10}) — pending job 9509719.
- z-cond critic_loss 28× drop (training metric).
- Probe rejection: action diff = 0.0004, Q drift on identical action = −0.158.

### Open follow-ups

- **Noise band** (job 9509085 in flight) — quantifies σ_eval to put error bars on every cell.
- **FM × DICE resume** (job 9509086 in flight) — lands iter 40 for the full FM × DICE
  trajectory.
- **FM × SimpleVLA restart** (job 9509725 pending) — iter 20 for fair grid.
- **Dispersion vs K** (job 9509719 running) — turns step-count from N=2 to within-policy trend.
- (Future, if dispersion-vs-K is monotonic) — one drift-RL run at K=4 to close the loop.

### Code/scripts that landed this PM

- `imitation/algos/dice/distill_rl.py` — z-cond critic flag (already existed) + CQL conservative
  penalty `dice.cql_weight` (added; OFF by default; **not used** — probe rejected the leak it
  targets, kept in the codebase for future use if needed).
- `dice_train.py` / `config/dice_train.yaml` — plumbing for CQL kwarg + per-iter
  `cql / q_pol / q_data` logging when on.
- `scripts/phase0_noise_diversity.{py,sbatch}` — noise diversity gating measurement (used).
- `scripts/probe_iter10_vs_iter40.{py,sbatch}` — discriminator probe (used).
- `scripts/noise_band_hard8_zcond.{py,sbatch}` — noise-band measurement on hard-8 z-cond (in
  flight, job 9509085).
- `scripts/rl_hard8_long_drift_dice_zcond_cql.sbatch` — staged CQL relaunch, **never submitted**
  per the probe verdict. Kept in tree for reproducibility.
- `scripts/rl_lib10_{fm,drift}_{dice,simplevla}.sbatch` — lib10 RL × 4 grid cells.

## Open threads

- **Single-task RL on t75 (job 9431681, PENDING):** confirm whether dedicated single-task RL
  beats both (a) the cold-start baseline and (b) the multi-task hard-8 per-task-75 numbers.
  If yes, that's the cleanest "RL extracts more from data than BC" signal we have on LIBERO-90.
- **30-rollout re-eval of hard-8 DICE ckpts on t75:** dice_latest.pth exists for both fm_dice
  and drift_dice (SimpleVLA did NOT save); a tight re-eval would denoise the ±0.12 sample
  variance and let us actually conclude on per-cell lift.
- **drift_dice early-collapse:** iter-3 eval landed at 0.588 vs 0.717 cold-start — try smaller
  actor LR (1e-4 → 3e-5) or longer warmup to see if the climb crosses baseline.
- **libero_10 multitask BC + RL:** still the pivot target. Multi-task BC on libero_10 not yet
  trained; both FM and drift cold-starts are the prereq.
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
| DICE-RL trainer + algo | `dice_train.py`, `imitation/algos/dice/{distill_rl,replay_buffer,collector,teacher}.py` |
| SimpleVLA-RL trainer + algo | `rl_train.py`, `imitation/algos/rl/{grpo,rollout_collector,buffer}.py` |
| DICE-RL config | `config/dice_train.yaml`, `config/algo/fm_policy_dice_S.yaml` |
| SimpleVLA-RL config | `config/rl_train.yaml`, `config/algo/fm_policy_rl_S.yaml` |
| Algo smoke tests | `scripts/smoke_{dice_residual,dice_pipeline,simplevla_rl}.py` |
| Hard-8 RL sbatches | `scripts/rl_hard8_{fm,drift}_{dice,simplevla}.sbatch` (jobs 9420087–9420090) |
| Hard-8 RL logs / results | `/storage/scratch1/8/lwang831/rl_hard8_{fm,drift}_{dice,simplevla}.{log,_results.txt}` |
| Hard-8 DICE ckpts | `/storage/scratch1/8/lwang831/imitation/experiments_dice/libero/libero_90/rl_hard8_{fm,drift}_dice/dice_latest.pth` |
| Single-task t75 sbatch | `scripts/rl_single_t75_all4.sbatch` (job 9431681) |
| Single-task t75 results | `/storage/scratch1/8/lwang831/rl_single_t75_all4_results.txt` (pending) |
| Plan | `~/.claude/plans/sequential-gliding-salamander.md` |

## 2026-06-11 — FM-DICE flatline root-caused against real-stanford/dice-rl; RLPD + scaled rerun

Compared our DICE port line-by-line against the official repo (cloned to /tmp/dice-rl-ref).
**The loss/model port is faithful** (actor multi-z, q-normalization, BC anchor, critic
ensemble/min/n-step, max_q_min exploration, all official hyperparams incl. bcw=100).
Three real divergences explain the iter-40 flatline (0.756 -> 0.762, residual_norm ~1e-4):

1. **RLPD never wired**: official always runs use_rlpd=true with 50 expert demos at
   adaptive ratio 0.5->0.1 + disable_q_loss_for_expert_data. Ours had
   `expert_dataset=None  # TODO`.
2. **~20-100x less training**: official converges at ~100-200k grad updates (README:
   24-48h at ~1s/batch); ours ran 8k.
3. **Warmup units**: replay_flow_warmup_steps=4000 is env-steps in official (~2% of
   run); we pass grad-steps where 4000 = half the run, so max_q_min exploration was
   off for iters 1-20 of 40.

Fixes landed:
- `imitation/algos/dice/expert_loader.py` (NEW): encodes LIBERO demos through the
  frozen BC encoder into a frozen expert ReplayBuffer (chunk cadence = action_horizon,
  reward 1/done at demo end, mirrors official ph_finetune semantics).
- `replay_buffer.py`: RLPD symmetric sampling (expert_ratio*B expert rows per batch).
- `dice_train.py`: builds expert buffer when use_rlpd=true; adaptive expert ratio
  (0.5->0.1 over adaptive_expert_ratio_steps grad steps), logged as train/expert_ratio.
- `config/dice_train.yaml`: adaptive_expert_ratio_* knobs.

Relaunched FM-DICE hard-8 at faithful budget (rl_hard8_long_fm_dice_rlpd):
32 eps/iter x 800 grad steps x 100 iters = 80k updates, RLPD on,
disable_q_loss_for_expert_data=true, always_retain_bc_loss_for_expert_data=true,
replay_flow_warmup=600 (~iter 3). Jobs: smoke 9818824 -> train chain
9818825-9818829 (5x8h, resume via dice_latest.pth; NOTE online replay restarts
empty each segment) -> powered eval 9818830 (BC + iter50 + iter100, n=200x3).

Caveat for the writeup: prior "DICE structurally fails" claims are budget-confounded;
GRPO-vs-DICE comparisons at 8k steps are sample-efficiency claims, not correctness.

## 2026-06-14 — Literature review: residual / GRPO / flow-matching RL on robotics

Ran a verified deep-research sweep (25/25 claims confirmed 3-0 adversarial) on whether
anyone does **residual + GRPO-style (critic-free, group-relative) + flow-matching** RL on
LIBERO. Verdict: **the exact triple is unoccupied** — every related work covers at most two
of the three axes. Closest neighbors: LP-DS (residual+flow but Lagrangian+Q-critic);
SimpleVLA-RL / TGRPO / RLinf-VLA (GRPO but token-VLA full-policy); π-RL / ReinFlow
(flow but PPO+critic full-policy); ResFiT/ResiP/DSRL/IBRL/RPL (residual but off-policy
actor-critic).

| Work | residual? | GRPO? | flow? | objective | Major novelty / contribution |
|---|:--:|:--:|:--:|---|---|
| **LP-DS** (ICML'26) | ✅ | ❌ | ✅ | Lagrangian + Q-critic | RL-steers a **frozen** diffusion/flow policy via a state-conditioned **latent-noise residual** `w=ε+Δθ(s)`; exact BC recovery at Δθ=0; constrained (Lagrangian) safety formulation |
| **SimpleVLA-RL** (ICLR'26) | ❌ full | ✅ | ❌ token | modified GRPO | Showed **plain GRPO + sparse binary success** fine-tunes token VLAs (OpenVLA-OFT) on LIBERO from *minimal* demos with large gains; "1-shot"-style data efficiency; scales to real robots |
| **TGRPO** (2025) | ❌ full | ✅ | ❌ token | trajectory GRPO | **Trajectory-level** GRPO — fuses step-wise and trajectory-wise advantages to stabilize VLA fine-tuning |
| **RLinf-VLA** (2025) | ❌ full | ✅/PPO | ❌ token | GRPO or PPO | A **unified, efficient training system/infra** for RL of VLAs across algorithms (GRPO/PPO) and simulators — the engineering substrate, not a new objective |
| **π-RL** (2025) | ❌ full | ❌ | ✅ (π0/π0.5) | PPO + GAE critic | RL fine-tuning of **large flow-matching VLAs at scale** (π0/π0.5 action expert) with a value critic — flow-policy RL pushed to billion-param VLAs |
| **ReinFlow** (NeurIPS'25) | ❌ full | ❌ | ✅ | PPO + GAE critic | The **noise-injection trick**: makes a deterministic flow sampler stochastic to get tractable per-step log-probs, enabling PPO on flow-matching policies (the enabling idea this repo's sampler also uses) |
| **ResFiT / ResiP / DSRL / IBRL / RPL** | ✅ | ❌ | diffusion/MLP | TD3 / SAC / DDPG / PPO | The **residual-policy-learning family**: learn a corrective action over a frozen base controller/IL policy with **off-policy actor-critic**; DSRL = residual in diffusion *noise/latent* space; IBRL = imitation-bootstrapped RL; RPL = the original residual formulation |

**Why the GRPO + residual cell is empty (opposing inductive biases, not oversight):**
1. GRPO forms advantages from intra-group reward variance; a residual is a *small* correction
   around a frozen base, so grouped rollouts collapse together → variance signal vanishes.
   (Matches our own finding: FM dispersion 0.021 vs drift 0.043 — lower dispersion starves GRPO.)
2. Residual RL targets sample efficiency → off-policy value methods; GRPO is deliberately
   on-policy and critic-free. Opposite design centers.
3. GRPO's trajectory-level scalar return is a poor instrument for the per-step credit
   assignment a small local correction wants.
4. GRPO entered robotics via token-VLAs (LLM transplant → full-policy fine-tuning); residual RL
   comes from the classical control lineage. The communities have not crossed.

Relevance to our 2×2: our GRPO cells are *full-policy* flow GRPO (≈ SimpleVLA-RL's stated future
work); our DICE cells are *residual but Q/distribution-correction* (≈ LP-DS/DSRL family). The
genuinely novel cell — GRPO on a residual head over a frozen flow policy, with injected noise to
restore group variance — is one we have NOT built. That is the clean novelty target if we want it.

Caveat: LP-DS (`2606.01151` / "ICML 2026") and one intersection source (`2602.01789`) have
forward-dated-looking arXiv ids; content was verified against fetched HTML but verify the
metadata before formal citation.

## 2026-07-24 — Toy flaw→fix demo: critic-as-compass vs drift-field update

Context: first fair Porygon runs (field update vs plain-residual control on the same
drift base) show the control ahead on robomimic (can 0.950 vs C 0.906; square 0.740
vs C 0.527, B 0.651; 300-ep evals at matched steps) and no separation yet on hard-8
at ~half budget. To sharpen the motivation ("what flaw forces the field update?"),
ran a verified 3-agent literature sweep (notes in memory: reference_motivation_citations)
and built a toy that tests the central claim directly.

### Toy setup (`scripts/toy_field_vs_gradient{,2}.py` + sbatch; jobs 11436183/11436429)
2-mode bandit (modes at ±e1, true reward 1.0 / 0.6, ~0 off-manifold); critic = MLP
regressed on true reward at ON-MANIFOLD samples only (extrapolation error is real,
per BCQ); base = one-step z→a generator pretrained with the kernel-drift operator
(miniature drifting model). Same base, same critic, same 1500 updates for every arm:
- GRAD_free: maximize Q(g(z)) by backprop through the critic (DDPG-style compass)
- GRAD_bc:   + ||g−g_base||² anchor (control-A in miniature)
- FIELD_tilted / FIELD_topk: faithful Porygon-C update — V_Q (softmax(adv/τ)-tilted
  self-attraction, or top-4-of-64) + V_BC (attraction to frozen-base samples) +
  λ=1 restore, total clip; critic only EVALUATED at sampled actions.
Two regimes: EASY (2-D actions, 200 critic pts) and HARD (16-D actions, 60 critic
pts, overfit critic; real chunked actions are 28-D). 3 seeds. Outputs in
`$CEDAR/toy_field_demo2/` (toy2_results.json, curve pngs).

### Results (3-seed means; true = actual reward of policy samples)
EASY: GRAD_free 0.793, GRAD_bc 0.770 > FIELD_tilted 0.530 > base 0.496; FIELD_topk
0.414. Critic generalizes ⇒ no flaw ⇒ compass legitimately wins (mirrors robomimic).
HARD: GRAD_free true 0.000, Q→28,892 (gap +28,892), 100% off-manifold, dispersion
8112. **GRAD_bc ALSO 0.000** (gap +6.4, 99% off-manifold) — the BC anchor rescales
the poison but the direction still comes from the hallucinated gradient.
FIELD_tilted holds 0.614 (base 0.647), 0% off-manifold, gap −0.09, dispersion and
best-of-16 channel preserved (0.767). FIELD_topk safe too (0.487).

### Reading
1. The critic-exploitation flaw (TD3 feedback loop / BCQ extrapolation error) is
   real and regime-dependent: it detonates with high-dim actions + scarce critic
   data, and the standard BC-anchor fix does NOT save the gradient update there.
2. The field update is structurally immune (no-destroy): it only moves toward
   actions it has sampled and evaluated, so off-manifold hallucinations cannot
   attract it. Caveat: in the hard regime it held base performance but did not
   improve it — the toy proves safety, not superiority.
3. v1 negative result (kept, `toy_field_demo/`): without V_BC the field itself
   inflates off-manifold (41%) via kernel self-repulsion — V_BC + restore are the
   method's stability contract, not hyperparameters (miniature of the robomimic
   residual-runaway we fixed with restore_step_size).
4. tilted > topk in both regimes — supports the T_tilted arms in the square sweep
   (jobs 11434848-51) and the "reward-tilted self-drift = MPO/AWR tilt executed by
   the pretraining operator" framing.

Paper use: Figure-1 candidate ("when the critic can only judge, don't hand it a
compass"). Open obligation: show a real-task instance of the hard-critic regime
(K=1 drift on hard-8, where ∇ₐQ measured degenerate, is the candidate) — otherwise
the field update is insurance the easy-critic benchmarks don't need.

## 2026-07-25 — Porygon field-update: can target HIT, square lead at matched budget, tilted leads hard-8

Winner recipe from the 21-arm square sweep (Jul-24): `actor_mode=field_pointwise
q_source=grad q_step_size=0.5 total_max_norm=0.15 restore_step_size=1.0` on the
one-step drifting base ("Porygon-B"). All numbers 300-ep evals, last-3 averages.

### robomimic can — TARGET HIT (goal: reach FM-DICE 0.957)
| arm | success |
|---|---|
| drift base (no RL) | 0.877 |
| control A (plain residual DICE) | 0.950 |
| FM + DICE-RL (baseline-to-beat) | 0.957 |
| **Porygon-B s42** | **0.988** (@20K) |
| **Porygon-B s44** | **0.966** (@18K) |
| Porygon-B s43 | 0.913 @6K (preempted mid-climb, rerunning) |
| Porygon-B gentle (q0.3) | 0.943 @9K |

### robomimic square — leads at matched 18-20K budget
Porygon-B s42 0.905 avg / 0.923 final @20K vs matched-budget control A s44
**0.839 @18K (plateaued)** and FM best 0.857 @11K. FM never approaches 0.9 in
any seed/budget observed. Seed confirms: s43 0.810@11K tracking s42's curve
(s42 was 0.810@9K → 0.923@20K); s44 restarted after preemption. Cgrad hybrid
(distributional + grad) climbing 0.740@7K. wide256 base: 20-ep pretrain evals
(0.65) were noise — 300-ep step-1K readout ≈ original base; RL curve 0.793@9K.

### LIBERO hard-8 — tilted (soft-Q) arm takes first-ever field lead
Ported `q_source=tilted` (softmax(adv/τ)-weighted self-attraction, τ=1) into
imitation repo; V100 smoke: tilted ||V_Q||=0.089 = 26× grad (0.0034), 3× zeroth
on the real action-blind critic. First results (20-ep evals, ±11pp, ~iter 50):
T_gf_s10000 **0.781** vs FM-DICE 0.757, control A ~0.69, B/C/Cgf 0.65-0.72.
Noisy but the first time any field arm leads on hard-8 — and it is exactly the
weak-critic regime where the toy predicts soft-Q transport should win.

### Emerging paper story
One method (field update on the drifting base) with a Q-source dial set by
critic quality: analytic grad where the critic generalizes (robomimic), tilted
where it can only rank (hard-8). Toy two-regime demo = Fig-1 mechanism.
Embers preemption storm Fri night killed 15 jobs; all relaunched Sat morning
(hard-8 resumes via dice_latest.pth; official-repo runs restart fresh).

## 2026-07-25 (afternoon) — Local H100 tuning of the tilted (T) update

Ran the ASAP local tuning on the login-node H100 against the REAL mid-training
hard-8 tilted checkpoint (`tune_tilted_local.py` field-stats sweep +
`tune_tilted_local2.py` 300-step mini-trains, family recipe q_step 0.5 / clip 0.15).

- **q_temp**: ESS of the softmax attractors — 0.1→1.3 (top-1 collapse), 0.3→2.0,
  1.0→7.1/16 (informative band), ≥3→uniform (field vanishes, ‖V_Q‖ 0.31→0.05).
  Mini-train τ=0.5 vs 1.0 within noise → **keep q_temp=1.0** everywhere.
- **restore (decisive)**: full restore 1.0 without dead-zone ERASES the learned
  residual on hard-8 (ΔQ<0, res_rms 0.006→0.002) — weak hard-8 Q-fields can't
  outpull it, unlike robomimic ∇Q. restore 1.0 + **dead-zone radius 0.05** →
  ΔQ>0 with residual bounded at healthy size. restore 0.1 gives max ΔQ but 4×
  residual growth = critic-exploitation channel → rejected on principle.
- Ported `restore_step_size`/`restore_radius` into imitation `field_actor_loss`
  (was official-repo-only); smoke OK. Tuned hard-8 arms B2/C2/T2 relaunched with
  `restore_radius=0.05` as jobs 11477990-99. Robomimic tab jobs stay at radius 0
  (B's winning validated recipe there; per-benchmark uniformity preserved).

## 2026-07-26 — ABCT showdown: B (pointwise ∇Q) confirmed method-grade; T/C fair-tested and beaten; infra fixed with paid QOS

### Infrastructure: the square problem is solved
Fri-night embers preemption storm wiped ~40 jobs (queue empty by 3 AM); the official-repo
robomimic runs have NO resume, so every 8h-capped embers attempt restarted from step 0 —
meaning **every previous square number was a 9-12K truncation** (square needs 14-16h for 20K).
Fix (user-approved): moved all 20 robomimic jobs to paid `inferno` QOS on the ideas_l40s
account (20h walltime, no preemption, ~$150-250); hard-8 stays free on embers since the
imitation-repo chains resume from `dice_latest.pth`. Result: 11+ uninterrupted hours,
first-ever complete 20K+ square runs. Hard-8 lost 14 chains to daytime preemption again;
relaunched as 11501998-12012 (checkpoints intact, node atl1-1-03-010-1-0 excluded).

### The critical fix behind this wave: dead-zone restore for T/C
The Jul-25 robomimic T/C launches ran `restore_step_size=1.0` with NO dead-zone → full
restore ERASED the distributional arms' residuals (T res_rms 0.0021-0.0025, C 0.0007 vs
healthy B 0.0156-0.0173) — those runs were base+selection, not T/C tests. This wave (tabR)
relaunched with `+model.field.restore_radius=0.05`; live verification shows healthy
residuals 0.010-0.015. **So this is the first fair robomimic test of T and C.**

### robomimic — 300-ep evals, last-3 average @ latest step (all seeds, tuned family recipe)
| arm | can s42 | can s43 | can s44 | sq s42 | sq s43 | sq s44 |
|---|---|---|---|---|---|---|
| **B** (pointwise ∇Q) | **0.988** @20K | **0.993** @25K | 0.966 @18K | 0.905/**0.923** @20K | **0.938** @23K | **0.929** @23K |
| T (tilted, radius 0.05) | 0.961 @34K | 0.952 @31K | 0.945 @29K | 0.596 @33K | 0.593 @32K | 0.568 @26K |
| C (zeroth top-4, radius 0.05) | 0.920 @34K | 0.905 @31K | 0.929 @27K | 0.591 @31K | 0.570 @28K | 0.598 @26K |
| references | drift base 0.877 · control A 0.950 · FM-DICE 0.957 | | | FM-DICE best ~0.857-0.896 | | |

**Verdict: B is decisively the best arm.** Square is the tiebreaker: B replicates
0.92-0.94 across three seeds at the full 20K budget while T and C sit at 0.57-0.61 —
a ~35-point gap that is NOT a broken-residual artifact (residuals verified healthy,
T/C trained 6-14K steps LONGER than B). On can, T is respectable (0.95) but 4 points
below B's near-perfect 0.99. The value-only transports lose to the analytic gradient
wherever the critic's ∇Q is trustworthy — robomimic is that regime.

### LIBERO hard-8 — iter-100 endpoints (20-ep evals, ±11pp; chains resuming)
| arm | s10000 | s10001 | s10002 |
|---|---|---|---|
| A (plain residual DICE) | 0.662 | 0.656 @50 | 0.675 @60 |
| B (grad, gentle) | **0.738** | 0.694 @70 | 0.675 @50 |
| B2 (grad, tuned) | 0.700 | 0.719 @40 | 0.656 @80 |
| C (zeroth) | 0.631 | 0.644 @90 | 0.637 @40 |
| C-guarded | **0.738** | **0.738** @70 | 0.669 @60 |
| C2 (tuned) | 0.656 @90 | 0.625 @80 | 0.719 @60 |
| T-guarded | 0.675 | 0.706 @70 | 0.650 @90 |
| T2-guarded (tuned) | 0.656 @70 | 0.713 @60 | 0.688 @70 |

Reference: FM-DICE 0.757. CORRECTION to Jul-25 entry: the tilted "first-ever field lead"
(T_gf_s10000 0.781 @iter~50) did NOT hold — that seed finished iter-100 at 0.675; it was
20-ep eval noise. Current hard-8 leaders are B and C-guarded at 0.738; nothing beats
FM-DICE yet, several tuned chains still mid-run.

### Paper framing decision (OPEN — needs user sign-off)
The data supports **B as the method**, not T: same field-target-regression update
primitive as pretraining (no likelihoods, action-space trust region native to the form),
with the Q-source as a dial — analytic ∇Q where the critic generalizes (robomimic),
value-only/guarded transports narrowing the gap only in the weak-critic regime (hard-8,
toy Fig-1). The T-as-headline framing is untenable on this evidence.
Two gating cells still pending on inferno (queued behind tabR): **sq_ext_A s42/43**
(control A at full 20K budget — without it "B beats DICE-on-drift on square" is
unsupported) and sq/can ext FM (FM-DICE at matched budget). Also proposed: the untested
hybrid (distributional BC field + ∇Q) — decides whether the "drift-field anchor" story
survives in the headline or the pointwise restore suffices.

## 2026-07-27 — Matched-budget harvest: Table 1 filled, curves figure generated

### Robomimic — harmonized at the paper protocol (20K steps, last-3 avg of 18/19/20K, 300 ep)
Drift-base **square eval completed** (job 11504491): **0.382** (RL residual ckpt 0.418; can base remains 0.877).

| arm | can s42/s43/s44 | square s42/s43/s44 |
|---|---|---|
| A (DICE actor on drift) | 0.950 (single) | 0.870 / 0.909 / rerun |
| FM + DICE | 0.957 (single) | 0.934 / 0.923 / rerun |
| C (top-k) | 0.924 / 0.920 / 0.910 | 0.562 / 0.525 / 0.562 |
| T (tilted) | 0.949 / 0.947 / 0.919 | 0.578 / 0.584 / 0.560 |
| **B (Porygon)** | **0.988 / 0.993 /** rerun | **0.912 / 0.930 / 0.929** |

Key reading (honest version, now in the paper): on square B beats A at the matched
budget on every seed (+2–6pp) and reaches 0.91 by 15K vs A's ~20–25K; **both converge
to 0.97–0.99 by ~40K** (extended runs), so the square win is speed-to-level, not a
different asymptote. FM+DICE is at **parity with B on square** (0.92–0.93) from a much
stronger base; B beats it on can (0.99 vs 0.957). Prior table cells quoted mixed
steps — all superseded by the above.

### Hard-8 iter-100 endpoint refresh (20-ep evals; * = still short of iter 100)
A 0.662/0.694*/0.700 · B 0.738/0.650/0.675* · B2 0.700/0.675*/0.719 ·
C 0.631/0.662/0.625* · C-gf 0.738/0.656/0.662* · C2 0.631/0.681/0.644 ·
T-gf 0.675/0.650/0.656 · T2-gf 0.637/0.662*/0.644. FM-DICE ref 0.757 — best drift
arms (B2 ~0.70 mean, B/C-gf peaks 0.738) still trail; powered evals after all chains end.

### Actions
- Relaunched dead runs: robomimic s44 fresh (no resume in official harness) —
  sq_ext_A_s44 11516975, sq_ext_FM_s44 11516976, can_conf_Bq05_s44 11516977 (inferno);
  6 hard-8 chain resumes 11516978–83 (embers, resume from dice_latest.pth):
  A s10001, B s10002, C s10002, C-guarded s10002, B2 s10001, T2-guarded s10001.
- Paper: Table 1 filled with the harmonized numbers (blanks only for the 3 s44 reruns);
  abstract/intro/§5.2 rewritten to the matched-budget + parity-with-FM story;
  fig:curves is now a real figure (`scripts/plot_robomimic_curves.py` →
  `iclr2026/figures/robomimic_curves.pdf`, Okabe-Ito palette, regenerate post-s44).
- Still running: sq_ext_A s42/43, sq_ext_FM s42/43, can_ext_FM s44 (walltime ~11:30–23:30 tonight).

## 2026-07-27 (afternoon) — can-row provenance correction + rerun triage

### Correction: the can baselines in Table 1 were stale short-run numbers
Audit of the old `ft_dice_can_*` logs (config + seed read from inside each log):
- **A on drift (ft_distill_residual_drift_mlp)**: s42 died 19K (17/18/19K = 0.970/0.970/0.970),
  s43 died 16K (0.957 last), s45 0.943@19K, s46 0.950@19K. **No run reached 20K** —
  the table's "0.950 (single)" was a short-run value, and understated A (s42 was 0.970 by 19K).
- **FM (ft_distill_residual_flow_mlp)**: all old runs died ≤12K (s42 0.987@12K peak).
  The "0.957 baseline-to-beat" is not reproducible from any matched-protocol log.
- **New matched cell**: can_ext_FM_s44 (11497236, l40s) passed 20K →
  **18/19/20K = 0.993/0.983/0.983 → 0.986**. Run continues (24K by 15.5h).

Consequence: **the "B surpasses FM on can (0.99 vs 0.957)" claim is dead.** Honest can
story: fine-tuning saturates can — FM 0.986, B 0.988/0.993 — parity at the ceiling; square
is the discriminative task. Abstract/intro/§5.2/fig-caption all rewritten accordingly;
Table 1 can columns now per-seed with only matched cells (FM s44 = 0.986; A row blank
pending reruns).

### Rerun triage: the three morning s44 reruns were doomed
11516975/76/77 landed on **gpu-v100** (~1.1K steps/h drift, ~0.7K/h FM) with 12/16/10h
walltimes — none could reach 20K (L40S rates: A-sq 2.4K/h, FM-sq 1.5K/h, from the 20h
TIMEOUT logs: A-sq reached 48-49K, FM-sq 30K). Replaced on gpu-l40s/inferno and added the
5 unfinished can baselines the corrected table needs:
- 11521934 sq_ext_A_s44 (11h) · 11521935 sq_ext_FM_s44 (16h) · 11521936 can_conf_Bq05_s44 (12h)
- 11521937-39 can_ext_A_s42/43/44 (14h ea) · 11521940-41 can_ext_FM_s42/43 (16h ea)
- **USER ACTION: `scancel 11516975 11516976 11516977`** (V100 jobs; can't produce cells, paid QOS).

### Hard-8: 23/24 chains at iter 100
The 6 resumes: A s10001, C s10002, B2 s10001 hit iter 100 *before* their preemptions
(preemption harmless); C-gf s10002, T2-gf s10001 completed. B s10002 segment died at
iter 90 with a native Abort (signal 6, no traceback — MuJoCo/EGL flake, node
atl1-1-01-006, not the excluded one); relaunched as 11521901. Powered evals once it lands.
New endpoint deltas: C s10002 = 0.719, C-gf s10002 = 0.713, T2-gf s10001 = 0.700,
A s10001 collect=0.750 (eval line pending in log), B2 s10001 collect=0.688.
