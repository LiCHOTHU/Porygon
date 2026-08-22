# Porygon — experiment plan

**Thesis.** For one-step generative policies, critic-guided RL should be performed as
*constrained action-space target transport*, not direct critic backpropagation. The critic
proposes an action-space improvement; an explicit action-space trust region limits how much of
that proposal the actor is asked to learn; only the constrained target is regressed.

`Porygon = constrained critic-guided field-target regression.` The anchor is **one**
implementation of the constraint, not the headline.

Section structure and status of every cell: `iclr2026/sections/05_experiments.tex`.
Running notes and provisional numbers: `DEVLOG.md`. **No provisional number enters the paper.**

---

## 1. Hypotheses — stated as claims

Each is a falsifiable assertion, not a question. The paper's job is to support or refute each one.

| | claim | table / figure | status |
|---|---|---|---|
| **H1** | *Porygon matches the strongest existing fine-tuners. The methods closest to it, which have no action-space constraint, do not improve their base at all.* | `tab:baselines` | 2/5 external cells |
| **H2** | *Change only the actor update, and Porygon gets more out of the same critic than backpropagating -Q does.* | `tab:matched`, `tab:libero-single`, `fig:curves` | **complete** |
| **H3** | *Take the action-space constraint away, and the update destroys the policy.* | `tab:constraint`, `fig:containment` | 1/4 cells |
| **H4** | *The constraint has to act on the action target. The same shape placed in the loss, or on the parameter gradient, does not work.* | `tab:locus` | 2/5 rows |
| **S1** | *The critic's gradient direction matters. Porygon is not winning just by moving less.* | `tab:resolution` | complete |
| **S2** | *The update needs only a frozen noise-to-action map, so it transfers to other generator types.* | `tab:libero-single` lower block | 4/8 tasks |

**H1 is deliberately not a superiority claim.** We are at parity with FM+DICE-RL and with GRPO;
claiming otherwise invites rejection on our weakest axis. Its second clause is what earns the
table its place.

## 1b. What each baseline is for

Baselines here do three different jobs. Mixing them is what made the old draft hard to read, and
each job wants a different table.

| baseline | job | proves | lives in |
|---|---|---|---|
| **Backprop actor** (DICE-RL actor on *our* base) | **control**, not a baseline | the gain is *caused* by the update form — everything else is byte-identical | H2, `tab:matched` |
| **FM + DICE-RL** (published pipeline, own base) | external reference | we are competitive with the strongest published system, from a much weaker base | H1, `tab:baselines` |
| **DIPO, QSM, DQL, IDQL, AWR** (official code, own base, own budget) | **family members lacking our mechanism** | the unconstrained form of *our own* update fails externally, at 125x the budget — external replication of `B_NONE` | H1 table, argument lands in H3 |
| **GRPO** (critic-free PG, matched budget) | alternative paradigm | we match what a policy gradient extracts, without its noise-injection surrogate | H2 prose |
| **Frozen base** (no RL) | reference line | headroom exists; and every arm's Delta is measured against *its own* base | every table |
| **Porygon top-k / tilted** | *not baselines* — our own dial | the gradient direction is load-bearing; we are not winning by being conservative | S1, `tab:resolution` |

Two consequences worth being explicit about in the text:

- **The DIPO/QSM rows are not there to be beaten.** They are there because they are our
  unconstrained corner, run by their own authors. If we frame them as competitors we invite the
  budget objection; framed as the corner, the 125x budget makes the point *stronger*.
- **The classic diffusion-RL baselines are robomimic-only** — their official implementations do
  not support LIBERO. State this in the caption rather than leaving the gap silent. LIBERO's H1
  comparison is carried by FM+DICE-RL and GRPO.

## 2. Standing conventions

- **Train:** `sbatch scripts/field_single_task.sbatch <ARM> <TASK> <SEED> [TAG] [HYDRA OVERRIDES...]`
  Arms: `A` (backprop control) · `B` (frozen recipe) · `B_CLIP` / `B_ANC` / `B_NONE` (H3 factorial).
  Extra args after `TAG` pass straight through to Hydra — no launcher edit needed for sweeps.
- **FM base:** prefix `BASE_CKPT=<fm ckpt> NUM_INF_STEPS=10` and add them to `--export`.
- **Checkpoints:** `$CEDAR/imitation/experiments_dice/libero/libero_90/<RL_NAME>/dice_latest.pth`
  where `CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch`.
- **Eval (the only protocol that counts):** powered = 100 rollouts × 3 eval seeds, last-3
  checkpoints. `sbatch --export=ALL,CELL=<cell>,LABEL=<label>,CKPT=<path>,TASK=<t> scripts/powered_eval_single.sbatch`
- **Tasks:** t65 (base 0.573, weak critic ‖∇ₐQ‖≈0.035) and t32 (base 0.610, strong critic ≈0.55)
  are the standard ablation pair — they sample opposite critic regimes deliberately.
- **Seed caution:** t65 carries a measured ±8.9pt spread. Single-seed differences below ~9pt on
  t65 are not claims. t32/t53/t73/t75 spreads are 0.000–0.019 and are safe at a few points.

---

## 3. Wave 1 — the two experiments that decide the paper

Roughly 15 training runs. Everything else waits on these.

### 1a. `DICE + hinge` vs `B_ANC` — the falsification test  ⚠️ highest priority

**Why.** By `prop:equiv`, putting Porygon's anchor potential λ(‖r‖−ρ)₊² into DICE's *loss*
yields the same first-order gradient as `B_ANC` (anchor on, clips off). The two therefore share
constraint geometry and differ **only in the update form**. This is the one experiment that can
falsify the thesis, and it is two-sided:

- `DICE+hinge` ≈ `B_ANC` → geometry is the whole story, the form is presentation, **H2 collapses into H4**.
- `DICE+hinge` fails where `B_ANC` works → the form is the mechanism, thesis nailed.

**Code change required** (not a launcher flag). In `imitation/algos/dice/distill_rl.py::actor_loss`,
`mse_per_sample` (~L654) is ‖r‖² per sample. Add a config-gated hinge:

```python
# dice.bc_hinge_rho: 0.0 reproduces current behaviour EXACTLY ((rms-0)_+^2 == mse)
if self.bc_hinge_rho > 0.0:
    rms = mse_per_sample.clamp_min(1e-12).sqrt()
    mse_per_sample = (rms - self.bc_hinge_rho).clamp_min(0.0).pow(2)
```

Backward compatibility is exact at `bc_hinge_rho=0`, so no existing number moves. Verify with
`scripts/verify_bc_filter_fix.py` before launching.

```bash
for T in 65 32; do
  sbatch scripts/field_single_task.sbatch A $T 10000 _hinge +dice.bc_hinge_rho=0.05
  sbatch scripts/field_single_task.sbatch A $T 10001 _hinge +dice.bc_hinge_rho=0.05
done
```

### 1b. H3 factorial — the clean 2×2

**Why.** The published ρ-sweep never isolated the guards. Every Porygon arm ships
`bc_step_size=0.05`, and in `field_pointwise` mode `bc_field = -old_res`, so `restore_radius→∞`
zeroes only `restore_delta` — the −0.05·r pull survives. "Clip-only works" was never measured.
Both corners are contaminated; only the 0.000 cell is clean.

Arms are already in the launcher. "Clip off" disables **both** `q_max_norm` and
`total_max_norm` (1e9, verified an exact no-op in `clip_field_norm`); "anchor off" sets **both**
`bc_step_size=0` and `restore_step_size=0`.

```bash
for T in 65 32; do
  for ARM in B_CLIP B_ANC B_NONE; do
    sbatch scripts/field_single_task.sbatch $ARM $T 10000
  done
done
# B cells already measured: t65 0.781, t32 0.684
```

**Expected:** `B_NONE` ≈ 0.000 (reproducing DIPO/QSM inside our system). If it is *not* ~0.000,
that is the most important result of the wave and H3 needs rewriting, not the launcher.

---

## 4. Wave 2 — completes H4 and instruments H3

### 2a. Parameter-space rung of `tab:locus` — cheaper than it looks

`dice.grad_clip: 1.0` is **already applied to the actor unconditionally** (`dice_train.py:388`),
so the backprop control already carries a parameter-space trust region and still gains +0.9.
This is a sweep of an existing knob, not a new feature.

```bash
for GC in 0.1 1.0 10.0 1e9; do
  sbatch scripts/field_single_task.sbatch A 65 10000 _gc${GC} dice.grad_clip=${GC}
done
```

### 2b. Hard base projection — soft-vs-hard column of `tab:locus`

Π_{‖r‖≤ρ}(r + V_Q): replaces the soft restoring field with a conventional hard trust region.
Small change in `field_actor_loss` next to the existing `gate` computation (clamp the residual
target's norm instead of pulling it back). 2 runs.

### 2c. `P_out` logging + `D_base(t)` curves — nearly free

`residual_norm` is **already computed every actor step and printed into the plain-text
per-iteration log line** (`dice_train.py:427`). So `D_base(t)` is recoverable from `.log` files
for every run ever done, including all of Wave 1 and the backprop control — *check the logs
before spending compute here.* Only `P_out` needs instrumenting: one line beside the `gate`
computation in `field_actor_loss`, `(rn > restore_radius).float().mean()`.

`fig:containment` is the substance of H3 — the 2×2 table is one bit (three cells "works", one
0.000); the residual trajectories are what show containment happening rather than merely being
present.

---

## 5. Wave 3 — defensibility

### 3a. Seeds ⚠️ under-prioritised everywhere so far

`tab:constraint`, `tab:locus`, `tab:rho-sweep` and `tab:libero-ablation` are all single-seed on
a benchmark with a documented ±8.9pt spread on t65. The 0.000 cells are unambiguous;
**everything else in those tables is inside noise of everything else**, and a reviewer will say
so. Two extra seeds on the Wave-1 and Wave-2 rows (~10–15 runs) is the difference between a
mechanism section and a suggestion.

### 3b. Finish in-flight work

- DQL / IDQL / AWR — running via `scripts/baseline_inferno.sh` on `ideas_l40s`. Harvest only.
- FM base 4/8 → 8/8 tasks (t8, t21, t73, t81) so the lower block of `tab:libero-single` gets a
  row mean. Completes an existing claim rather than opening a new one.

---

## 6. Deferred — and why

| item | why deferred |
|---|---|
| **MimicGen suite** | Scope creep at submission time. Our claims are mechanistic, not breadth claims; a new suite buys generality we never asserted, at the cost of the seeds that defend what we did. |
| **FPO** | Strengthens H1 — the lowest-return hypothesis — at the highest implementation cost, and by our own related-work argument it can only run on the FM base (no denoising loss over τ on a one-step drifter). Cite as the acknowledged missing comparison. |
| **Long-horizon clip-vs-anchor** | It is an anchor-vs-clip experiment, the question the reframing retired. Run as robustness only, and be prepared for "both stay stationary" — which is a fine answer that must not be allowed to restructure the section. |
| **DMC finetune arms** | Plumbing and base are done (2026-08-08). Domain generality, not generator generality; below the mechanism experiments. |
| **ρ, η_anc secondary sweeps** | `tab:rho-sweep` already establishes the flat band. More resolution there answers nothing new. |

---

## 7. Decision rules — written before the runs

| result | consequence |
|---|---|
| `B_NONE` ≈ 0.000 | H3 confirmed; DIPO/QSM rows in `tab:baselines` become external replication of our own corner |
| `B_NONE` ≫ 0 | H3 is wrong as stated; the 0.000 in `tab:rho-sweep` was configuration-specific and must be re-diagnosed before anything else runs |
| `DICE+hinge` ≈ `B_ANC` | thesis fails: geometry, not form. Rewrite around "the right regularizer shape", drop the form claim, keep H1/H3/S1 |
| `DICE+hinge` ≪ `B_ANC` | thesis confirmed at its sharpest point; this becomes the headline row of `tab:locus` |
| `grad_clip` sweep flat and at base | H4's parameter-space rung closed; "isn't this just gradient clipping?" is answered with data |
| `B_CLIP` ≈ `B_ANC` ≈ `B` | expected; report as "either implementation suffices", do **not** re-open anchor-vs-clip |

---

## 8. Cost summary

| wave | runs | what it buys |
|---|---|---|
| 1 | ~10 train + evals | the two hypotheses that carry the paper |
| 2 | ~6 train + log scraping | `tab:locus` complete, `fig:containment` |
| 3 | ~15 train | seeds; in-flight harvest |

Waves 1–2 are ~16 training runs and close every open cell in H3 and H4.
