# Experiments revision — author notes

## Scope and deliverables

The revision changes only the Experiments section, its supporting displays, and a new standalone experimental supplement. Other manuscript sections, `root.tex`, `0-preamble.tex`, the bibliography, and class geometry are unchanged by this revision. The two preamble compatibility fixes visible in the working tree predate this task.

- Main paper: [root.pdf](root.pdf).
- Revised experiment source: [sections/05_experiments.tex](sections/05_experiments.tex), including [the four-paragraph physical study](sections/05_real_robot.tex) and the figure/table inputs it references.
- Experimental supplement: [experiments_supplement.pdf](experiments_supplement.pdf), built from [experiments_supplement.tex](experiments_supplement.tex) and [sections/05_experimental_supplement.tex](sections/05_experimental_supplement.tex).
- Numerical chart source: [figures/experiment_chart_data.json](figures/experiment_chart_data.json).
- Reproducible chart generator: [figures/generate_experiment_charts.py](figures/generate_experiment_charts.py).

## Revised outline

| Subsection | Purpose and evidence |
|---|---|
| Opening | State H1 useful refinement, H2 persistent reference, and H3 preservation/adaptation; identify the actor-update comparison. |
| A. Policy Improvement in Simulation | Establish per-task LIBERO gains across two bases; complement them with robomimic results; separate final success from environment steps to reach 90% success. |
| B. Real-Robot Refinement on Stair-Based Manipulation | Explain the physical tasks and staircase rationale, offline refinement and autonomous evaluation, measured gains, and release-failure interpretation. |
| C. The Role of Restoration and Clipping | Compare four implemented control configurations; distinguish preserving nonzero success from improving the base. |
| D. Choosing the Constraint and Critic Guidance | Define alternatives beside their evidence; explain task reversals and the role of the restoration radius. |
| E. Refinement under Dense and Sparse Rewards | Present DMC gains across both reward settings, followed by the H1--H3 evidence summary. |

## Display changes

- **Real robot:** retain the same eight committed demonstration frames and all eight measured success counts. Remove timestamps, clarify the staircase views, use Base/BC/DICE-RL/CAST labels and consistent colors, and preserve shared 0–100% axes. The caption separates teleoperation from autonomous evaluation.
- **Diffusion efficiency:** plot all three first evaluated checkpoints reaching 90% success for each method and their means. Both 160K CAST observations remain visible. This is not a reconstructed learning curve. The caption retains 496K endpoints.
- **Components:** show every task's change from its own base under every configuration, a separate task mean, a zero line, and a weak/radial/caps key. Points are tasks, not seeds; no error bars are inferred.
- **LIBERO:** retain every cell, both base blocks, both fixed settings, and the mean row. Clarify evaluation versus training seeds.
- **Alternatives:** preserve both campaign-specific CAST references. The latest experimenter correction specifies a 38.2% drifting square base in both design blocks, a 48.0% loss-space hinge result, and a 97.0% constraint-study CAST result. This supersedes the earlier 60.0% attribution. Preserve the guidance study's separate 91.2–93.0% CAST reference.

## Relocation map

| Previous main-text material | Main-text replacement | Supplement location |
|---|---|---|
| Evaluation counts, candidate count, restoration/clipping settings | Overview now names state/image inputs, sparse/dense reward, and real-robot evaluation; essential caption details remain | S1 Simulation Protocols |
| Robot cameras, encoder, chunks, control rates, episode composition, optimization and replay details | Main text restores offline dataset counts, 40 BC epochs, 6000 critic updates, and 20 autonomous trials; task/results figure retained | S2 retains detailed implementation |
| Complete robomimic table and external-baseline discussion | Main-text cross-generator table restored, alongside diffusion threshold chart | S3, Table S1 retains complete external baseline block |
| Component table | Task-level change chart | S4, Table S2; enabled-controls Table S3 |
| Detailed variant settings and campaign qualifications | Definitions immediately before each comparison | S4 Component and Design Studies |
| Radius-sweep details | Interpretation of positive radius, zero, and infinity | S4, Table S4 |
| Complete DMC table | Restored to main subsection E, with all dense/sparse rows; circle markers moved to the supplement, while the main text retains the unequal-budget qualification | S5, Table S5 retains supporting copy and details |
| Transport/shared-critic discussion | Removed from the Experiments narrative; existing conclusion still discusses these findings | S5, Table S6 retains full results |

## Numerical checks

The experimenter corrected constraint-square CAST from 94.0% to 97.0%, and subsequently corrected its base from 60.0% to 38.2% and the loss-space hinge from 28.0% to 48.0%. The latest baseline correction supersedes the earlier diffusion-base attribution; the separate diffusion efficiency comparison remains at a 60.0% base. Other measurement rows are preserved. Means and gains were recomputed from the available task entries before rounding; more precise raw evaluation records were not available.

- Drifting LIBERO means: base 66.125%, DICE-RL 66.9875%, CAST1 73.8%, CAST2 75.475%. DICE and CAST2 gains are 0.8625 and 9.35 pp, reported as 0.9 and 9.4 pp.
- Flow-matching LIBERO means: 75.3375%, 74.9875%, 75.6%, 77.6625%; no per-task best-setting selection.
- Component means: base 67.28%, radial-only 65.74%, full CAST 70.34%; mean changes −1.54 and +3.06 pp.
- Diffusion threshold means: 149333.33 and 240000 environment steps; reduction 37.78%. These are first evaluated checkpoints, not interpolated threshold times.
- Physical counts remain 11/16/18/19 of 20 for placement and 7/13/12/14 of 20 for stacking.
- External square references retain the corrected 34.8% same-harness base. DMC remains in return units. Table IV omits circle superscripts but retains the partial-budget qualification and tuned/high-variance markers; the supplement retains individual incomplete-run markers.

## Author verification list

1. **Evaluated robot checkpoints:** link the reported 20-trial counts to the final checkpoint paths/hashes. The corrected protocol uses 30 additional BC demonstrations per task; historical “success-only BC” labels do not establish the updated training dataset. Paired resets and the common post-release scoring interval are not recorded.
2. **Constraint-square corrections — resolved:** the latest experimenter instruction specifies the 38.2% drifting base, 48.0% hinge, and 97.0% CAST. The earlier 60.0% attribution is superseded. Keep the distinct CAST reference for the guidance study. Raw per-run manifests for historical simulation campaigns remain unavailable.
3. **Component replication:** the ICLR caption says “three seeds per cell,” but available aggregate files do not establish the training/evaluation seed identities. The revision does not promote this to a verified independent-training-seed claim and adds no variance estimate. Also verify executed configurations against the launcher's radial-only omission of the weak pull.
4. **Budgets and campaign matching:** recover the flow-matching-square interaction budget, actual budgets of incomplete can/DMC runs, and checkpoint/config mappings for the differing CAST references. Existing markers and separate study blocks remain in place.

## Build and review

Both documents are compiled with `/tmp/opencode/porygon-tex/bin/tectonic --keep-logs` from `2026_ICRA_CAST/`. Main-paper page limit: **at most nine pages including references**. No class, margin, or other manuscript-section changes are used to meet it. The stale equation reference in the experiment discussion is removed; the method remains untouched.

Final build: **8 main-paper pages including references**, plus a **5-page experimental supplement**. Both compile without unresolved citations/references or overfull boxes. Routine font/package and underfull-box warnings remain in the main build. The experiment pages and all supplement pages were visually inspected. Hash comparison confirms 21 protected non-experiment files are unchanged; all result-table measurements are preserved except the explicitly authorized square corrections recorded above.

## Follow-up requested by the experimenter

Hypotheses are now itemized. Simulation precedes the physical subsection. Main text restores the cross-generator and DMC tables and the actual offline physical data/training counts. Constraint-square uses the latest experimenter-corrected drifting base (38.2%), hinge result (48%), and CAST result (97%); projection's mug win remains visible. No other manuscript section changed.

## Hypothesis and protocol clarification

- LIBERO narrative, caption, and supplement now explicitly state training on all 90 tasks followed by refinement on the same eight challenging tasks. Means describe that selected subset.
- Result paragraphs and the closing synthesis connect performance/efficiency evidence to H1, the restoration component comparison to H2, and constraint reversals/radius sweeps to H3. Evidence is described as support, not proof.
- The corrected hinge result improves square by 9.8 percentage points over its 38.2% base, while trailing projection and CAST.
- Main DMC circle superscripts are removed for readability. The caption and narrative retain the fact that some runs end before the 400K target; the supplement identifies those runs. No partial run is relabeled as completed.

## Real-robot dataset correction

The latest experimenter description supersedes the older 18/24-rollout campaigns in both main text and supplement:

| Task | Pretraining demonstrations | Additional BC demonstrations | Shared RL rollouts |
|---|---:|---:|---:|
| Placement | 40 | 30 | 40 |
| Stacking | Approximately 60 | 30 | 40 |

Both successes and failures are retained in each RL rollout pool, and refinement remains offline. The 20 autonomous evaluations per task/method and all measured success rates are unchanged. Old outcome splits and campaign-specific checkpoint/collection-policy identifiers are removed from the manuscript protocol. Exact stacking pretraining count and any extra demonstration data used by RL are pending clarification; equal expert/rollout sampling is not asserted. Existing optimization budgets are retained from the documented recipe rather than inferred from the corrected episode counts.

## Results-focused experiment narrative

At the experimenter's request, the main Experiments section no longer has a standalone limitations paragraph. Repeated statements about unmeasured mechanisms, output guarantees, optimal radii, and audit limitations are removed from the narrative. The section retains the actual task-level orderings and explicit H1--H3 evidence links. DMC closes the evaluation with dense- and sparse-reward results. Essential budget/replication and component-comparison qualifications remain in captions and the supplement; the existing Discussion and Conclusion is unchanged. Robot checkpoint-linkage details remain in the supplement, whose cross-reference wording is synchronized. No measurements, tables, figures, or training-data counts change in this pass.
