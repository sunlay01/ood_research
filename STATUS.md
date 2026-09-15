# OOD theory status

## Current branch

`ood-theory` contains the completed problem-first formalization audit and the
finite-dimensional theorem probes through Stage 13R.1. The current working
decision is **`FAIL-ESF-AS-INDEPENDENT-MASTER`**; no universal representation or
source-only global ESF certificate has been accepted.

## Epistemic status

The project has completed two evidence stages:

1. [`notes/evidence/ood_bridges/`](notes/evidence/ood_bridges/) — algorithm-specific OOD/DG proof bridges.
2. [`notes/evidence/framework_construction/`](notes/evidence/framework_construction/) — cross-domain lessons on how mature fields construct formal theories.

The resulting construction guide is [`formalization_framework_construction_guide.md`](notes/evidence/framework_construction/synthesis/formalization_framework_construction_guide.md). It ends with `CONSTRUCTION-GUIDE-READY` and does not select the project's final framework.

## What is authoritative

For a future GPT-6 design pass, read in this order:

1. `research_program.md`
2. `notes/foundations/*`
3. `notes/evidence/ood_bridges/extraction/*`
4. `notes/evidence/ood_bridges/synthesis/final_report.md`
5. `notes/evidence/framework_construction/extraction/*`
6. `notes/evidence/framework_construction/synthesis/formalization_framework_construction_guide.md`

Do not recursively read `archive/` by default. It contains failed or superseded reasoning retained for provenance. In particular, `archive/failed_formalization_round1/` is marked **FAILED — PREMATURE FRAMEWORK CONSTRUCTION**.

## Proof status

`proofs/active/` is intentionally empty. Existing source-only, robust, and conditional theorem scaffolds are in `proofs/baselines/`; their audit is in `proofs/audits/`. No theorem has been promoted as the project's active result.

## Current design result

### Synthetic failure-regime audit

The controlled latent experiment in
[`round3_redesign/task_failure_regime_synthetic/`](round3_redesign/task_failure_regime_synthetic/)
has completed its preregistered 32-cell x 5-seed factorial (800 checkpoint
rows). It uses frozen-head repair, nonlinear representation probes, true-core
oracle probes, nuisance/spurious counterfactual pairs, and balanced full
retraining. After removing circular priority labels and replacing raw-logit
contamination with probability-space `C_prob`, the verdict is
**`TWO-AXIS-STRUCTURE-SUPPORTED; CONTAMINATION-UNRESOLVED`**. The seed-level
interaction z-scores are -23.41 for `rho x sigma_c -> G_use`, 5.76 for
`k x optimizer -> G_repr`, and -8.41 for `k x d_z -> G_repr`. The follow-up
convergence-matched audit finds mean absolute matched `G_repr` difference
0.0173, so the optimizer interaction is not evidence of implicit bias under
this protocol. The dense sweep shows a capacity/difficulty trend but no single
phase boundary. These are continuous synthetic diagnostic results only. They
do not prove a theory or justify a universal taxonomy. Raw rows, checkpoint
summaries, cell summaries, and the final verdict are in the experiment's
`results/` directory.

The binary preference model is now exact under its declared conditional-noise
assumptions. With acquisition error `delta`, effective core reliability is
`q_t = 1-sigma_c-(1-2 sigma_c)delta`, and the source-optimal shortcut
coefficient exceeds the effective core coefficient iff `rho > q_t`. For the
pooled source mixtures used by the neural audit, high has `rho_bar=.90` and
remains shortcut-preferred at perfect acquisition `q_t=.80`; low has
`rho_bar=.60` and switches at `delta_c=1/3` when `sigma_c=.20`. Exact target
BCE and balanced-head repair gain are computed by finite-state enumeration. A
trajectory reanalysis yields 156 source-risk matched pairs, mean absolute
`G_repr` gap `0.0227`, and 89.7% of gaps below `0.05`, consistent with optimizer
speed/progress as the main confound. The added `q_hat` checkpoint diagnostic
finds 25/60 below-to-above crossings of `rho_bar=.90`; correlations with
counterfactual shortcut reliance and `G_use` are `0.240` and `0.310`. These
are descriptive coupling evidence, not a neural feature-acquisition theorem.

The design audit is recorded in `notes/framework_design/`. Read
`adequacy_audit.md`, `problem_instances.md`, `reuse_test.md`,
`information_sufficiency.md`, `bottlenecks.md`, the route files,
`comparison.md`, `novelty_check.md`, and `decision.md` in that order. Existing
IPM, conditional/causal, DRO, and meta-domain formalisms are adequate for the
declared problem instances; the remaining work is an algorithm-local theorem,
an explicit ambiguity term, or an impossibility result.

The subsequent bottom-up synthesis is recorded in
[`notes/framework_synthesis/`](notes/framework_synthesis/). Its entry point is
`README.md`; it starts from exact method cards, attacks candidate primitives,
and proposes a typed bridge calculus as a research architecture. This does not
promote a theorem to `proofs/active/` and does not replace the problem-first
audit.

An exposure-geometry probe is documented in
`notes/framework_synthesis/source_exposure_geometry.md` and
`exposure_geometry_novelty_audit.md`. It treats the typed calculus as an outer
scaffold and tests a source-domain covariance operator plus a target-relevant
task functional as a possible inner core; it is not yet accepted as the final
theory.

The minimal finite-dimensional affine population theorem and deterministic
falsification tests are documented in
`notes/framework_synthesis/regularization_ood_bound_methodology.md` and
`notes/framework_synthesis/minimal_population_experiment.md`. The tests verify
the projection identity, amplitude scaling, and rank-sensitive nullspace support.
The candidate operator/gauge abstraction is documented in
`notes/framework_synthesis/operator_gauge_master_theorem.md`; it separates
learner sensitivity, target coverage radius, and exposure-blind residual. This
advances the probe to `ADVANCE-TO-MASTER-THEORY`; it does not establish that
V-REx, MMD, and DRO are exact corollaries.

Stage 9 physical target coverage calibration is recorded in
`notes/framework_synthesis/physical_target_coverage.md`, with deterministic
checks in `notes/framework_synthesis/experiments/physical_target_coverage_tests.py`.
The compact-family finiteness, exact product support, external-family outer
bound, scale calibration, orientation precursor, and ideal/stable geometry
distinction pass. The upgraded control gate also passes for explicit
domination, restricted-excitation, shift-generator, and non-vacuity instances;
these are deterministic sufficient conditions, not claims that every physical
family is automatically covered. Algorithm mapping and new regularizer design
remain blocked.

The first machine-checked Stage 9 slice is under `lean/`: finite-dimensional
range/kernel facts, the exact product-class support identity (with the correct
`sqrt(a)` coordinate scale for `A = diag(a,0)`), scale calibration,
and the monotone control/non-vacuity composition lemma all pass `lake build` on
the local Lean 4.33.1 + mathlib cache. The general PSD-domination,
principal-angle, and shift-generator theorems remain explicit proof obligations;
they are not claimed as Lean-verified yet. Stage 10 geometry orientation is
complete on paper and has deterministic matrix checks in
`notes/framework_synthesis/experiments/coverage_orientation_tests.py`; its
concrete witnesses are also machine-checked under `lean/OodTheoryVerification/Stage10`.

## Stage 11 risk-representation hard gate

The restricted finite-dimensional audit is complete and recorded in
[`notes/framework_synthesis/risk_representation_hard_gate.md`](notes/framework_synthesis/risk_representation_hard_gate.md), with deterministic checks in
[`notes/framework_synthesis/experiments/risk_representation_tests.py`](notes/framework_synthesis/experiments/risk_representation_tests.py).
The exact supervised quadratic state `(E[XX^T], E[XY], E[Y^2])` passes,
including conditional/label information and zero residual in the declared
model. The overall decision is **`REVISE`**, not `ADVANCE`: Euclidean split
quantities such as `N_A` and pseudoinverse radii are not invariant under
arbitrary non-orthogonal coordinate changes. The theorem interface must
therefore declare its metric/gauge or be rewritten in a coordinate-covariant
form; nullspace sensitivity is also not source-risk identifiable without extra
structure.

`lean/OodTheoryVerification/Stage11/Basic.lean` machine-checks the exact
restricted squared-loss state/risk identities. Taylor bounds, coordinate-gauge
claims, identifiability counterexamples, and exact-support comparisons remain
paper proofs plus deterministic executable witnesses.

Stage 11 was also compiled as an independent Lean target with
`lake env lean OodTheoryVerification/Stage11/Basic.lean` (Lean 4.33.1, local
mathlib cache), in addition to the repository-wide `lake build`.

## Stage 11R metric/gauge interface revision

Stage 11R is complete and recorded in
[`notes/framework_synthesis/stage11r_metric_gauge_revision.md`](notes/framework_synthesis/stage11r_metric_gauge_revision.md), with deterministic checks in
[`notes/framework_synthesis/experiments/risk_representation_revision_tests.py`](notes/framework_synthesis/experiments/risk_representation_revision_tests.py).
The coordinate-free backbone is the primal/dual pairing, exact target support
`h_U(g)`, source exposure operator `A : V* -> V`, invariant seminorm
`sqrt(g(A g))`, and source-observable quotient `V*/S°`. `N_A` is no longer
treated as a primitive source-risk-observable quantity. Metric-dependent
projections and `rho`/`kappa` are valid only relative to a declared metric `G`
and its paired coordinate transformation. Lean now independently verifies the
finite-source exposure-energy identity, concrete `ker A = S°` equality, and an
abstract linear-map paired-covariance theorem in
`lean/OodTheoryVerification/Stage11R/Basic.lean`; the quotient isomorphism and
general convex-geometry statements remain paper proofs.

The revised decision is **`ADVANCE-TO-STAGE-12`**. Keep the Stage 10
orientation theorem fixed. Stage 12 may now formulate the population master
theorem, but algorithm mapping, large benchmarks, and new regularizer design
remain blocked until that theorem and the later Stage 12.5 exact novelty audit
pass.

## Stage 12 population theorem package

Stage 12 is complete in finite dimension and recorded in
[`notes/framework_synthesis/stage12_population_master_theorem.md`](notes/framework_synthesis/stage12_population_master_theorem.md), with deterministic checks in
[`notes/framework_synthesis/experiments/stage12_population_theorem_tests.py`](notes/framework_synthesis/experiments/stage12_population_theorem_tests.py).
It contains the exact residual transfer identity, the proof that `h_U(g)` is
the smallest uniform additive certificate given only `delta_T in U`, the
source-observable quotient `V*/S°`, the compatible-fiber ambiguity lower bound,
the unbounded blind-direction impossibility theorem, and the exposed-span
ellipsoid corollary `h_U(g) <= rho * sqrt(g(A g))` with equality witnesses.
The package includes both a positive domination theorem and an impossibility
theorem. The exact transfer algebra has a Lean witness in
`lean/OodTheoryVerification/Stage12/Basic.lean`; general supremum,
annihilator, quotient, and convex-geometry arguments remain paper proofs with
deterministic tests.

## Stage 13 regularizer translation gate

Stage 13 is complete in the finite-dimensional scope and recorded in
[`notes/framework_synthesis/stage13_regularizer_translation.md`](notes/framework_synthesis/stage13_regularizer_translation.md), with deterministic checks in
[`notes/framework_synthesis/experiments/stage13_regularizer_translation_tests.py`](notes/framework_synthesis/experiments/stage13_regularizer_translation_tests.py).
The common affine supervised state is reused by ERM, V-REx, GroupDRO, MM-REx,
fixed-state MMD, CORAL, and restricted ideal IRM. V-REx and finite-group
GroupDRO have exact translation theorems; MM-REx is exact for a declared
bounded affine coefficient set. IRMv1 is a fixed-scale relaxation, and Fishr
requires higher-order derivative moments outside the minimal state.

The decision is **`PARTIAL-UNIFICATION`**: this is stronger than a notation-only
wrapper but does not establish a universal exact theory for all methods.
`lean/OodTheoryVerification/Stage13/Basic.lean` machine-checks the exact
V-REx exposure identity, finite two-group convex-mixture/max-risk algebra,
CORAL label-shift witness, ideal-IRM stationarity, and a blind-direction
witness. General RKHS, learned-representation, quotient, optimization, and
Fishr higher-moment bridges remain paper proofs with deterministic tests.

## Stage 13R environmental-sensitivity revision gate

Stage 13R is recorded in
[`notes/framework_synthesis/stage13r_environmental_sensitivity_master.md`](notes/framework_synthesis/stage13r_environmental_sensitivity_master.md), with the direct Moment Alignment comparison in
[`notes/framework_synthesis/stage13r_moment_alignment_separation.md`](notes/framework_synthesis/stage13r_moment_alignment_separation.md), the recent-literature audit in
[`notes/framework_synthesis/stage13r_novelty_audit.md`](notes/framework_synthesis/stage13r_novelty_audit.md), deterministic checks in
[`notes/framework_synthesis/experiments/stage13r_esf_master_tests.py`](notes/framework_synthesis/experiments/stage13r_esf_master_tests.py), and a stable algebraic Lean slice in
`lean/OodTheoryVerification/Stage13R/Basic.lean`.

The frozen functional is
`S_D(f;xi) = sup_{delta in D(xi)} D_xi R(f,xi)[delta]`, with `D` declared
independently of any native algorithm. Gate A passes: ESF is coordinate/gauge
covariant, reduces exactly to the Stage 12 support function in affine models,
and has an explicit nonlinear Taylor remainder. Gate B passes as a genuine
absolutely-continuous path theorem, but source-local sensitivity is not a
source-only global certificate without path coverage/envelope control. Gate C
gives exact/dual/projection/restricted translations for several methods; IRMv1
remains a surrogate and Fishr has no natural bridge in the minimal state. Gate D
passes via an all-order parameter-derivative versus environment-derivative
separation witness for Moment Alignment. Gate E yields shared path-control and
support-ordering consequences. The final decision is
**`REVISE-ESF-MASTER`**, not `ADVANCE-ESF-MASTER` and not
`FAIL-NOVELTY-MOMENT-ALIGNMENT`.

Lean status is deliberately narrow: the Stage 13R file checks finite affine
increment/support algebra, translation invariance, and a concrete separation
witness. General Gateaux derivatives, path integration, Taylor bounds, and
literature-level novelty claims remain paper proofs with deterministic tests.

## Stage 13R.1 master-functional invariance gate

Stage 13R.1 is recorded in
[`notes/framework_synthesis/stage13r1_master_invariance_audit.md`](notes/framework_synthesis/stage13r1_master_invariance_audit.md), with the excess-risk comparison in
[`notes/framework_synthesis/stage13r1_excess_vs_moment_alignment.md`](notes/framework_synthesis/stage13r1_excess_vs_moment_alignment.md), the fixed-tangent re-audit in
[`notes/framework_synthesis/stage13r1_fixed_tangent_reaudit.md`](notes/framework_synthesis/stage13r1_fixed_tangent_reaudit.md), the novelty audit in
[`notes/framework_synthesis/stage13r1_novelty_audit.md`](notes/framework_synthesis/stage13r1_novelty_audit.md), deterministic tests in
[`notes/framework_synthesis/experiments/stage13r1_master_invariance_tests.py`](notes/framework_synthesis/experiments/stage13r1_master_invariance_tests.py), and stable algebraic Lean checks in
`lean/OodTheoryVerification/Stage13R1/Basic.lean`.

The gate finds that raw ESF changes under `R -> R+c(xi)`, even though `c` is
predictor-independent environment difficulty. Excess ESF removes this nuisance,
but under explicit Danskin/envelope and path regularity assumptions it is exactly
a differential/path representation of Moment Alignment's excess-risk transfer
measure. The original Stage 13R witness therefore proves information separation,
not predictor-relevant OOD separation. A pairwise risk quotient is nuisance
invariant, but it only certifies relative risk/ranking and does not yield the
required absolute target-risk certificate. Re-auditing V-REx, GroupDRO, fixed-
state MMD, and ideal IRM under one externally fixed Euclidean tangent ball gives
only coverage-dependent upper bounds, relaxations, or structural surrogates, not
a new common ESF theorem.

The final decision is **`FAIL-ESF-AS-INDEPENDENT-MASTER`**. This is not a failure
of the Stage 12/13 support calculus: those local results remain valid and
authoritative. It means ESF should no longer be presented as an independent inner
master for absolute OOD risk. Any future relative-risk quotient or alternative
master requires a new scientific target and a separate theorem/novelty gate.

## Master Functional Search (active, no accepted master)

Stage 13R.1 killed ESF as an independent master, not the search for a master
functional. The first candidate screen is recorded in
[`notes/framework_synthesis/master_functional_search.md`](notes/framework_synthesis/master_functional_search.md), with the risk-landscape discrepancy hard gate in
[`notes/framework_synthesis/master_functional_discrepancy_audit.md`](notes/framework_synthesis/master_functional_discrepancy_audit.md) and deterministic checks in
[`notes/framework_synthesis/experiments/master_functional_search_tests.py`](notes/framework_synthesis/experiments/master_functional_search_tests.py). A finite algebraic slice is checked in
`lean/OodTheoryVerification/MasterFunctionalSearch/Basic.lean`.

The search is explicitly conducted at the risk-functional level
`R_P : F -> R`, not as another scalar sensitivity for one predictor. The first
screen stops the transfer-functional candidate as already covered by Moment
Alignment, stops optimizer-map diameter because it has no target-risk
certificate, and marks robust regret as crowded with robust decision/DRO
theory. The risk-landscape quotient modulo environment-only constants remains
`REVISE`: under the natural sup norm it is exactly half the classical
loss-class pairwise discrepancy of Mansour--Mohri--Rostamizadeh/Ben-David, so a
non-arbitrary alternative geometry plus a genuinely new cross-method theorem
would be required. This is a search-stage result, not an accepted master or a
novelty claim.

## Stage 13R.2 transfer-measure source-identifiability theorem

The project now accepts the transfer measure as an imported endpoint rather than
claiming it as a new master. Stage 13R.2 is recorded in
[`notes/framework_synthesis/stage13r2_transfer_measure_source_identifiability.md`](notes/framework_synthesis/stage13r2_transfer_measure_source_identifiability.md), with method certificates in
[`notes/framework_synthesis/stage13r2_regularizer_certificates.md`](notes/framework_synthesis/stage13r2_regularizer_certificates.md), the primary-source audit in
[`notes/framework_synthesis/stage13r2_novelty_audit.md`](notes/framework_synthesis/stage13r2_novelty_audit.md), deterministic tests in
[`notes/framework_synthesis/experiments/stage13r2_transfer_measure_tests.py`](notes/framework_synthesis/experiments/stage13r2_transfer_measure_tests.py), and a finite algebraic Lean slice in
`lean/OodTheoryVerification/Stage13R2/Basic.lean`.

The theorem package lifts Stage 12 to an established transfer endpoint only
under additional conditions. An affine representation for raw risk does not
imply one for excess risk, so `(ER-affine)` is an explicit extra assumption;
the centered source average is also distinct from Moment Alignment's
center-of-mass reference. Under `(ER-affine)`, the support/quotient machinery
gives the centered-excess identity, compatible-fiber lower bound, and blind
direction impossibility. Native V-REx instead controls `Var_e R_e(f)` and can be
connected to the imported transfer measure only through a conditional bound
with correction `(1/2) range_e R_e^*`. GroupDRO/MM-REx/MMD/CORAL/IRM remain
restricted or projection certificates, not generic exact translations.

This is not a new transfer measure, discrepancy, or Moment Alignment theorem.
The defensible potential contribution is the source-observability, sharp
certificate, and impossibility layer at that established endpoint. The generic
regularizer lift is not yet accepted. The current direction is
**`REVISE-TRANSFER-LIFT`**, not a publication-level novelty pass. New
regularizer design and Stage 14 remain blocked. The next certificate gate must
compare theorem-level overlap with Xu et al. (ICML 2022) and Partial
Transportability for Domain Generalization (NeurIPS 2024), in addition to the
already checked transfer-measure papers.

## Next authorized step

Do not claim `UNIFICATION-PASS`, design a new regularizer, or run large
benchmarks. Stage 13R.1 terminates ESF as an independent inner master: preserve
the valid Stage 12/13 local support and translation results. Stage 13R.2 is
currently `REVISE-TRANSFER-LIFT`; the next task is a strict native
regularizer-to-transfer certificate gate with explicit optimum-risk,
representation, source-center, and target-coverage assumptions. Do not enter
Stage 14 until that gate and the expanded novelty comparison pass.
Keep `proofs/active/` empty until any local theorem is independently audited and
promoted. The Stage 12.5 novelty audit is complete; any future new method must
still pass a separately defined theorem and novelty gate.
