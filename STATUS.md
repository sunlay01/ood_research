# OOD theory status

## Current branch

`ood-theory` contains the completed problem-first formalization audit. The
working decision is `YES-WITH-LOCAL-THEOREM`; no new universal representation
has been accepted.

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

## Stage 11R metric/gauge interface revision

Stage 11R is complete and recorded in
[`notes/framework_synthesis/stage11r_metric_gauge_revision.md`](notes/framework_synthesis/stage11r_metric_gauge_revision.md), with deterministic checks in
[`notes/framework_synthesis/experiments/risk_representation_revision_tests.py`](notes/framework_synthesis/experiments/risk_representation_revision_tests.py).
The coordinate-free backbone is the primal/dual pairing, exact target support
`h_U(g)`, source exposure operator `A : V* -> V`, invariant seminorm
`sqrt(g(A g))`, and source-observable quotient `V*/S°`. `N_A` is no longer
treated as a primitive source-risk-observable quantity. Metric-dependent
projections and `rho`/`kappa` are valid only relative to a declared metric `G`
and its paired coordinate transformation.

The revised decision is **`ADVANCE-TO-STAGE-12`**. Keep the Stage 10
orientation theorem fixed. Stage 12 may now formulate the population master
theorem, but algorithm mapping, large benchmarks, and new regularizer design
remain blocked until that theorem and the later Stage 12.5 exact novelty audit
pass.
Keep `proofs/active/` empty until any local theorem is independently audited and
promoted. Do not begin V-REx/MMD/DRO mapping or a new regularizer before Stage
12.5 exact novelty audit is scheduled after the population theorem.
