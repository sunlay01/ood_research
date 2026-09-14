# OOD theory branch

This branch is the statistical/macroscopic theory line of the project. It is intentionally sparse: it does not inherit A/O/Pi, C/K, response-geometry, matched-fork, trajectory, or mechanism-memory machinery as foundational theory.

Start with [`STATUS.md`](STATUS.md), then [`research_program.md`](research_program.md) and the authoritative evidence under [`notes/evidence/`](notes/evidence/). There is currently no accepted core representation system.

For literature, follow [`literature/README.md`](literature/README.md): the compact ledger is authoritative, while historical surveys and audits are retained for provenance and are not default reading.

The companion branch `ood-algorithm` retains the operational network-level experiments. A rigorous bridge, if one is ever found, will be a new result rather than an assumption of this branch.

The notes tree is organized by epistemic role: `foundations/`, `evidence/`, and `framework_design/`. Superseded or failed work is under `archive/`; proof scaffolds are under `proofs/baselines/` until promoted explicitly.

## Absolute research mainline

This branch follows one scientific objective:

```text
source-only OOD/DG
  -> formalize target-risk upper bounds for regularized learners
  -> derive different regularizer bounds from one mathematical state and one master theorem
```

The current candidate core is an operator/gauge view of source exposure, under
revision by the environmental-sensitivity functional (ESF). The existing typed
bridge calculus remains the outer scaffold for source information, target
assumptions, identifiability, estimation, optimization, and failure semantics.
Neither candidate is accepted as a universal inner theorem yet.

### Required order

1. **Physical target family (narrow first).** Work initially in `H = R^d` with
   a compact, explicit family, such as an ellipsoid
   `U_phys={delta: delta^T Q^{-1} delta <= 1}` or a bounded target subspace.
   Define this family before selecting an operator and keep it external to
   source-derived statistics.
2. **Coverage control and non-vacuity (Stage 9 hard gate).** Finiteness is not
   enough. For the fixed physical family, derive explicit small upper bounds on
   `rho_A` and `kappa_A` from coverage domination (`Q <= cA`), principal-angle
   alignment, restricted excitation, or a shift generator `B`. Require the
   resulting transfer term `rho_A S_A + kappa_A N_A + 2 epsilon_repr` to be below
   a pre-registered tolerance (and below one for `[0,1]` losses). Keep target
   structure, learner sensitivity, and source/augmentation design as separate
   control paths.
3. **Geometry orientation.** Test source operators with matched rank, spectrum,
   trace, and condition number but different orientation relative to target
   shifts. Diversity counts alone are not accepted as coverage evidence.
4. **Risk-representation hard gate.** Before any algorithm mapping, establish in
   a restricted finite-dimensional exact model a task-relevant embedding and an
   affine risk representer with controlled residual. The embedding must have
   scientific meaning, be algorithm-independent, preserve needed conditional or
   label information, and yield controllable `g_f`, `C_S`, and representation
   error. If this gate fails, stop the unification claim; do not hide the issue
   behind a larger RKHS.
5. **Population master theorem.** Prove the operator/gauge bound with exposed
   sensitivity, target coverage, exposure-blind ambiguity, and representation
   residual as separate terms. Keep the theorem in `proofs/candidates/` until
   independently audited.
6. **Exact novelty audit (Stage 12.5).** Before deriving a new method, perform
   an explicit prior-art audit of the exact operator/support structure. Search
   for equivalent or near-equivalent results involving operator/gauge duality,
   range/kernel decomposition, pseudoinverse-normalized sensitivity,
   source-exposure covariance, target-support or coverage-radius calibration,
   and risk bounds of the form sensitivity times coverage plus a nullspace
   ambiguity term. Check primary papers and recent work, especially 2024-2026,
   across OOD/DG, DRO, RKHS robustness, operator learning, optimal recovery,
   and risk regularization. Broad keyword search or repository absence is not
   evidence of novelty; record exact overlaps, partial overlaps, and negative
   evidence before proceeding.
7. **Common corollaries.** Use the same scientific construction rule, state
   semantics, physical target family, and master theorem for V-REx, MMD, and
   DRO. The numerical operator may vary when the declared representation varies,
   but it must always be produced by one rule `A_S = A(P_1,...,P_m; Psi)`.
   A weaker trace-level MMD corollary or a negative result is valid; forced exact
   equivalence is not a goal.

   The intended division of labor is:

   | Method | Primary slot in the master theorem |
   |---|---|
   | V-REx | learner sensitivity |
   | MMD | exposure / representation geometry |
   | DRO | target uncertainty and support |

   “Unified” means these slots are generated by the same risk-transfer equation,
   not that every method is the same regularizer or must use numerically identical
   matrices.
8. **Bound-derived method.** If the common theorem survives and the exact novelty
   audit finds no equivalent construction, derive a computable regularizer from
   its bound. Do not invent a new algorithm before the theorem and novelty gate.
9. **Stable finite-sample geometry.** From the first population statement,
   maintain ideal quantities `(A, A^dagger, range(A), ker(A))` separately from
   estimable `A_lambda=A+lambda I` or spectral-cutoff `A_tau`. Quantify their
   discrepancy before adding operator concentration, estimation, and optimization
   errors. Then run synthetic tests before any large benchmark.
10. **Extensions last.** IRMv1, Fishr, MLDG, and broad universal-DG claims remain
   later work and must not drive the first theorem.

### Hard gates

- `ADVANCE` only when coverage radii have clear semantics independent of the
  regularizer, are finite for the initial physical family, and the population
  theorem is non-vacuous.
- `REVISE` when the theorem holds only after tightening the target family,
  embedding, spectral regularity, or stable-geometry assumptions.
- `REVISE` when the exact novelty audit is incomplete or finds a close overlap
  that requires a narrower claim or a different contribution.
- `KILL` the unification claim if every method requires a different inner state,
  if risk representation fails the hard gate, if nullspace ambiguity is always
  unbounded, or if the proposed certificate is no stronger than a relabeled
  existing bound. Do not design a new regularizer before Stage 12.5 passes.

### Current position

The finite-dimensional affine mother bound, PSD-operator/gauge inequality,
deterministic amplitude/rank probes, and controlled Stage 9 coverage conditions
are complete. Stage 10 is also complete: its principal-angle, rank/domain-count,
same-spectrum orientation, fixed-spectrum min-max, and negative-control results
are recorded in `notes/framework_synthesis/coverage_orientation_theorem.md`.
Stage 11 has now been audited in a restricted finite-dimensional supervised
model. Stage 11R completes the metric/gauge interface revision: the exact
support function, primal/dual pairing, exposure seminorm, and annihilator
quotient are the coordinate-free backbone; metric-dependent split quantities
are subordinate corollaries. Stage 12 now supplies the population theorem
package: exact transfer identity, sharp support minimality, source-observable
quotient, blind-direction impossibility, and exposed-span seminorm domination.
Stage 13 has now tested the method translations. The result is
`PARTIAL-UNIFICATION`: risk-level methods share exact/restricted translations,
fixed-state MMD/CORAL and ideal IRM have explicit restricted bridges, while
IRMv1 remains a surrogate and Fishr requires a richer state. New regularizer
design and large benchmarks remain blocked.
Stage 13R then tested the frozen environmental-sensitivity master functional
`S_D(f;xi)=sup_{delta in D(xi)} D_xi R(f,xi)[delta]`. Its exact affine reduction
to Stage 12, path-integral risk theorem, all-order separation from parameter-side
Moment Alignment, and cross-method support consequences survive deterministic
and Lean checks. The decision is **`REVISE-ESF-MASTER`**: a source-only global
certificate still needs an explicit path-envelope/coverage assumption, while
IRMv1 and Fishr do not have unconditional natural bridges. Stage 14 and any new
regularizer remain blocked until this interface is repaired or explicitly scoped
as an oracle/path-local theorem.
Stage 13R.1 subsequently ran the nuisance/invariance hard gate. Raw ESF changes
under predictor-independent additive environment difficulty; excess ESF removes
that nuisance but is endpoint/path equivalent to Moment Alignment's excess-risk
transfer measure. A pairwise risk quotient survives only as a relative-risk
object and does not yield the required absolute target-risk certificate. The
decision is **`FAIL-ESF-AS-INDEPENDENT-MASTER`**. Stage 12/13 support and
translation results remain valid, but ESF is no longer treated as an independent
inner master functional.
The project therefore continues with a separate **Master Functional Search**,
not a return to ad hoc method stitching. The first candidate screen is recorded
in `notes/framework_synthesis/master_functional_search.md`: transfer and
optimizer-diameter candidates are stopped, robust regret is crowded, and the
risk-landscape quotient is retained only for a bounded discrepancy hard gate.
Its natural sup-norm quotient is already exactly half classical loss-class
discrepancy, so no new master has been accepted.
The next research direction is **Stage 13R.2 — source identifiability of the
established transfer measure**, currently at **`REVISE-TRANSFER-LIFT`**. We treat
Zhang et al. (2021)'s and Moment Alignment's transfer measure as an imported
endpoint and retain the source-information layer: source quotient `V*/S°`, sharp
compatible-fiber certificate, exposed-span positive theorem, and blind-direction
impossibility. The regularizer lift is not yet generic: excess-risk affinity is
an extra assumption, native V-REx controls `Var R_e` rather than `Var E_e`, and
the source average is not Moment Alignment's center-of-mass reference. A native
V-REx route with an explicit `range_e R_e^*` correction is now the next gate.

Authoritative working notes:

```text
notes/framework_synthesis/operator_gauge_master_theorem.md
notes/framework_synthesis/regularization_ood_bound_methodology.md
notes/framework_synthesis/minimal_population_experiment.md
notes/framework_synthesis/exposure_geometry_novelty_audit.md
notes/framework_synthesis/physical_target_coverage.md
notes/framework_synthesis/experiments/physical_target_coverage_tests.py
notes/framework_synthesis/coverage_orientation_theorem.md
notes/framework_synthesis/experiments/coverage_orientation_tests.py
notes/framework_synthesis/risk_representation_hard_gate.md
notes/framework_synthesis/experiments/risk_representation_tests.py
notes/framework_synthesis/stage11r_metric_gauge_revision.md
notes/framework_synthesis/experiments/risk_representation_revision_tests.py
notes/framework_synthesis/stage12_population_master_theorem.md
notes/framework_synthesis/experiments/stage12_population_theorem_tests.py
notes/framework_synthesis/stage13_regularizer_translation.md
notes/framework_synthesis/experiments/stage13_regularizer_translation_tests.py
notes/framework_synthesis/stage13r_environmental_sensitivity_master.md
notes/framework_synthesis/stage13r_moment_alignment_separation.md
notes/framework_synthesis/stage13r_novelty_audit.md
notes/framework_synthesis/experiments/stage13r_esf_master_tests.py
lean/OodTheoryVerification/Stage13R/Basic.lean
notes/framework_synthesis/stage13r1_master_invariance_audit.md
notes/framework_synthesis/stage13r1_excess_vs_moment_alignment.md
notes/framework_synthesis/stage13r1_fixed_tangent_reaudit.md
notes/framework_synthesis/stage13r1_novelty_audit.md
notes/framework_synthesis/experiments/stage13r1_master_invariance_tests.py
lean/OodTheoryVerification/Stage13R1/Basic.lean
notes/framework_synthesis/master_functional_search.md
notes/framework_synthesis/master_functional_discrepancy_audit.md
notes/framework_synthesis/experiments/master_functional_search_tests.py
lean/OodTheoryVerification/MasterFunctionalSearch/Basic.lean
notes/framework_synthesis/stage13r2_transfer_measure_source_identifiability.md
notes/framework_synthesis/stage13r2_regularizer_certificates.md
notes/framework_synthesis/stage13r2_novelty_audit.md
notes/framework_synthesis/experiments/stage13r2_transfer_measure_tests.py
lean/OodTheoryVerification/Stage13R2/Basic.lean
```
