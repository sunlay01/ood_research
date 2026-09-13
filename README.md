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

The current candidate core is an operator/gauge view of source exposure. The
existing typed bridge calculus remains the outer scaffold for source information,
target assumptions, identifiability, estimation, optimization, and failure
semantics. It is not the inner unifying theorem.

### Required order

1. **Physical target family.** Define an externally meaningful target family
   before selecting an operator. For a source-derived positive operator `A`,
   compute the induced exposed coverage radius and the exposure-blind radius.
   Keep these target assumptions separate from source-derived statistics.
2. **Coverage calibration.** Prove when the radii are finite, when support is
   exact, and when the nullspace term is unavoidable. Check scale invariance for
   a fixed physical target family.
3. **Geometry orientation.** Test source operators with matched rank/spectrum but
   different orientation relative to target shifts. Rank, trace, or domain count
   alone are not accepted as coverage evidence.
4. **Common risk representation.** In a restricted finite-dimensional model,
   establish a task-relevant embedding and an affine risk representer with an
   explicit residual. Reject any construction that changes the state merely to
   fit each algorithm.
5. **Population master theorem.** Prove the operator/gauge bound with exposed
   sensitivity, target coverage, exposure-blind ambiguity, and representation
   residual as separate terms. Keep the theorem in `proofs/candidates/` until
   independently audited.
6. **Common corollaries.** Only after steps 1-5, test V-REx, MMD, and DRO using
   the same state, physical target family, and master theorem. A valid outcome
   may be a weaker trace-level MMD corollary or a negative result; forced exact
   equivalence is not a goal.
7. **Bound-derived method.** If the common theorem survives, derive a computable
   regularizer from its bound. Do not invent a new algorithm before the theorem.
8. **Finite-sample and experiments.** Add spectral cutoff/ridge stability,
   operator concentration, estimation and optimization errors, then run synthetic
   tests before any large benchmark.
9. **Extensions last.** IRMv1, Fishr, MLDG, and broad universal-DG claims remain
   later work and must not drive the first theorem.

### Hard gates

- `ADVANCE` only when coverage radii have clear semantics independent of the
  regularizer and the population theorem is non-vacuous.
- `REVISE` when the theorem holds only after tightening the target family,
  embedding, or spectral regularity assumptions.
- `KILL` the unification claim if every method requires a different inner state,
  if the nullspace ambiguity is always unbounded, or if the proposed certificate
  is no stronger than a relabeled existing bound.

### Current position

The finite-dimensional affine mother bound, PSD-operator/gauge inequality, and
deterministic amplitude/rank falsification probes are complete. The current
priority is **physical target coverage calibration**, followed by orientation and
risk-representation theorems. No algorithm-specific mapping, new regularizer,
large benchmark, or universal novelty claim should be started before those gates
pass.

Authoritative working notes:

```text
notes/framework_synthesis/operator_gauge_master_theorem.md
notes/framework_synthesis/regularization_ood_bound_methodology.md
notes/framework_synthesis/minimal_population_experiment.md
notes/framework_synthesis/exposure_geometry_novelty_audit.md
```
