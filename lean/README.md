# Lean4 verification for the OOD theory branch

This directory contains machine-checked statements for the mathematical stages
of the OOD theorem roadmap. Formalization is intentionally separate from the
research notes: a theorem is listed as verified only after `lake build` checks
it without `sorry`.

Current scope:

- `Stage9/Basic.lean`: finite-dimensional coverage definitions and elementary
  range/kernel facts;
- `Stage9/Support.lean`: exact support identity for an explicit product class;
  for `A = diag(a,0)`, the range-coordinate radius is `rho * sqrt(a)` because
  the gauge uses `A^dagger/2`;
- `Stage9/Calibration.lean`: scale calibration for a diagonal finite-dimensional
  operator.
- `Stage9/Control.lean`: monotone composition of explicit `rho`/`kappa`, learner
  sensitivity, and representation-residual upper bounds into the non-vacuity
  certificate.
- `Stage10/Basic.lean`: concrete two-dimensional hidden-direction, same-spectrum
  line-strength, and isotropic negative-control witnesses.
- `Stage11/Basic.lean`: exact restricted supervised quadratic-state expansion
  and empirical linear-state risk identity for squared-loss linear regression.
- `Stage11R/Basic.lean`: finite-source exposure-energy identity, concrete
  `exposureKernel = sourceAnnihilator` theorem, and abstract paired linear-map
  covariance theorem supporting the primal/dual revision.
- `Stage12/Basic.lean`: exact two-source affine transfer identity; general
  support minimality, quotient barriers, and exposure domination remain paper
  theorems with deterministic witnesses.
- `Stage13/Basic.lean`: Lean-checked V-REx exposure identity, finite two-group
  convex-mixture/max-risk algebra, CORAL label-shift witness, ideal-IRM
  stationarity identity, and a blind-direction witness. General RKHS,
  optimization, and Fishr higher-moment claims remain paper-level.
- `Stage13R/Basic.lean`: finite-dimensional affine environmental-sensitivity
  equals a finite support calculation, translation invariance of the affine
  increment, and a concrete separation witness showing that parameter-side
  derivatives can agree while environment derivatives and target risks differ.
  General path-integral, Taylor, and Gateaux-derivative claims remain paper
  proofs.
- `Stage13R1/Basic.lean`: algebraic nuisance-shift and pairwise-risk cancellation
  identities, plus the reclassified Stage 13R separation witness. Envelope,
  path, and quotient-space theorems remain paper-level.

The files currently use explicit Euclidean coordinates where this keeps the
formal proof auditable. Stage 11's exact quadratic identities are machine
checked; its Taylor remainder, coordinate/gauge audit, identifiability limits,
and exact-support comparison remain paper proofs with deterministic executable
witnesses. Stage 11R's quotient isomorphism, metric-aware projection, and
general convex-geometry statements remain paper proofs; its finite-source
annihilator equality and abstract paired covariance are Lean checked. The
control-composition lemma is formalized, while the general PSD
domination, principal-angle, rank-nullity, source-domain-count, and
Courant--Fischer bounds remain scientific obligations documented in the Stage 9
and Stage 10 notes. No unproved V-REx/MMD/DRO mapping is formalized.

The checked-in lakefile pins mathlib to the commit used by the local cache
(Lean 4.33.1). No build artifacts or machine-specific paths are committed;
Lake may reuse an existing `.lake/packages` cache or fetch the pinned source
when setting up a fresh checkout.

Run from this directory:

```text
lake build
```

To compile the Stage 11 file independently of the aggregate target:

```text
lake env lean OodTheoryVerification/Stage11/Basic.lean
```

This direct target check passes on the local Lean 4.33.1 toolchain.
