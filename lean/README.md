# Lean4 verification for the OOD theory branch

This directory contains machine-checked statements for the mathematical stages
of the OOD theorem roadmap. Formalization is intentionally separate from the
research notes: a theorem is listed as verified only after `lake build` checks
it without `sorry`.

Current scope:

- `Stage9/Basic.lean`: finite-dimensional coverage definitions and elementary
  range/kernel facts;
- `Stage9/Support.lean`: exact support identity for an explicit product class;
- `Stage9/Calibration.lean`: scale calibration for a diagonal finite-dimensional
  operator.
- `Stage9/Control.lean`: monotone composition of explicit `rho`/`kappa`, learner
  sensitivity, and representation-residual upper bounds into the non-vacuity
  certificate.

The files currently use explicit Euclidean coordinates where this keeps the
formal proof auditable. The control-composition lemma is formalized, while the
general PSD domination, principal-angle, and shift-generator bounds remain
scientific obligations documented in the Stage 9 notes. No unproved V-REx/MMD/
DRO mapping is formalized.

The checked-in lakefile pins mathlib to the commit used by the local cache
(Lean 4.33.1). No build artifacts or machine-specific paths are committed;
Lake may reuse an existing `.lake/packages` cache or fetch the pinned source
when setting up a fresh checkout.

Run from this directory:

```text
lake build
```
