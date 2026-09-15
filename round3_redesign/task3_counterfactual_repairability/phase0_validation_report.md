# Phase 0 Validation Report

**Scope:** finite-dimensional centered adaptive repair only.  No target risk,
benchmark selection, semantic mechanism label, or static bias `z_0` is used.

The implemented quantity is

`R_adap(Pi) = 1/2 ||A + Pi O_S||_op^2`.

The controller class is a response-space linear class with an optional
operator/Frobenius budget.  Every class optimum retains its controller,
membership residual, solver trace, and spectral-slack certificate.

## Checks

| Check | Status |
| --- | --- |
| `toy_head_full` | PASS |
| `toy_information_limited` | PASS |
| `toy_spectral_slack` | PASS |
| `nested_monotonicity` | PASS |
| `coordinate_invariance` | PASS |
| `solver_grid` | PASS |

Overall Phase-0 status: **PASS**.

## Toy values

| Quantity | Value |
| --- | ---: |
| Toy 1 `R_info` | `0` |
| Toy 1 `R_head*` | `0.5` |
| Toy 1 `R_full*` | `0` |
| Toy 1 `R_actual` | `0.25` |
| Toy 1 value of representation-changing intervention | `0.5` |
| Toy 1 realization gap | `0.25` |
| Toy 2 `R_info = R_class*` | `1` |
| Toy 3 `||E||_op` | `1` |

Toy 3 has nonzero `E` and still attains the information floor.  This is an
explicit regression against restoring `E = 0` as a necessity.

## Independent checks

- `pytest -q tests/test_counterfactual_*.py` covers the controller interface,
  membership bridge, locality convention, nested classes, transport, and grid
  audit.
- `formalization/OODRelevance/RestrictedRepair.lean` was checked with the
  locally cached Lean/mathlib environment.  Its scope is the abstract class
  optimality and zero-observation identities; it does not formalize numerical
  optimization.
- Existing frozen tests remain a separate regression gate and are not folded
  into this Phase-0 scientific status.

## Interpretation boundary

This report establishes only that the proposed centered repairability objects
are internally consistent on declared finite-dimensional classes.  It does not
show that a class is a faithful model of any training algorithm, that a local
response predicts finite updates, or that response regret is target OOD risk.
Those are later gates and must not be inferred from this report.
