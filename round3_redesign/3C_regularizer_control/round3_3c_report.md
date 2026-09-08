# 3C Regularizer Control Report

## Verdict

`3C-ACTION-PASS-COVERAGE-PARTIAL`

The analysis compares L2, CORAL, IRMv1 and V-REx through the chain
`constraint object -> (b, K) -> OOD-response coverage`.  The primary objects
are individual source-whitened response vectors.  The six 3B clusters and
oracle mechanism labels are excluded from primary calculations.

## Benchmark

- Relevant shifts: `33` / `35`.
- Response rank: `5`.
- Derivative audits passed: `True`.
- 3B semantic clusters used: `False`.

## Method status

| Method | Predictor status | Gauge status | Relevant restricted rank | Relevant kernel dimension | PSD |
|---|---|---|---:|---:|---|
| L2 | predictor-intrinsic | not-applicable | 5 | 0 | True |
| CORAL | representation-level fixed-gauge conditional | gauge-dependent/induced-degenerate | 0 | 5 | True |
| IRMV1 | predictor-intrinsic | not-applicable | 3 | 2 | True |
| VREX | predictor-intrinsic | not-applicable | 3 | 2 | False |

## Interpretation

The local-action identities are population quadratic results.  Curvature
ratios, steering signs, blind fractions and nonselective curvature are
response-level diagnostics.  CORAL is explicitly not predictor-intrinsic on
the unconstrained representation fiber: the gauge audit records the exact
`c^4` scaling and zero induced infimum.

The verdict is partial because the local geometry is auditable, but this
benchmark does not justify a universal cross-method OOD-control theorem.  No
3B cluster is interpreted as a semantic or causal mechanism.

## Boundaries

The results are population calculations, not finite-sample guarantees, deep
network theorems, or source-identifiability results.  Exposure annotations are
post-hoc only.
