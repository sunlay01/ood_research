# TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR Report

## Why the previous audit is invalid

`AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH`: the old A omitted `-D grad R_S`; thresholded finite data differences were not tangents; O included IRMv1 penalty processing; and Pi was evaluated at a newly refined head-only equilibrium.

## Repaired construction

The primary space is exactly R^3 and uses smooth four-outcome empirical expectations. `A = H_S^(-1/2) D[grad R_T - grad R_S]`; `O_S` is method-independent concatenated source risk gradients. Derived common/antisymmetric directions are linearity checks only.

## Gates and verdict

- G0 tangent linearity: `True`
- G1 corrected-A toy identity: `True`
- G2 smooth derivative consistency: `True`
- G3 method-independent O: `True`
- G4 Pi_head diagnostic: `True`
- G5 full response reconstruction/replay: `True`

Verdict: `AOPI-REPAIR-PARTIAL`.

`Pi_head != Pi_full`. The full response is reported in source functional coordinates. A matched full-network mismatch is `NOT_ESTABLISHED`; this repair makes no algorithm, causal, theory-validation, or universal DG claim.
