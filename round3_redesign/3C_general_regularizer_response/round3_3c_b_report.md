# 3C-B Source-Adaptive Affine Response

## Verdict

`3C-B-EXACT-IFT-AFFINE-PASS`

The primary comparison uses exact population source first-order conditions at
the actual regularized solution.  In fixed 3A source-whitened coordinates,
`F(z,y)=0` gives `Pi = -(D_z F)^-1 D_y F` and the local policy is
`z(u)=z0 + Pi O_S u`.  The frozen-`w*` `(b,K)` and quadratic action are audit
quantities only.

## Main audit

- methods: L2, IRMv1, V-REx
- rows: 15; valid rows: 15
- coupled world tangent dimension: 8
- source observation shape: `(275, 8)`
- maximum trained-solution central-difference relative error: `3.69e-10`
- target risk used for fitting or selection: `false`
- semantic labels, clusters, and exposure used for fitting or selection: `false`

The central-difference optimizer audit agrees with the IFT tangent at the
registered steps.  The source state is the stacked task-complete moment state;
the target enters neither the source model fit nor the affine coefficient.

## Boundaries

V-REx is admitted only while the local IFT metric is invertible and positive
definite.  A deliberately over-large lambda is recorded as an invalid path
boundary and excluded from the primary table.  CORAL is retained only as a
representation/gauge boundary and is not a predictor-space affine response.

The same frozen curvature pair `(b,K)` does not determine the adaptive action:
different source-state derivatives can produce different `Pi`.  Therefore
frozen curvature is insufficient to identify source-adaptive response.

This is a population local response result.  It is not a causal mechanism
claim, finite-sample guarantee, target-risk lower bound, or universal DG
theorem.
