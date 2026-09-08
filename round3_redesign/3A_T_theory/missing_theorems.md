# Missing Theorems Closed in 3A-T

The completed 3A report had the exact quadratic identities but did not
record the following results as standalone statements.

## Independent nuisance coordinates

For block-diagonal source geometry and zero nuisance residual gradient,
block inversion gives
\[
M_{aug}^{-1}=\operatorname{diag}(M_E^{-1},M_N^{-1}),
\]
so the nuisance block contributes exactly zero to the dual quadratic form.
The conclusion concerns only the leading support term. A nonzero nuisance
block in the target quadratic matrix can still change `K_s` and hence finite
budget vulnerability.

## Metric coordinate invariance

Using `(T^{-top} M T^{-1})^{-1}=T M^{-1} T^top`, direct multiplication gives
\[
(T^{-top}g)^T(T M^{-1}T^T)(T^{-top}g)=g^TM^{-1}g.
\]
This is an intrinsic scalar statement, not an invariance claim about a chosen
coordinate basis.

## Common burden

Changing only `c_e` changes every model's target loss by the same amount. It
can produce arbitrarily large burden with zero discrimination inside every
source-risk neighborhood.

## Smooth extension

Positive definiteness of the source Hessian, together with localization of the
global source sublevel set at `w*`, implies local constants `a,b>0`
for which
\[
a\delta^TH_S\delta\le R_S(w^*+\delta)-R_S(w^*)
\le b\delta^TH_S\delta.
\]
Differentiability of the shift controls the upper bound; a scaled dual-norm
direction, reduced by a factor `1-o(1)`, supplies the lower bound. A bounded
local shift Hessian gives the `O(epsilon)` null-gradient result.
