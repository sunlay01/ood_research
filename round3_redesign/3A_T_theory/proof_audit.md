# 3A-T Proof Audit

## T1: source ellipsoid — PASS

The algebra needs only `M_S w*=m_S`. Positive definiteness is needed to call
the set an ellipsoid and to obtain boundedness. The attempted failure mode is
singular `M_S`; it invalidates boundedness but not the identity.

## T2: shift polynomial — PASS

The proof needs finite quadratic moments and uses the symmetric part of
`A_s`. In the moment model `A_s` is symmetric automatically. An arbitrary
non-symmetric matrix would not change `delta^T A_s delta` after symmetrizing.

## T3: ellipsoid support — PASS-WITH-EXPLICIT-BOUNDARY

`M` must be symmetric positive definite. The supremum is finite and attained
because the ellipsoid is compact. At `epsilon=0` or `g=0`, the value is zero;
the nonzero witness formula is intentionally not used in those cases.

## T4: finite-epsilon bound — PASS

`K_s` is finite by compactness, or equals the operator norm of the whitened
symmetric matrix. The lower bound uses the linear maximizing witness and
`|a+b| >= |a|-|b|`; it does not assume the quadratic term has a favorable sign.

## T5: first-order-null scaling — PASS

Homogeneity under `delta=sqrt(epsilon)u` gives exact scaling, including
`epsilon=0`. `g_s=0` is not overinterpreted: `K_s` may be positive.

## T6: residual coupling — PASS

The same source optimizer `w*` is used under both environments. Finite first
and second moments are required. The source residual moment vanishes by the
normal equation; no causal or mechanism assumption is used.

## Proposition 7 — PASS AS MOMENT COUNTEREXAMPLE

The displayed source and target matrices are positive definite and hence are
valid quadratic moment pairs. They are also realizable by finite Gaussian
vectors: take centered `X` with covariance `M_e`, and set
`Y = m_e^T M_e^{-1} X + eta` with independent zero-mean noise. The explained
second moments are `1/2` for the source and `2/3` for the target, so the
displayed `c_e=1` values leave nonnegative residual noise variances. Thus this
particular counterexample is a genuine finite-second-moment Gaussian
construction, not merely a formal moment assignment. The relevance
conclusion depends only on `(M,m)` in any case.

## Proposition 8 — PASS

Changing only `c_T` adds the same constant to every target risk, so burden is
nonzero while all model differences vanish.

## Theorem 9 — PASS

Block diagonal positive-definite matrices invert blockwise. The conclusion is
only leading relevance invariance. A target `A_s` nuisance block can make
finite-epsilon vulnerability change.

## Corollary 10 — PASS

Invertibility of `T` is essential. Direct substitution proves the scalar
invariance; no claim is made about basis-dependent vectors.

## Theorem 11 — PASS-WITH-STRONGER-ASSUMPTION

`C^2` source regularity, zero source gradient, positive-definite Hessian,
shift differentiability, and localization of the global source sublevel sets
at `w*` suffice for the `o(sqrt(epsilon))` result. Without localization a
distant source-good component is a counterexample to the global statement.
The proof must include the local sandwich and a scaled feasible witness.

## Theorem 12 — PASS-WITH-STRONGER-ASSUMPTION

The `O(epsilon)` remainder needs local quadratic control of the shift and the
stated cubic source remainder. `C^2` source regularity alone is insufficient.

## Global audit result

No central quadratic identity failed. The main correction is terminological:
the smooth `O(epsilon)` claim is conditional, and the original displayed
two-dimensional counterexample is a moment-level construction unless `c_T`
is enlarged to make a probability realization explicit.
