# Intrinsic Geometry

The primary response for shift `s` is

\[
u_s=H_S^{-1/2}g_s,
\qquad G_{ij}=u_i^\top u_j=g_i^\top H_S^{-1}g_j.
\]

The implementation uses one eigendecomposition of the positive-definite
source Hessian and reports the Gram matrix, numerical rank, and SVD bases.
All rank statements are tolerance-dependent numerical statements; they are
not causal identifiability claims.

A module is represented by the span of the retained response columns assigned
to it.  Internal SVD rank is reported rather than forced to one.  A minimum
norm least-squares reconstruction is used for mixed shifts.  It is unique
only when the module images form a direct sum.

The predictor reparameterization audit uses
\(H'=T^{-\top}HT^{-1}\) and \(g'=T^{-\top}g\).  It checks the intrinsic Gram
invariant, not coordinate singular vectors.
