# Theory

Let `E_S=span{psi_e-psi_0}` and `A` be the linear response map above.
The design-exposed response space is `R_exp=A(E_S)`. Its dimension is the
rank of the matrix of response images of source contrasts.

For any larger response space `R`, the unexposed quotient dimension is
`dim(R)-dim(R_exp)` when `R_exp` is contained in `R`. This is an exact linear
algebra statement. Exposure fractions based on Euclidean projection are
metric-dependent diagnostics.

Changing the reference source replaces each contrast by a linear combination
of the old contrasts. Therefore the span and its response image are invariant.
Duplicate source rows add no direction. A risk-null state perturbation can
increase state rank while its response image remains zero. A new source
contrast raises exposed rank exactly when its image lies outside the prior
response image.
