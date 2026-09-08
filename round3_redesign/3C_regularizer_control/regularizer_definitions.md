# Regularizer Definitions

## L2

\(\Omega(w)=\frac12\|w\|^2\), with \(a=w^*\) and \(J=I\).  This is a
predictor-intrinsic parameter geometry but is nonselective with respect to
OOD response directions.

## V-REx

\(\Omega(w)=\operatorname{Var}_e(R_e(w))\), using population source
environments with equal weights.  The implementation differentiates the exact
quadratic per-environment risks.

## IRMv1

\(\Omega(w)=m^{-1}\sum_e[\partial_\alpha R_e(\alpha w)|_{\alpha=1}]^2\).
This is the scalar-scale IRMv1 objective, not a generic gradient-alignment
substitute.

## CORAL

CORAL directly constrains representation covariance differences.  With a fixed
representation it has zero predictor-level gradient and Hessian.  With a
trainable factorization it is gauge-dependent and its unconstrained induced
cost is degenerate under representation rescaling.
