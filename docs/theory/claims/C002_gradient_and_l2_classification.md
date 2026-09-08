# C002-G and C002-L2: Classification in the Scalar SCM

## Status

`PROVED_INTERNAL / BASELINE_CLASSIFICATION`

## Full Gradient Matching

Consider two zero-mean scalar correlation environments with distinct relations
\(r_1\ne r_2\).  For \(w=(u,a)\), the difference of population squared-risk
gradients is

\[
\nabla R_{r_1}(w)-\nabla R_{r_2}(w)
=2(r_1-r_2)
\begin{pmatrix}
\ell a\\
\ell u-\beta+(r_1+r_2)a
\end{pmatrix}.
\]

Consequently, if \(\ell\ne0\), full shared-head gradient matching is zero
if and only if

\[
a=0,\qquad u=\beta/\ell.
\]

It is therefore selective in this narrow population model: its zero set is
nuisance-free.  But this does **not** establish the README calibration
definition.  The zero point need not be source-observationally optimal.

For the C002-IRM parameters, the gradient-invariant point is \((u,a)=(1,0)\)
with source risk \(2.01\), while the source observational oracle risk is
\(0.51\).  By continuity and uniqueness of source ERM, any sequence whose
source excess tends to zero has gradient-matching penalty tending to the
strictly positive value at ERM (`0.968888...`).  The interface
"source excess -> 0 and gradient penalty -> 0" is empty in this model.

**Classification:** `SOURCE-FIT / INVARIANCE INCOMPATIBILITY`, not a positive
calibration theorem and not a gradient-alignment failure counterexample.

## L2

Population ridge has the exact form

\[
w_\lambda=(\Sigma_S+\lambda\operatorname{diag}(0,I))^{-1}c_S.
\]

It always supplies the coarse inequality
\(\|w_A\|\le\|w_{\lambda,\mathrm{nonintercept}}\|\), but it cannot
separate nuisance shrinkage from task-observation shrinkage.  In the C002-IRM
parameters, increasing \(\lambda\) reduces \(|w_A|\) and correlation-ball
robust excess, while raising source risk toward the constant-predictor risk.

| \(\lambda\) | source risk | \(|w_A|\) | correlation-ball causal excess |
| ---: | ---: | ---: | ---: |
| 0 | 0.5100 | 0.8333 | 2.6458 |
| 0.1 | 0.5237 | 0.5960 | 1.9193 |
| 1 | 0.6310 | 0.1804 | 1.0034 |
| 100 | 0.9891 | 0.0030 | 0.9868 |

**Classification:** `NONSELECTIVE SHRINKAGE TRADEOFF`.  L2 can enter a
conditional norm-based bound, but its value is not evidence that it isolates
harmful nuisance directions.

## Scope

These are population statements for the identity representation and squared
loss.  They do not generalize automatically to learned nonlinear encoders.
