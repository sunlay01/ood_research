# C004a: Source-Unobservable Nuisance Direction

## Status

`PROVED_INTERNAL / NUMERICALLY_VERIFIED`

## Statement

Multiple source environments do not imply that every nuisance direction is
observable.  There can be source variation in one nuisance direction while a
different, source-predictive nuisance direction remains completely absent from
the source response operator and changes at target time.

## Construction

Use the C001 model with two-dimensional \(C,U,A\),

\[
L=I_2,\quad \beta=(1,1)^\top,\quad
\Sigma_\xi=2I_2,\quad \Sigma_A=\tfrac1{50}I_2,
\]

and sources

\[
R_1=\operatorname{diag}(0.7,0.4),\qquad
R_2=\operatorname{diag}(-0.1,0.4).
\]

Let \(H_S\) stack centered source differences of the augmented observed
feature covariance.  The second nuisance column of \(H_S\) is identically
zero: environments vary only the independent \((C_1,U_1,A_1)\) block.  Hence
both the direct and task-adjusted smallest singular values of the nuisance
response are zero.

Nevertheless, population source ERM is

\[
w_{\rm ERM}=(0,\;1/4,\;1/19,\;5/6,\;40/19),
\]

so it uses the second nuisance coordinate.  The two admissible same-task
targets

\[
R_T^+=\operatorname{diag}(-0.1,0.4),\qquad
R_T^-=\operatorname{diag}(-0.1,-1)
\]

have identical source observations and task mechanism, but the same source
ERM has risks approximately \(0.949\) and \(10.256\), respectively.

## Consequence

No source statistic whose information about nuisance variation factors only
through this source response operator can certify robustness to the second
target direction.  This is not a claim about all source-only regularizers:
one that is directly given the identity of the second nuisance coordinate can
penalize it explicitly.

## Verification

The zero singular values are asserted in
`tests/test_intervention_linear.py`.  The numerical values are population
quantities, not sampled estimates.
