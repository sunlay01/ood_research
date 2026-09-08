# C005: Anchor Regression and DRIG Positive-Control Match

## Status

`PARTIAL / NOT_COMPARABLE_FOR_CURRENT_INTERVENTION_BALLS`

## Anchor Regression

Rothenhausler et al., *Anchor Regression: Heterogeneous Data Meet Causality*,
JRSSB 2021, DOI `10.1111/rssb.12398`, use an SCM with observed exogenous
anchors and additive anchor effects.  Its robustness statement is tied to an
anchor-induced perturbation class, not an arbitrary conditional relation ball
\(\|R-R_0\|_F\le r\).

**Match verdict:**

- `I_corr`: `NOT_COMPARABLE`.  Changing the conditional map \(R\) is not an
  additive exogenous-anchor intervention under the present model.
- `I_mom`: `CONDITIONALLY_COMPARABLE` only after adding an observed exogenous
  environment/anchor variable and restricting the mean shift to the anchor
  span.  The current free Euclidean mean ball is broader than that statement.

Anchor Regression is therefore not yet implemented as a direct positive
control.  It becomes one only in a separately documented anchor-compatible
subclass.

## DRIG

Shen, Buhlmann, and Taeb, *Causality-Oriented Robustness: Exploiting General
Noise Interventions*, arXiv:2307.10299v2 (2025), subsequently JASA 2026,
define Distributional Robustness via Invariant Gradients (DRIG).  The paper's
Theorem 3 identifies the population DRIG solution with worst-case squared-risk
minimization over a source-derived PSD noise second-moment set.  It explicitly
contains Anchor Regression as an additive-mean special case and permits more
general structural noise interventions.

**Match verdict:** `RELATED_NOT_IDENTICAL`.

The current `I_corr` is a Frobenius ball on the conditional nuisance relation,
and `I_mom` independently bounds nuisance mean/covariance.  Neither equals
DRIG's source-derived PSD order set without a further SCM embedding and a
change of uncertainty geometry.  DRIG is a mandatory closest prior work for
the general calibration language, but it does not automatically prove or
refute C002-IRM.

## Required Follow-up

The next literature pass must compare C002-IRM against DRIG's gradient
invariance definition and the IRMv1 failure literature theorem-by-theorem.
No claim that the present framework "recovers" Anchor or DRIG is permitted
until an exact uncertainty-set equality is proved.
