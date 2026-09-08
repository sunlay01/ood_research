# 3B Assumptions

This track studies only differential order in the 3A population quadratic-risk response space.

- The model ensemble is the 660-model ensemble already trained for the retry experiment.
- A structural path is affine: `eta(t) = eta_0 + t v`.
- The structural covariance matrices are required to remain positive definite at every evaluated
  point. No eigenvalue-floor projection is used on this path, because projection would introduce
  nonsmooth, non-polynomial behavior.
- The predictor is affine in `X=(1,C,A)` and risk is exact population squared loss.
- The analytic path is used for rank claims; finite differences are validation only.
- `core`, `nuisance`, and `relation` remain bookkeeping names inherited from the registered
  parameter blocks. They are not semantic conclusions in this track (`DEFER-TO-3C`).
- Regularizer interpretation is outside scope (`DEFER-TO-LATER`).

