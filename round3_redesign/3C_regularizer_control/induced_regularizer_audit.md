# Induced Regularizer Audit

The predictor map is intrinsic for L2, V-REx and scalar-scale IRMv1 in this
linear predictor benchmark.  CORAL is representation-level: \(A\mapsto cA\),
\(v\mapsto v/c\) preserves the predictor while the covariance penalty scales
as \(c^4\).  Therefore its unconstrained fiber infimum is zero.

The fixed-representation CORAL rows are retained as an explicit null-action
baseline.  Any gauge-fixed representation conclusion is conditional on that
gauge and is not merged into predictor-intrinsic curvature results.
