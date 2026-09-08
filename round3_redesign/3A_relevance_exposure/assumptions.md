# Assumptions

This track uses population linear-Gaussian regression with squared loss. The
outcome law is fixed; environments change feature couplings or feature
marginals. The source objective is the mixture of declared source
environments. Target environments are evaluated offline only.

The benchmark contains a noisy predictive coordinate, a source-active
shortcut, optional target-emergent coordinates, a target-stable predictive
coordinate, and independent noise coordinates. Relevance is model
discrimination inside a source-risk ellipsoid, not feature usage and not a
causal claim.

# Scope

The track does not perform clustering, mechanism naming, response-order
analysis, or regularizer analysis. Those questions remain deferred.
