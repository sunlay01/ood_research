# Environment-Family Layer assumptions

The admissible local environment family is an explicit input to the Round 3
geometry.  A family supplies a reference environment, a finite tangent chart,
a positive-definite tangent metric, and legal source/target perturbations.

The legacy family is the Gaussian/mechanism-family instantiation used by the
frozen 3A--3E benchmark.  Its eight directions and standardized scales are a
coordinate chart, not a universal list of all OOD shifts.  The source-induced
family is built from observed source task-state contrasts and uses a
source-state-orthonormal metric.  This metric choice is explicit and
metric-dependent.

The source-induced basis construction has no access to target risk, response
maps, regularizer geometry, clusters, or semantic labels.  These quantities
may be evaluated later as separate audits, but never define the family.

All conclusions are population, finite-dimensional and local.  They do not
claim causal identification, finite-sample estimability, deep-model coverage,
or a universal domain-generalization theorem.
