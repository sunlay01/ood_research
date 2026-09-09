# CMNIST Environment Families

The data generator remains the existing binary task `Y=1[digit>=5]` and the
existing red/green color construction.  The bridge uses source correlations
`(0.9, 0.8)` and target correlation `0.1`.

`declared_source_target_coupled_correlation` has two source-correlation
tangent coordinates.  This is a declared coupled family, not a source-only
inference of target behavior.  The hidden mechanism family adds
`rho_hidden`, which affects the target correlation but not source states in
its hidden design.  The exposed design makes the same coordinate alter both
source and target correlations.  `brightness_nuisance` is an irrelevant
source-diversity control implemented as a source-only feature scale
perturbation.

No tangent direction is manually inserted into an operator.  Every column is
obtained by central differences through the declared family map.
