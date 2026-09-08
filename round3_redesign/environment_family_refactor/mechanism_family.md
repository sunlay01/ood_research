# Legacy mechanism wrapper

`MechanismFamily` is a lightweight metadata wrapper around the legacy Gaussian
family.  It records mutable shortcut/noise/emergent mechanism names while
delegating all population perturbations to the legacy implementation.  The
names are descriptive metadata and do not create a causal identification
result.  The tangent vectors remain chart-dependent coordinates.
