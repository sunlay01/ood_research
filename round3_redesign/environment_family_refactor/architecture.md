# Architecture

The new dependency direction is:

`EnvironmentFamily -> TangentSpec -> FamilyTaskGeometry -> (A, O_S) -> downstream audits`.

`environment_family.base` defines the protocol and immutable metadata.
`legacy_gaussian` preserves the frozen eight-coordinate Gaussian benchmark.
`source_induced` constructs a source-only state-span family with a parameter
Jacobian pullback.  `metrics` handles tangent metrics and coordinate transport.
`geometry` builds the task-complete source observation and frozen 3A response
Jacobian.  `invariance_audit` reports reference and coordinate checks.

`round3r_3e_world_tangent.py` is a compatibility facade.  Existing imports of
`WorldTangentSpec` and `coupled_primary_geometry()` remain valid, while the
implementation delegates to `LegacyGaussianFamily` and `build_task_geometry`.
The 3C-B benchmark accepts a family and therefore can reuse the same geometry.

No old 3A--3E artifact is replaced.  The refactor runner compares the new
family path with frozen scalar and matrix compatibility facts before reporting
the cross-family table.
