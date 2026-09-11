# TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL Context

This task extends the already modular CMNIST A/O/Pi survey into a common-budget spectral and flatness panel. It keeps the same CMNIST data, model, seeds, optimizer base, source schedule, R5 smooth world, A/O definitions, functional banks, normalization, and blind grouping semantics.

Common-harness scope:

- Paper references define algorithmic updates; paper-specific architecture, batch size, horizon, or benchmark accuracy are not imported.
- Existing methods `ERM`, `IRMv1`, `VREX`, `CORAL`, `FISHR`, and `MLDG` remain behavior-preserving.
- `WEIGHT_NUCLEAR` and `FEATURE_NUCLEAR` are preserved as legacy-only code paths and are not default primary methods.

Runnable additions:

- Spectral family: `SPECTRAL_NORM_REG`, `SPECTRAL_REG_2024`, `SVB_ORTHDNN`, `STABLE_RANK_NORM`.
- Flatness family: `SAM`, `ASAM`.

Deferred candidates:

- `SVD_SPARSE` is deferred because faithful singular-vector parameterization/sparsification would change the fixed model parameterization.
- `FAD` and `DISAM` are deferred until exact update semantics are implemented; no SAM surrogate is allowed.

Diagnostics:

- Weight, representation, gradient spectrum, compute-budget, and source flatness diagnostics are auxiliary tables only.
- These diagnostics do not enter `O` and do not justify causal claims.

Canonical state remains unchanged; completion writes only `active/STATE_DELTA.md`.
