# LATENT-001 Code-Gate Stop Report

Date: 2026-09-05

## Decision

`LATENT-001` stopped before the ten-seed run. The independent Supervisor issued
`REVISE_ONCE` at the first `CODE_GATE` and `VETO` at the second. The frozen
protocol forbids a third repair or continuation after this repeated failure.

Audit evidence:

- `docs/audits/LATENT-001-CODE_GATE-20260905T080917Z.json`
- `docs/audits/LATENT-001-CODE_GATE-20260905T081824Z.json`

## Blocking Defect

The contract requires each method's lambda to be selected solely by source
risk plus its source objective. The runner enumerates the configured strength
grid but has no explicit source-only selection function or tests proving that
target risk, target moments, target labels, target-derived semantics, and bound
tightness cannot enter selection.

The Supervisor passed the objective definitions, source/target data separation,
paired ERM execution, and target-independent intervention budgets. Those checks
do not repair the missing selection interface.

## MVP Result That Remains Valid

The MVP and repaired MVP artifacts are retained as falsification evidence, not
as a seven-method result. Their projectors satisfy symmetry, idempotence,
pairwise orthogonality, and support completeness to numerical precision, and
the component risk accounting closes. However, the maximum change under legal
semantic residualization orders is approximately `0.4116`, above the frozen
`0.25` threshold. Therefore the correct status is:

```text
SEMANTIC_DECOMPOSITION_NOT_IDENTIFIED
```

Algebraic orthogonality is not sufficient for semantic identification. In this
SCM and estimator, the raw task/relation/mean/covariance ranges overlap enough
that hierarchy choice materially changes the resulting subspaces.

## Claims Not Authorized

- No ten-seed comparison was run.
- No conclusion is available for what each of the seven regularizers can OOD.
- No method-specific latent suppression or failure matrix is available.
- No Omega-to-component generalization bound was established.
- MVP target rows cannot be used to select lambda or promote correlations to
  control statements.

## Requirements For A New Registered Attempt

1. Define an explicit source-only lambda-selection functional before targets are
   instantiated or evaluated.
2. Test that the selection function cannot access any target-derived object.
3. Decide whether semantic order dependence is itself the target phenomenon or
   add assumptions/estimators that identify the raw semantic ranges without an
   arbitrary hierarchy.
4. Register a new experiment ID and reset the gate protocol explicitly; do not
   overwrite this stopped run.
