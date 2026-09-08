# 3D Source Exposure and Response Identifiability

## Verdict

`3D-DESIGN-EXPOSURE-PASS-STRUCTURAL-PARTIAL`

The primary observable is the task-complete source state
`psi_e=(svec(M_e), m_e, c_e)`.  Exposure is the image of source contrasts
under the frozen source-response operator.  Target risk, mechanism labels,
3B clusters and 3C regularizers are excluded from this construction.

## Exposure geometry

- Source state dimension: `55`.
- Main source contrast rank: `4`.
- Exposed response rank: `3`.
- Total frozen response rank: `5`.
- Unexposed response quotient dimension: `2`.
- Source-reference invariant: `True`.

These are finite-dimensional population linear-algebra results.  The exposure
fraction is metric-dependent; rank, kernel and quotient statements are the
intrinsic statements.  Repeated source rows do not add a contrast direction.
Risk-null nuisance changes can increase raw state diversity without increasing
the response image.  A new relevant source contrast can increase exposed rank.

## Structural identifiability

The identifiable linear world family satisfies the kernel inclusion criterion
and therefore factors through source observations.  The hidden-emergent family
has nonzero structural ambiguity: two worlds share source observations while
their target responses differ.  The exact Gaussian pair is recorded in
`exact_world_counterexample.json`.

## 3C intersection

The regularizer table is post-hoc only.  It does not define source exposure,
choose source designs, or select rank.  Its CORAL row retains the fixed
representation/gauge caveat inherited from 3C.

The coordinate audit preserves the response Gram to numerical precision, and
the nonlinear moment family is recorded as a diagnostic only.

## Boundaries

This does not establish finite-sample estimability, causal mechanism
identification, deep-network identifiability, or a universal DG theorem.
Mechanism interpretation and any 3E risk decomposition remain out of scope.
