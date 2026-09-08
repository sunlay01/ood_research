# 3E Optimal Source-Only OOD Response Recovery

## Verdict

`3E-SHARP-MINIMAX-RECOVERY-PASS`

The primary theorem is an exact finite-dimensional population information
result.  For arbitrary deterministic source-only recovery rules,

`inf_Phi sup_{||u||<=1} ||A u - Phi(O u)|| = ||A P_ker(O)||_op`.

The lower bound uses the indistinguishable pair `v,-v` in `ker(O)`.  The
pseudoinverse rule `Phi*(y)=A O^dagger y` attains the bound.  This is a
response-recovery theorem, not a target-risk lower bound.

## Main audit

| setting | dim U | rank O | dim ker O | rank A | d_struct | alpha | normalized alpha | ambiguity diameter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coupled 3A/3D world tangent | 8 | 7 | 1 | 5 | 1 | 0.31994203 | 1 | 0.63988406 |

The eight world coordinates are standardized `S1/S2` relation, mean, and
variance perturbations, independent-noise variance, and target-emergent `U`
coupling.  Their declared magnitude-one scales are recorded in
`results/summary.json`.  The `U` tangent is source-null
(`0`) but response-active
(`0.32`) in the primary hidden-emergent
design.  This is an explicitly metric-conditional local model, not a
canonical causal parameterization.

The recoverable/irreducible operator split is `A O^dagger O` plus
`A P_ker(O)`.  The response images of these two domain components need not be
orthogonal.  The value depends on the declared world metric; invertible source
coordinate recodings and world/response isometries preserve it when the
metric is treated consistently.  Arbitrary world reparameterizations do not
preserve alpha without changing the declared metric.

## 3D recovery

The identifiable 3D family has zero minimax error.  The hidden-emergent family
has positive minimax error.  The finite source-identical/target-different pair
is independently checked.  It reaches `2 alpha` only under its explicitly
declared one-dimensional embedding, not as a metric-free extremality claim for
the raw Gaussian pair.

## Information and counterexamples

Adding source information cannot increase alpha, but can leave the worst
direction unchanged: duplicate and independent-noise additions are controls.
The separately recorded U-exposed source observation removes the hidden worst
direction.  Two matched two-source stacks have different alpha solely because
their observation geometry differs.  Source rank, source count, and structural
ambiguity dimension alone do not determine recovery difficulty; the orientation
of `ker(O)` relative to `A` does.  The abstract ladder remains a theorem
fixture, not the primary benchmark.

## Boundaries

3A supplies the source-whitened response norm and vulnerability interpretation;
3D supplies task-complete source observations and the zero-error endpoint.  3B
and 3C are exclusions, not theorem inputs.  No 3B semantic labels, 3C
regularizer geometry, target-risk oracle, finite sample claim, causal
identification claim, or universal DG lower bound is used.  The nonlinear
experiment is a local tangent diagnostic only.
