# Environment-Family Layer Refactor

## Verdict

`ENV-FAMILY-REFACTOR-PASS`

The refactor makes the admissible environment family an explicit upstream
argument:

`EnvironmentFamily -> TangentSpec -> (A_theta, O_S, Pi_theta) -> regret`.

The old eight coordinates are the local coordinate basis of the declared
`legacy_gaussian_mechanism` family.  They are not a universal OOD tangent and
their names are chart labels, not invariant mechanisms.

## Legacy regression

- observation shape: `(275, 8)`;
- response shape: `(9, 8)`;
- `rank(O_S)`: `7`;
- hidden-U information floor: `0.05118145108609`;
- legacy matrix snapshot match: `True`;
- scalar 3C-B table regression: `True`;
- scalar 3E-B table regression: `True`;
- CORAL noncanonical/gauge status: `True`.

The compatibility facade and the shared family geometry produce the frozen
matrix snapshot without changing the old mathematical objects.  V-REx's
large-lambda non-positive local metric remains an invalid path boundary.

## Source-induced family

The second family is built only from observed source task-state contrasts and
an environment-parameter Jacobian pullback.  Its state-span rank is
`4`, while its realizable rank is
`4`.  The source-reference projector error is
`6.06e-15` and the duplicate-source
rank check is `True`.  A new mean
contrast increases raw source-state rank to
`5`; a risk-null noise contrast
may increase raw state rank without adding a response direction.

No target risk, `A`, `Pi`, semantic label, cluster, or downstream regularizer
quantity is read while the source-induced basis is constructed.  The
source-induced metric is explicitly source-state orthonormal and therefore
metric-dependent.

## Cross-family geometry

| family | tangent dim | rank O | rank A | dim ker O | information floor | source-defined | mechanism-defined |
|---|---:|---:|---:|---:|---:|---|---|
| legacy_gaussian_mechanism | 8 | 7 | 5 | 1 | 0.051181451 | False | True |
| source_induced | 4 | 4 | 3 | 0 | 0 | True | False |

A smaller source-induced floor means that the source-induced family promises
to handle a smaller admissible uncertainty set.  It is not evidence that the
learner is universally more robust.  The legacy hidden-U direction is
modeled-but-source-unobservable; a direction absent from the source-induced
tangent would instead be family omission.  In the generated audit this is
recorded as `modeled_in_legacy_family=true`, `legacy_source_null=true`, and
`omitted_from_source_induced_family=true`; omission is not a source-null claim.

## Concrete regularizer comparison

The following table is computed independently on both declared families.  It
is the concrete L2/IRMv1/V-REx comparison, not merely a family-rank summary.
`Pi@O_S` is the affine source-adaptive action and `E` is the recoverable
response residual.  Regret is evaluated in the family-declared world metric.

| family | method | lambda | valid | ||z0|| | ||Pi@O_S||op | ||E||op | affine regret | information floor | method excess | status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| legacy_gaussian_mechanism | L2 | 0 | True | 1.9626156e-17 | 0.22607629 | 1.3973015e-12 | 0.051181451 | 0.051181451 | 0 | PASS |
| legacy_gaussian_mechanism | L2 | 0.01 | True | 0.0029246194 | 0.22410936 | 0.0027539554 | 0.051185728 | 0.051181451 | 4.2768433e-06 | PASS |
| legacy_gaussian_mechanism | L2 | 0.1 | True | 0.028165111 | 0.20801359 | 0.024964178 | 0.051579156 | 0.051181451 | 0.00039770518 | PASS |
| legacy_gaussian_mechanism | IRMV1 | 0 | True | 1.9626156e-17 | 0.22607629 | 1.3973015e-12 | 0.051181451 | 0.051181451 | 0 | PASS |
| legacy_gaussian_mechanism | IRMV1 | 0.01 | True | 3.5824083e-05 | 0.22606563 | 3.4756256e-05 | 0.051181452 | 0.051181451 | 6.4168244e-10 | PASS |
| legacy_gaussian_mechanism | IRMV1 | 0.1 | True | 0.0002970545 | 0.22598243 | 0.00029243278 | 0.051181495 | 0.051181451 | 4.4120722e-08 | PASS |
| legacy_gaussian_mechanism | VREX | 0 | True | 1.9626156e-17 | 0.22607629 | 1.3973015e-12 | 0.051181451 | 0.051181451 | 0 | PASS |
| legacy_gaussian_mechanism | VREX | 0.01 | True | 5.8908241e-06 | 0.22607584 | 3.0811368e-06 | 0.051181451 | 0.051181451 | 1.7350905e-11 | PASS |
| legacy_gaussian_mechanism | VREX | 0.1 | True | 5.889985e-05 | 0.22607182 | 3.0805787e-05 | 0.051181453 | 0.051181451 | 1.7345962e-09 | PASS |
| source_induced | L2 | 0 | True | 1.9626156e-17 | 1.0254527 | 3.0885393e-12 | 2.5511126e-28 | 0 | 2.5511126e-28 | PASS |
| source_induced | L2 | 0.01 | True | 0.0029246194 | 1.0140523 | 0.011621252 | 8.841212e-05 | 0 | 8.841212e-05 | PASS |
| source_induced | L2 | 0.1 | True | 0.028165111 | 0.92179365 | 0.10574382 | 0.0073822776 | 0 | 0.0073822776 | PASS |
| source_induced | IRMV1 | 0 | True | 1.9626156e-17 | 1.0254527 | 3.0885393e-12 | 2.5511126e-28 | 0 | 2.5511126e-28 | PASS |
| source_induced | IRMV1 | 0.01 | True | 3.5824083e-05 | 1.0253403 | 0.00015565831 | 1.7941939e-08 | 0 | 1.7941939e-08 | PASS |
| source_induced | IRMV1 | 0.1 | True | 0.0002970545 | 1.0244253 | 0.001312409 | 1.2605104e-06 | 0 | 1.2605104e-06 | PASS |
| source_induced | VREX | 0 | True | 1.9626156e-17 | 1.0254527 | 3.0885393e-12 | 2.5511126e-28 | 0 | 2.5511126e-28 | PASS |
| source_induced | VREX | 0.01 | True | 5.8908241e-06 | 1.0254473 | 1.4153429e-05 | 1.6115445e-10 | 0 | 1.6115445e-10 | PASS |
| source_induced | VREX | 0.1 | True | 5.889985e-05 | 1.025399 | 0.00014151107 | 1.6109842e-08 | 0 | 1.6109842e-08 | PASS |

The exact vectors/matrices for every row are retained in
`results/method_comparison_by_family.json` and in JSON-valued cells in
`results/method_comparison_by_family_full.csv`.  In particular, those files
contain `z0`, `pi_O`, `E`, and `affine_regret` for all nine rows of the
source-induced family and all nine rows of the legacy family.  The scalar CSV
is retained as a compact compatibility view.

## Failure taxonomy and scope

The three distinct cases are family omission, in-family source-information
failure, and in-family source-visible algorithm failure.  3A supplies the
response norm, 3D supplies source task-state observations, and 3C/3E consume
the resulting family-relative geometry.  3B semantic modules and old 3C
regularizer geometry are boundary/audit inputs, not family-definition inputs.

Coordinate recoding preserves the information geometry only when `A`, `O_S`
and the tangent metric are transported together.  An untransported recoding
changes the uncertainty geometry.  These are population local statements;
they do not claim coverage of all shifts, causal identification, finite-sample
estimability, deep-network identifiability, or a universal DG theorem.
