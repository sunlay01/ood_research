# Formalization Report

## Status

`LEAN-CORE-PASS`

Lean `4.33.1` and mathlib `v4.33.1` are pinned. The generated
`lake-manifest.json` records mathlib and all eight transitive packages by
exact Git commit. The final command

```bash
lake build OODRelevance
```

completed successfully and produced six project `.olean` files: the five
theorem modules and their aggregate import.

## Formalized deterministic core

`QuadraticRisk.lean` proves the exact source-excess identity for a symmetric
bilinear form satisfying the source normal equation, and the exact shift
polynomial identity for two quadratic risks.

`EllipsoidSupport.lean` proves the Euclidean support upper bound after source
metric whitening, constructs an equality witness including the zero cases,
and packages the result as an `IsGreatest` maximum statement.

`RelevanceBound.lean` proves exact first-order-null quadratic scaling under
`delta = sqrt(epsilon) u` for nonnegative `epsilon`.

`NuisanceInvariance.lean` represents the inverse source metric by its dual
bilinear form and proves, in arbitrary product dimensions, that a zero
nuisance gradient contributes zero to the block direct-sum dual value.

`Counterexamples.lean` proves that a target-only constant burden is model
independent. For the two-coordinate source-usage counterexample it proves the
exact source excess formula, global source optimality, zero second source
coordinate, the target gradient `(0,-1)`, target positive definiteness, and
the complete local shift-response polynomial.

## Integrity audit

The retained `.lean` sources contain no `sorry`, `admit`, custom `axiom`, or
`True` placeholder. Each module compiles independently and through the
aggregate target. The formal scope is the deterministic quadratic core; the
probability expectation layer, general smooth theorem, matrix square-root
construction, and singular-Hessian boundary remain paper proofs or boundary
analysis and are not claimed as Lean-certified.

## Dependency repair

The initial Git/HTTP failures left an invalid `aesop` checkout and forced a
partial source build of mathlib. The repair restored the exact manifest
commits, reused the completed `.olean` cache, replaced temporary local path
dependencies with Git-pinned dependencies, narrowed project imports, and
successfully compiled the project modules. The official binary-cache endpoint
was unavailable during the final retry, but the local compiled dependency
cache was sufficient. This distinguishes mathlib cache completion from actual
theorem compilation.
