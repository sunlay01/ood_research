# Unified Lean Formalization

This is the single reusable Lean workspace for the project's formal proofs.
It combines the existing `OODRelevance` and `OODRegularizer` libraries while
leaving their original round directories unchanged.

The workspace is pinned to Lean 4.33.1 and mathlib v4.33.1. The mathlib cache
is shared through `MATHLIB_CACHE_DIR`; the dependency checkout is reused from
the already verified 3A-T workspace to avoid another multi-gigabyte source
copy. Future formalization modules should be added here and built together.

```bash
source ../scripts/lean_env.sh
lake build
```

This workspace contains only deterministic finite-dimensional/algebraic
formalization. It does not claim to formalize probability, general smooth
asymptotics, finite-sample ERM, or causal mechanism discovery.
