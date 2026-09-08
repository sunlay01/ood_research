# Lean Formalization

Status: `LEAN-CORE-PASS`.

The project pins Lean `4.33.1` and mathlib `v4.33.1`. Build with:

```bash
source ../../../scripts/lean_env.sh
lake exe cache get
lake build OODRelevance
```

`lake exe cache get` is an optimization. If the remote cache endpoint is
temporarily unavailable, an existing local mathlib build can still be used.

The current checkout has completed `lake build OODRelevance`. The five
theorem modules and aggregate module compile without `sorry`, `admit`, custom
axioms, or placeholder propositions.

The formalized scope is the deterministic quadratic core. General smooth
asymptotics, probability expectations, and singular source geometry are not
claimed as Lean-certified.
