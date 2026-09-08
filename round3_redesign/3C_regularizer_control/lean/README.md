# 3C Lean Formalization

Pinned to Lean 4.33.1 and mathlib v4.33.1.  The current core formalizes the
finite-dimensional scalar quadratic identities used by the numerical audit:
local minimizer, whitened identity, steering, curvature ratio, exact blind
direction, infinitesimal derivative, scalar PSD monotonicity, and opposite
steering with equal curvature.

The full matrix spectral kernel-limit theorem is intentionally not claimed in
this first formalization pass.

Build with the repository's shared official mathlib cache:

```bash
source ../../../scripts/lean_env.sh
lake exe cache get
lake build OODRegularizer
```
