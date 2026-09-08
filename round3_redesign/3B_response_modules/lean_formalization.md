# Lean Formalization

The 3B extension is isolated in the existing pinned Lean 4.33.1/mathlib
v4.33.1 project at `round3_redesign/3A_T_theory/lean`.

Formalized statements:

- `direct_sum_two_unique`: uniqueness under the explicit two-subspace zero-sum
  condition;
- `overlap_nonidentifiability`: a nonzero overlap vector has two distinct
  module assignments;
- `response_additive` and `response_three_additive` for a linear response map;
- `gram_invariant_under_isometry` and normalized Gram invariance under an
  inner-product and norm preserving linear map.

These are abstract deterministic theorems.  The Gaussian benchmark, numerical
rank tolerance, bootstrap stability, and semantic ARI/NMI are not Lean
theorems.  The build was run with the absolute `lake` executable because this
shell does not put `~/.elan/bin` on PATH.
