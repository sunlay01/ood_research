# TASK3R Context

Current question:

- Can a source-only, leave-one-source-domain-out pseudo-response objective reduce the frozen response mismatch in an exact Gaussian/quadratic probe?

Frozen inputs needed:

- `T-3A-A`, `T-3C-PI`, `T-3D-IDENTIFIABILITY`, `T-3E-INFO-FLOOR`, `T-3E-SPECTRAL-SLACK`, `T-3E-AFFINE-REGRET`.
- `R-REPAIR-GATE`, `R-TASK1-MECHANISM`, `R-TASK2-SHARP`.

Required equations:

```text
Pi = -(H_R + lambda K)^-1 (B_R + lambda C)
P = P_ker(O_S), A_irr = A P, A_rec = A(I-P), E = A_rec + Pi O_S
```

Known exclusions:

- Do not use target outcomes for objective, beta, checkpoint, or selection.
- Do not claim causal identification, target-risk lower bounds, finite-sample guarantees, or universal DG.
- CMNIST is conditional on the exact Gaussian gate and is not run by default.

Relevant code entrypoints:

- `src/ood_repr_reg/round3r_3b_benchmark.py`
- `src/ood_repr_reg/round3r_3c_benchmark.py`
- `src/ood_repr_reg/sharp_optimality/geometry.py`

Historical reopen: none.
