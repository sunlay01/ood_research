# Theory to objective

## Frozen chain

The theorem objects are `A_rec`, `O_S`, and the actual source-adaptive response `Pi`. Their recoverable mismatch is `E = A_rec + Pi O_S`; the irreducible part `A_irr` belongs only to the information floor. The learner cannot set `Pi` freely.

## Candidate objectives

1. `C1` Frobenius matching, `R_S + beta ||E||_F^2`, is differentiable and is the selected controlled surrogate. It is faithful to the operator theorem only when Frobenius and worst-direction rankings agree, or as a conservative diagnostic; it is not the spectral theorem itself.
2. `C2` `R_S + beta ||E||_op^2` is closer to the theorem but nonsmooth at singular-value ties and expensive for a learned representation.
3. `C3` penalizing `[lambda_max(EE^T-S)]_+` is closest to the sharp adaptive condition, but requires a family response and slack estimate. Those quantities are not generally source-identifiable, so it is not the Phase-I learner objective.
4. `C4` adds `||z0||^2`; this is needed for full affine regret but is zero by construction in the centered prototype. It remains an audit term, not a hidden optimization target.
5. `C5` uses a source domain as pseudo-target and minimizes its local transfer loss plus response mismatch. This is the selected objective because every input is source-observable and it induces `Pi` through an exact quadratic head solution.

## Free-Pi sanity check

For fixed `O`, `A_rec`, the Frobenius problem `min_Pi ||A_rec + Pi O||_F^2` has the minimum-norm solution `Pi_0=-A_rec O^dagger`. The full minimizer set is `Pi_0 + N(I-O O^dagger)` for arbitrary `N` (equivalently arbitrary action on the orthogonal complement of `im(O)` in source-state space). Exact cancellation is possible iff `A_rec(I-O^dagger O)=0`. For the frozen decomposition this condition holds by construction because `I-P_ker(O)=O^dagger O`. Nullspace directions of `O` are never cancelable by `Pi O`, and are represented by `A_irr`/the information floor. A real optimizer only changes source loss and regularizer parameters, so its `Pi(theta)` is constrained by the Hessian and forcing blocks, rather than freely selecting this solution.

## Implemented reduction

For each leave-one-source-domain-out fold, `w_ref` and `y_ref` are the exact meta-source optimum and task state. The regularizer is

```text
Omega(w,y) = 1/2 (w-w_ref)^T K (w-w_ref) + (w-w_ref)^T C(y-y_ref)
K = L L^T, C = U v^T
```

Thus its exact first-order response is induced by

```text
Pi = -(H_R + lambda K)^-1 (B_R + lambda C).
```

The response target is a finite source-only optimum displacement from `w_ref` to the held-out source domain. The fit uses pseudo-target source risk plus directional response mismatch. A centered fold gives `z0=0` at the meta-source reference; it does not remove the full-affine static theorem, it makes its audit tax exactly zero for this prototype.
