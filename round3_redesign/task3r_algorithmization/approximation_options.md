# Approximation options

| option | fidelity | cost | status |
|---|---|---|---|
| O1 exact linear-head prototype | exact source moments, Hessian, IFT | low in Phase I | implemented |
| O2 HVP / implicit gradient | potentially exact in a differentiable model | medium/high | documented alternative |
| O3 Fish/MLDG first-order proxy | cheaper, but drops exact response forcing/filtering | low | documented alternative |
| O4 moment/closed-form surrogate | exact under matching quadratic assumptions | low | the Phase-I moments instantiate this option |

The response penalty is not asserted to be new. In a linear quadratic model, gradient inner-product, gradient variance, Hessian alignment, and one-step meta objectives can reduce to overlapping quadratic forms in source moments. The required label is `SPECIAL-CASE` for this setting, `STRICTLY-DIFFERENT` only for the constrained `K,C` response realization, and `UNRESOLVED` for nonlinear representation learning. No novelty claim is made.

## Equivalence audit

| comparison | label | reason |
|---|---|---|
| gradient mean alignment | `STRICTLY-DIFFERENT` | matching a mean gradient does not determine the induced inverse-Hessian response |
| gradient inner-product maximization | `SPECIAL-CASE` | a one-step quadratic expansion can produce the same directional cross term |
| gradient variance matching | `STRICTLY-DIFFERENT` | variance discards the signed pseudo-response target |
| Hessian alignment | `STRICTLY-DIFFERENT` | equal Hessians do not fix the forcing block `C` |
| gradient plus Hessian moment alignment | `SPECIAL-CASE` | in the exact quadratic head, all required blocks are moment functions |
| MLDG one-step meta-objective | `SPECIAL-CASE` | both use source LOO transfer, but MLDG does not explicitly constrain exact `Pi` |
| transferability local adversarial objective | `UNRESOLVED` | the target-risk transfer measure and the frozen affine response are different objects |
