# Minimal theorem: robust risk over a source-defined group uncertainty set

This result is a deliberately narrow robust target. It quantifies only over
mixtures of the observed source environments; it does not imply robustness to
an unseen domain with a new conditional mechanism.

## Setup

There are `m` source environments `P_1, ..., P_m`. From environment `e`, we
observe `n_e` iid labelled examples and define

`R_e(f) = E_{P_e}[ell_f]`,   `hat R_e(f) = (1/n_e) sum_i ell_f(Z_{ei})`,

where `0 <= ell_f <= 1` and `f` ranges over `F`. Let `W` be a nonempty,
source-declared subset of the probability simplex

`Delta_m = {w >= 0 : sum_e w_e = 1}`.

The admissible target family is the finite mixture family

`U_W(S) = { P_w = sum_e w_e P_e : w in W }`.

For a fixed `f`, the robust population and empirical risks are

`Q_W(f) = sup_{w in W} sum_e w_e R_e(f)`,

`hat Q_W(f) = sup_{w in W} sum_e w_e hat R_e(f)`.

## Uniform bound

Let `L_F = {z -> ell_f(z) : f in F}`. If, for each environment, the empirical
Rademacher complexity of `L_F` is at most `r_e` (conditional on the sample
size), then with probability at least `1-delta`, simultaneously for all
`f in F`,

```
Q_W(f) <= hat Q_W(f) + 2 max_e r_e
                    + sqrt(log(2m/delta)/(2 n_min)),
```

where `n_min = min_e n_e`. The same right-hand deviation bounds
`|Q_W(f) - hat Q_W(f)|`.

For non-identical classes or sample sizes, replace the maximum by the
environment-specific quantity

`max_e [2 r_e + sqrt(log(2m/delta)/(2 n_e))]`.

## Proof

For any vectors `a,b in R^m`, the simplex property gives

`|sup_{w in W} w^T a - sup_{w in W} w^T b| <= ||a-b||_infinity`.

Therefore,

`Q_W(f) - hat Q_W(f) <= max_e |R_e(f)-hat R_e(f)|`.

Apply the standard bounded-loss Rademacher inequality separately in each
environment and union bound over `m`. This yields the displayed deviation,
uniformly over `f`; no target samples are used.

## Interpretation and limitations

- `hat Q_W` and the complexity terms are source-estimable (given a declared
  `W` and class complexity bound).
- `W` and the assertion that the deployment target lies in `U_W(S)` are
  assumption-controlled. Choosing `W` after inspecting held-out target data
  invalidates the source-only claim.
- If the true target is outside the mixture family, the theorem gives no
  guarantee. A robust transfer statement should add the explicit
  misspecification remainder

  `M_W(P_T,f) = R_{P_T}(f) - Q_W(f)` (or its positive part).

- Taking `W = Delta_m` recovers worst-source-group risk. Taking a chi-square,
  KL, or other divergence ball around the uniform weights gives a softer
  GroupDRO objective, but the same finite-group concentration argument still
  applies as long as `W subseteq Delta_m`.

This theorem is complementary to a meta-distribution theorem for expected risk:
the finite-group robust target protects against reweighting known source
domains, whereas `E_{P~Pi} R_P(f)` concerns a fresh exchangeable domain. Neither
controls an arbitrary out-of-support target without an additional family or
tail assumption.

