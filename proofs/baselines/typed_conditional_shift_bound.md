# Minimal theorem attempt: typed marginal/conditional target bound

This theorem is the smallest bridge from the typed representation to target
risk. It is deliberately conditional on a declared target family.

## Setup

Let `P_bar` be a source mixture and `Q` a target law. For `f in F`, let
`ell_f(x,y)` lie in `[0,1]` and define

`m_P(x) = E_P[ell_f(X,Y) | X=x]`.

Assume the loss section `m_Pbar` belongs to a witness class `G_m` with IPM
norm at most `L`, and that the target family satisfies

`D_Gm(Q_X, Pbar_X) <= rho_m`,

and a conditional loss discrepancy budget

`E_{Q_X}|m_Q(X)-m_Pbar(X)| <= rho_c`.

The two radii are declared assumptions (or are estimated only under a separate
coverage model); they are not inferred from marginal alignment alone.

## Claim

For every `f` and every `Q` in the declared family,

`R_Q(f) <= R_Pbar(f) + L rho_m + rho_c`.

If source risks are estimated from `n_e` iid samples and `F` has per-domain
Rademacher complexities `r_e`, then with probability at least `1-delta`,
simultaneously over `f` and all such `Q`,

```
R_Q(f) <= Rhat_Pbar(f) + L rho_m + rho_c
          + sum_e alpha_e [2 r_e + sqrt(log(2m/delta)/(2 n_e))].
```

For an approximate regularized learner add `epsilon_translation + epsilon_opt`.

## Proof

The conditional change-of-measure identity is exact:

`R_Q(f)-R_Pbar(f) = E_{Q_X}[m_Q-m_Pbar]
                      + (E_{Q_X}m_Pbar-E_{Pbar_X}m_Pbar)`.

The first term is at most `rho_c` by assumption. The second is at most
`L D_Gm(Q_X,Pbar_X) <= L rho_m` by IPM duality. Finally apply the standard
bounded-loss Rademacher bound independently in each source environment and
average with fixed source-mixture weights `alpha`.

## Audit status

- Source-estimable: empirical source risk and complexity upper bounds.
- Assumption-controlled: `rho_m`, `rho_c`, witness inclusion, and target-family
  coverage.
- Irreducible without additional structure: `rho_c` and any out-of-family
  misspecification.

The binary channel-swap construction in `archive/failed_formalization_round1/critics.md`
shows that dropping `rho_c` makes the claim false even when all marginal MMD
and CORAL terms vanish.
