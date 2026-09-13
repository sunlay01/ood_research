# Formalization Round 1 Synthesis

## Decision

Adopt a **shared outer theorem with typed inner modules**, not a single common representation. The outer target is chosen first:

1. `Q_Pi(f)=E_{P~Pi} R_P(f)` for an exchangeable future domain; or
2. `Q_U(f)=sup_{Q in U(S)} R_Q(f)` for a declared robust family.

These targets are not interchangeable. A pointwise target statement needs a fresh-domain quantile/tail term or an explicit uncertainty-set coverage assumption.

## Framework

For `f in F`, use the source risk profile `g_f(P)=R_P(f)` and a typed direct sum

`Z_e(f) = (risk_e, marginal_IPM_e, conditional_operator_e, derivative_functional_e, robust_support_e)`.

Only components needed by a theorem are instantiated. A practical learner is abstracted as an approximate minimizer of a typed population functional, with `epsilon_translation` and `epsilon_opt` retained. The shared task mechanism is `Y=f_tau(C,epsilon_Y)`; shifts in `P(C)`, nuisance, observation and selection channels are permitted, but target coverage must be declared.

## Surviving components

| Component | Origin | Proof operation | Assumption cost |
|---|---|---|---|
| `g_f` and risk vector | Risk explorer; domain-of-domains literature | two-level concentration | exchangeable `Pi`, finite domain complexity |
| finite-mixture support | Robust explorer | simplex Lipschitz and uniform convergence | observed-group coverage |
| marginal/conditional IPM split | Discrepancy and causal explorers | duality and conditional change of measure | bounded kernels, overlap |
| invariant conditional slot | Causal explorer | SCM/conditional factorization | graph, interventions, faithfulness |
| variational translation slot | Variational explorer; Lai2024 | functional/TV arguments | smoothness, derivative class |

## Irreducible bottlenecks

1. **Target-family identification:** source data do not identify arbitrary target tails or conditionals; `Pi`, `U(S)`, or an oracle remainder is necessary.
2. **Regularizer-to-property bridge:** V-REx controls risk dispersion, MMD controls a witness class, and gradients control parameterized derivatives; none implies conditional stability without extra assumptions.
3. **Algorithm fidelity:** IRMv1/Fishr/MLDG practical objectives need derivative-class and optimization errors; ideal IRM is a population abstraction.
4. **Finite-sample operator control:** conditional and derivative operators require boundedness, overlap and complexity conditions that may be vacuous in deep classes.

## Minimal algorithm basis

Use ERM (risk mean), V-REx (risk variance), MMD (marginal IPM), ideal IRM plus IRMv1 translation (conditional/derivative), and GroupDRO or Wasserstein DRO (robust support). Fishr and MLDG remain stress-test surrogates rather than required theorem coverage.

## Error hypothesis

For a fresh target and source empirical mean,

`R_T(f) - Rhat_S(f) = epsilon_within + epsilon_domain + epsilon_fresh`.

The outer theorem bounds `epsilon_domain` by a typed structural term and leaves the fresh tail visible:

`R_T(f) <= Rhat_S(f) + B_Pi(Z_S(f)) + q_delta(f) + epsilon_within + epsilon_translation + epsilon_opt`.

For robust targets, replace the first three terms by `Q_U(f)` and add `M_U(P_T,f)` when the target is outside `U(S)`.

## Status

**NEEDS-TARGETED-VALIDATION.** The architecture survives criticism, but no general bridge from derivative statistics or marginal alignment to conditional target risk is established. The first theorem should therefore be a restricted source-only bound plus a matching counterexample, not a universal regularizer theorem.
