# Frozen Adversarial Critic Reports

The cards were reviewed against the five protocol attack axes.

## Scientific / OOD fidelity

- The `Pi` route is valid only for exchangeable future domains; it cannot be relabeled as arbitrary-target DG.
- `U(S)` routes are scientifically meaningful only when metric, radius, support, or intervention coverage is justified independently of the observed risks.
- Any card omitting a conditional-label term fails the shared-task-mechanism requirement. **Verdict: all universal versions FATAL; layered versions SURVIVES.**

## Algorithm / regularizer fidelity

- ERM, population V-REx, finite GroupDRO, fixed-class MMD/IPM and declared-set DRO are exact at their stated levels.
- CORAL and DANN are distinct moment/discriminator functionals; neither is MMD and classic DA forms use target data.
- Ideal IRM is not IRMv1. Fishr gradient covariance and MLDG update maps require population/surrogate labels plus optimization error. **Verdict: repairable only with typed fidelity labels.**

## Mathematical tractability

- Risk-vector and finite-mixture robust bounds have standard concentration and support-function proofs.
- Operator bounds need bounded kernels, loss-class inclusion and finite complexity; conditional operators need overlap.
- Derivative-class concentration is a bottleneck and cannot be hidden in `O(1/sqrt(n))`. **Verdict: typed outer theorem SURVIVES; unrestricted derivative unifier MAJOR-REVISION.**

## Representation sufficiency / counterexample

Let `Y=C` and all source domains expose a shortcut `S=C`. Predictors `f_C=C` and `f_S=S` have identical zero source risk and source risk vectors. In one admissible target `S=C`; in another `S=-C`, while `Y=C` is unchanged. Target risks of `f_S` are 0 and 1. Marginal MMD/CORAL and risk-vector summaries are identical. **Verdict: any theorem without a target-family or conditional remainder is FATAL.**

## Novelty / overlap

- Domain-of-domains concentration, RKHS discrepancy, conditional mean operators, DRO duality and TV translations are established families.
- A typed outer decomposition plus fidelity/bridge ledger is a new research protocol and possibly a theorem architecture, not a new universal representation.
- Novelty is **HIGH RISK** until a sharp regularizer-to-property comparison theorem or impossibility result is proved.
