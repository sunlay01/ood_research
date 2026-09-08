# ADR-005: Replace Calibration-First With Regularizer Mechanism Accounting

- Date: 2026-09-05
- Status: `ACCEPTED / ACTIVE MAINLINE`
- Supersedes: the calibration-first center previously recorded in this file
- Preserves: C001 risk transport, task-preserving interventions, oracle distinctions, source observability, C004a, C002-IRM, and the Anchor/DRIG comparison work

## Decision

The active research question is:

```text
What operator does a regularizer actually constrain?
Which harmful OOD components are controlled by that operator?
Which components are blind because of its nullspace, source invisibility,
or model/intervention mismatch?
How do those components, their interactions, and statistical/observation/
coverage remainders account for target error?
```

The active chain is therefore:

```text
Omega_j -> actual L_{j,S} -> C_j / B_j -> OOD error accounting.
```

The earlier chain

```text
Omega_j -> effective nuisance sensitivity -> universal intervention certificate
```

is no longer the primary research objective. A certificate or calibration theorem remains admissible only as a special case in which the blind and remainder terms are demonstrably controlled under declared assumptions.

## Reason

The completed IRMv1 result is already enough to show why a certificate-first program is too narrow: a method may constrain a real but insufficient response while leaving a target-harmful direction untouched. The scientifically useful object is not merely pass/fail robustness, but the mechanism-level partition of what is controlled, what is blind, and how both enter risk.

This shift also preserves the strongest existing assets:

- the causal task/interface prevents arbitrary target distributions from being called comparable domains;
- C001 supplies an exact error transport structure in the first model;
- source observability explains why some directions cannot be learned from source variation;
- operator nullspaces allow regularizer-specific failure statements;
- existing uncertainty-set methods can be assessed as matched positive controls rather than forced into an unrelated target ball.

## Operational Rules

1. Each method must specify an actual state space and `L_{j,S}` before being labelled invariant, aligned or robust.
2. `range(L^*)` and `ker(L)` are used only in a stated linear/local setting. A nonlinear objective requires a base point, tangent space and residual term.
3. A controlled/blind split is exact only when it follows from an already derived risk identity and a projector/operator independent of target outcome fitting.
4. Every positive controlled-component theorem must include a paired blind/nullspace, source-unobservable or out-of-family failure condition.
5. `Omega` reduction, latent compactness and PCA factors are diagnostics only until connected to an operator and a risk term.
6. Target risk, target moments and coverage residual are not inputs to source-only tuning. They may be offline labels or declared theorem remainders.
7. No existing exact identity, IRMv1 failure, or generic decomposition is a paper contribution without a separate literature collision decision.

## Consequences

- `C006` (operator-induced controlled/blind accounting) is the next Claim, followed by L2, full-gradient, CORAL/MMD and representation-effective-sensitivity analyses.
- C002-IRM is retained as a blind-component negative control, not an endpoint.
- Anchor Regression/DRIG enter only after strict uncertainty-set matching; `RELATED` is not a recovered guarantee.
- ADR-003 and ADR-004 remain historical evidence records. Their latent/PCA and cross-decomposition work cannot be relabeled as support for this mainline.
- The experiment ledger now requires a per-method operator, blind condition and error accounting, not only an OOD accuracy table or a bound score.

## Reversal Condition

Revisit this decision only if literature establishes an equivalent operator-plus-blind-plus-error-accounting framework for the same task-preserving intervention interface and regularizer classes, or if C006 shows that the decomposition cannot be made non-arbitrary even in the linear exact-transport setting. In either case, stop the novelty route rather than widening claims.
