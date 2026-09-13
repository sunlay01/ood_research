# Candidate Framework: Typed Source-Only OOD Error Layer

## Environment and information model

Sources `P_1,...,P_m` provide labeled samples; the target is unavailable. Every admissible environment shares `Y=f_tau(C,epsilon_Y)`. The theorem chooses either `P_T~Pi` or `P_T in U(S)`. All target-family, radius, support, overlap and SCM claims are assumption-controlled.

## Algorithm and representation layers

The abstract learner returns `f_hat` with optimization error `epsilon_opt` for a typed objective. The outer representation is `g_f(P)=R_P(f)`. Optional typed modules are:

- risk vector and variance;
- marginal RKHS/IPM mean and covariance operators;
- conditional kernel/operator or invariant mechanism;
- derivative score and covariance functionals;
- robust support function over `U(S)`.

## Translation policy

`EXACT`: ERM, population V-REx, finite observed-group GroupDRO, fixed-class IPM/MMD, declared-set DRO.

`EQUIVALENT-UNDER-ASSUMPTIONS`: ideal IRM, ICP/anchor and causal invariant models.

`FUNCTIONAL-TRANSLATION`: CORAL, DANN, IRMv1 and the Lai-style TV functional.

`POPULATION-ABSTRACTION` / `SURROGATE`: Fishr, MLDG, deep representation learners.

`NOT-COVERED`: arbitrary targets, target-dependent DA claims in source-only mode, and exact deep optimizer trajectories.

## Structural bridge and error layer

For a source mixture `P_bar` and target `Q`, use the exact conditional identity

`R_Q(f)-R_Pbar(f) = <m_Pbar,f, Q_X-Pbar_X> + E_QX[m_Q,f-m_Pbar,f]`.

The first term is controlled by a declared marginal IPM/radius; the second is a conditional-label remainder, zero only under verified invariant mechanism. Add source estimation, domain concentration, approximation, translation and optimization terms. Regularizers affect the bound only through a proved inequality from their population value to one of these radii or support functions.

## Adequacy gate

The framework is adequate only if a restricted non-vacuous theorem exists for at least ERM/V-REx/GroupDRO and one conditional or IPM refinement, with finite complexity and an explicit target-family term. A source-indistinguishable two-world construction must match the theorem's lower-order terms and show why the conditional or family term cannot be dropped.
