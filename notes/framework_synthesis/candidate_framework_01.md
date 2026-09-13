# Candidate framework 01: typed source-to-target bridge calculus

Status: `OUR-REFORMULATION`, not an accepted theorem. This is the strongest
architecture that survived the bottom-up attack.

## 1. Admissible worlds

A world is

`W = (P_1,...,P_m, P_T, H, ell, U_T, q_T, A)`,

where `P_e` are source laws, `P_T` is unobserved, `H` is the predictor class,
`ell` is bounded or otherwise controlled loss, `U_T` is an externally declared
target family, `q_T` is the target query, and `A` is the learning procedure.
The admissible class `W_adm` must state what can vary: covariates, conditionals,
environment weights, interventions, or distributions in a metric ball. Stable
mechanisms, rank, support, radius, meta-law and target inclusion are external
conditions, not source observations.

Population, estimation and computation are separate layers:

- population: exact `P_e`, `U_T`, bridge and query;
- estimation: `S_e ~ P_e^{n_e}`, empirical risks/statistics and confidence terms;
- computation: optimization/solver error `epsilon_opt` and relaxation error
  `epsilon_relax`.

## 2. Primitive state and typed bridge

The framework does not retain an arbitrary tuple of all method statistics. It
retains the exact method object `Omega_j` only as an algorithm trace, and a
single typed certificate `B_j` for the selected theorem:

`Omega_j(P_1,...,P_m; A) --T_j--> B_j`,

The certificate is a tagged sum:

`B_j ::= RiskSupport(r,U) | WitnessConditional(D,Delta_cond) | Mechanism(K,I) | ObjectiveFunctional(F,G) | Complexity(C)`.

The active branch is one of:

1. `RISK-SUPPORT`: `r_S(f)` plus a support/variance/meta-law functional;
2. `WITNESS-CONDITIONAL`: a pushforward discrepancy plus a conditional/joint
   residual;
3. `MECHANISM`: a stable conditional kernel/intervention response;
4. `OBJECTIVE-FUNCTIONAL`: a risk surface or derivative law, which is not a
   target certificate until a local theorem maps it to one of types 1-3;
5. `COMPLEXITY`: class/posterior/stability/information radius.

The map `T_j` is labeled `EXACT`, `PROVED`, `PROVED-UNDER-RESTRICTIONS`,
`CONJECTURAL` or `ABSENT`.

## 3. Source observation map

`O_S(W) = (P_1,...,P_m)` at population level; finite samples produce
`\hat O_S=(S_1,...,S_m)`. A source statistic is source-available if it is a
measurable function of `O_S`; it is source-inferred if a theorem maps it to a
population certificate; `U_T` membership and target query semantics are not
source-available in general.

Define source indistinguishability by

`W_1 ~_S W_2  iff  O_S(W_1)=O_S(W_2)`

and algorithm randomness/initialization are coupled. A source-only procedure has
the same output on indistinguishable worlds.

## 4. Target query and backbone theorem

The query is explicit, for example

`q_T(W,f)=R_{P_T}(f)`,
`q_U(W,f)=sup_{Q in U_T} R_Q(f)`, or
`q_Pi(W,f)=E_{P~Pi} R_P(f)`.

The generic certificate theorem is a typed implication:

`q_T(W,f) <= C(B_j, U_T, W) + epsilon_est + epsilon_opt + epsilon_relax`.

Examples of sound instantiations:

**Risk support:** if `P_T in U_T`,
`R_T(f) <= sigma_{U_T}(r(f))`.

**Witness/conditional:** for bounded loss,
`R_T(f)-R_S(f) <= D_G(P_T^Phi,P_S^Phi) + Delta_cond(f)`,
where `Delta_cond` is retained rather than silently absorbed.

**Mechanism:** if `K_T=K` on the covered intervention family and `Phi` identifies
the stable mechanism, the conditional term vanishes or is contracted; otherwise
the theorem returns a non-identification/coverage failure.

No undefined “shift term” is permitted: each residual has a semantic name and a
declared status.

## 5. Mechanism interfaces

| Method | Interface | Status |
|---|---|---|
| ERM | uniform functional of `r_S` | NATIVE |
| V-REx | variance of `r_S` | NATIVE |
| GroupDRO | support of finite simplex / declared `U_T` | NATIVE |
| MMD | RKHS witness in `WITNESS-CONDITIONAL` | NATIVE |
| CORAL | second-moment witness | REQUIRES-TRANSLATION-THEOREM |
| ICP / anchor | stable mechanism or shift-response kernel | NATIVE under stated model |
| ideal IRM | shared optimum / mechanism certificate | NATIVE under stated model |
| IRMv1 | derivative -> risk functional | REQUIRES-TRANSLATION-THEOREM |
| Fishr | gradient covariance -> typed certificate | REQUIRES-TRANSLATION-THEOREM; currently unproved |
| Wasserstein/f-DRO | support over declared uncertainty set | NATIVE |
| norm/PAC-Bayes | complexity layer | NATIVE as estimation layer; outside shift semantics |
| MLDG | update-map -> risk certificate | REQUIRES-TRANSLATION-THEOREM |

## 6. Identifiability and failure semantics

For a retained certificate map `B`, target identification requires

`B(W_1)=B(W_2) and W_1,W_2 in W_adm => q_T(W_1,f)=q_T(W_2,f)`.

At the source level the stronger condition is

`ker(O_S) subseteq ker(q_T)`

on the admissible class, or an approximate version with a bounded modulus.
If false, the framework must return one of:

- `SOURCE-INSUFFICIENT / NON-IDENTIFIABLE`;
- `TARGET-FAMILY-MISSPECIFIED`;
- `MECHANISM-MISMATCH`;
- `TRANSLATION-UNPROVED`;
- `ESTIMATION-OR-COMPUTATION-GAP`.

The binary label-reversal construction makes the first failure explicit.

## 7. Why this is a framework

This is more than a proof template because it fixes the semantic interfaces,
source observation map, admissible-world class, target query, typed translation
obligations and failure outputs. It does not claim one statistic controls every
method. A new algorithm can enter only by supplying `Omega -> B` with a stated
fidelity label and can reuse a generic theorem only when its `B` has the required
type.
