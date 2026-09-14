# Stage 13R.2: Source Identifiability of the Transfer Measure

## Decision and scope

Stage 13R.1 rejected environmental sensitivity as an independent endpoint. This
stage accepts transfer measure as an imported endpoint and asks a different
question:

> How much of the transfer measure can source environments identify or control?

The scope is finite-dimensional population theory. No new regularizer is
designed, and no claim is made that the transfer measure itself is novel.

**Decision: `ADVANCE-TO-TRANSFER-CERTIFICATE`.** The Stage 12 support/quotient
machinery lifts exactly to a robust transfer-measure theorem and a sharp
source-only identifiability barrier. This is a candidate contribution about the
source-information interface, not a replacement for Zhang et al. (2021) or
Moment Alignment (Chen et al., UAI 2025).

## 1. Imported endpoint

For predictor class `Gamma`, define excess risk

```text
E_P(f) = R_P(f) - R_P^*,       R_P^* = inf_{h in Gamma} R_P(h).
```

The one-sided transfer measure is

```text
T_Gamma(S || T) = sup_{f in Gamma} [E_T(f) - E_S(f)].
```

Zhang, Zhao, Yu & Poupart (NeurIPS 2021, arXiv:2106.03632) introduced this
transferability/transfer-measure framework, proved a target-error bound, and
showed equivalence with their transferability definition. Chen et al. (UAI
2025, arXiv:2506.07378) use the same excess-risk endpoint in their
multi-source Moment Alignment theory and upper-bound it with parameter-side
gradient/Hessian moments.

We therefore treat `T_Gamma` as established prior work and place the new theorem
interface after the endpoint.

## 2. Excess-risk affine representation

Let `Psi(P) in V`, source center

```text
Psi_bar_S = (1/m) sum_e Psi(P_e),
delta_T = Psi(P_T) - Psi_bar_S,
delta_e = Psi(P_e) - Psi_bar_S.
```

Assume the declared representation

```text
E_P(f) = a_f + g_f(Psi(P)) + eta_f(P),
```

with `|eta_f(P)| <= epsilon_f` on the declared source/target family. Then

```text
E_T(f) - (1/m)sum_e E_e(f)
 = g_f(delta_T) + eta_f(P_T) - (1/m)sum_e eta_f(P_e).
```

For the ideal affine case `eta=0`, the robust transfer measure over an external
target-shift family `U` is exactly

```text
T_Gamma^U(S) := sup_{delta in U} sup_{f in Gamma} [E_delta(f)-E_barS(f)]
              = sup_{f in Gamma} h_U(g_f),
h_U(g) = sup_{delta in U} g(delta).
```

With the uniform residual bound, the inequality becomes

```text
T_Gamma^U(S) <= sup_f h_U(g_f) + 2 sup_f epsilon_f.
```

This is the exact Stage 12 support theorem with the endpoint renamed to the
established transfer measure. The residual remains explicit.

## 3. Source-observable quotient and sharp certificate

Let `S = span{delta_e}` and `S° = {h : h|_S = 0}`. Source excess-risk responses
identify `g_f` only through `[g_f] in V*/S°`. For an admissible dual extension
class `H_adm`, define

```text
Fiber([g]) = {q in H_adm : q-g in S°},
C_U([g]) = sup_{q in Fiber([g])} h_U(q).
```

The sharp source-only robust transfer certificate is

```text
C_U^Gamma = sup_{f in Gamma} C_U([g_f]).
```

Any certificate that uses only source directional responses must be at least
`C_U^Gamma`: all dual extensions in the compatible fiber are source-
indistinguishable and one uniform certificate must cover every target support.
This is a minimality theorem at the transfer-measure level, not a new definition
of transfer measure.

## 4. Positive coverage theorem

If `U` lies in the exposed source span and is contained in the ellipsoid induced
by the source exposure operator `A`,

```text
U subset {delta in S : delta^T A^dagger delta <= rho^2},
A g = (1/m)sum_e g(delta_e) delta_e,
```

then

```text
h_U(g_f) <= rho * sqrt(g_f(A g_f)),
T_Gamma^U(S) <= rho * sup_f sqrt(g_f(A g_f)) + 2 epsilon_repr.
```

In the exact affine representation,

```text
g_f(A g_f) = (1/m)sum_e g_f(delta_e)^2
```

is the variance of the source excess-risk responses. Therefore V-REx supplies
an exact learner-side statistic for this certificate under the explicit exposed-
span coverage assumption. GroupDRO and MM-REx give alternative support choices
when `U` is respectively the observed convex hull or a declared bounded affine
extrapolation family. These are corollaries under different target families,
not claims that the native objectives are identical.

## 5. Blind-direction impossibility

If `U` contains `delta_perp notin S` and no restriction is imposed on blind dual
extensions, choose `h_0 in S°` with `h_0(delta_perp) != 0`. Then `g_f+t h_0`
has exactly the same source excess-risk responses for every `t`, but

```text
h_U(g_f+t h_0) -> +infinity
```

after choosing the sign of `h_0`. Hence no finite source-only certificate for
`T_Gamma^U` exists under unrestricted blind target directions. This formalizes
the target-information warning already present in Zhang et al. (2021): source
observations alone do not identify transferability outside the declared target
family. It also explains why Moment Alignment's derivative bounds require
convex-hull/IRM/curvature assumptions rather than following from source risks
alone.

## 6. What is actually new here

The endpoint, target-risk decomposition, and Moment Alignment derivative bounds
are imported. The surviving theorem contribution is the explicit information
layer between source observations and that endpoint:

```text
source excess-risk responses
 -> quotient V*/S°
 -> compatible-fiber ambiguity C_U^Gamma
 -> finite certificate iff target coverage / blind-extension structure holds.
```

This identifies exactly which transfer-measure values can be source-controlled,
gives a sharp lower bound on every source-only certificate, and separates
regularizer-side statistics from target-family assumptions. It is stronger than
relabeling Stage 12 because the theorem now targets the established transfer
measure and yields method-specific source certificates as corollaries; it is not
an assertion that support functions or transfer measures are new mathematics.

## 7. Proof status

Paper proofs: the affine excess-risk transfer identity with residuals, sharp
fiber minimality, exposed-span transfer bound, and blind-direction impossibility.
Lean checks: finite affine transfer algebra, finite support/max identities, and a
concrete blind-extension witness in `Stage13R2/Basic.lean`. Deterministic tests
cover positive and negative cases.

The next authorized task is a strict regularizer-to-transfer certificate gate.
It must use one declared target family at a time, expose all residual and
coverage assumptions, and compare against Zhang 2021, Hemati 2023, and Chen
2025 before any novelty claim.
