# Stage 13R.2: Source Identifiability of the Transfer Measure

## Decision and scope

Stage 13R.1 rejected environmental sensitivity as an independent endpoint. This
stage accepts transfer measure as an imported endpoint and asks a different
question:

> How much of the transfer measure can source environments identify or control?

The scope is finite-dimensional population theory. No new regularizer is
designed, and no claim is made that the transfer measure itself is novel.

**Decision: `REVISE-TRANSFER-LIFT`.** The source-observability quotient,
compatible-fiber certificate, and blind-direction barrier survive. The claimed
regularizer-to-transfer lift does not yet hold generically: excess risk is not
automatically affine, native V-REx controls raw-risk variance rather than
excess-risk variance, and the source average used below is not Moment
Alignment's center-of-mass reference. The transfer endpoint is imported from
prior work; the next gate must repair these interfaces before any certificate
claim is advanced.

## 1. Imported endpoint

For predictor class `Gamma`, define excess risk

```text
E_P(f) = R_P(f) - R_P^*,       R_P^* = inf_{h in Gamma} R_P(h).
```

The one-sided transfer measure is

```text
T_Gamma(S || T) = sup_{f in Gamma} [E_T(f) - E_S(f)].
```

Here `E_S` denotes the source reference used by the cited endpoint (pairwise
or the paper's multi-source center construction). It is not automatically the
uniform average `(1/m)sum_e E_e(f)` used in the centered construction below.

Zhang, Zhao, Yu & Poupart (NeurIPS 2021, arXiv:2106.03632) introduced this
transferability/transfer-measure framework, proved a target-error bound, and
showed equivalence with their transferability definition. Chen et al. (UAI
2025, arXiv:2506.07378) use the same excess-risk endpoint in their
multi-source Moment Alignment theory and upper-bound it with parameter-side
gradient/Hessian moments.

We therefore treat `T_Gamma` as established prior work and place the new theorem
interface after the endpoint.

## 2. Excess-risk representation is an additional assumption

Let `Psi(P) in V`, source center

```text
Psi_bar_S = (1/m) sum_e Psi(P_e),
delta_T = Psi(P_T) - Psi_bar_S,
delta_e = Psi(P_e) - Psi_bar_S.
```

Stage 12's affine representation for raw risk does **not** imply an affine
representation for excess risk, because `R_P^* = inf_h R_P(h)` is generally a
concave lower envelope of affine functions. We therefore make the following
additional, explicit assumption whenever the support theorem is invoked:

```text
E_P(f) = a_f + g_f(Psi(P)) + eta_f(P),
```

This `(ER-affine)` condition can follow, for example, from a common optimizer
whose optimal risk is itself affine on the declared family; it does not follow
merely from an IRM common optimizer. Under this extra assumption,

with `|eta_f(P)| <= epsilon_f` on the declared source/target family. Then

```text
E_T(f) - (1/m)sum_e E_e(f)
 = g_f(delta_T) + eta_f(P_T) - (1/m)sum_e eta_f(P_e).
```

For the ideal affine case `eta=0`, define the centered-average quantity over an
external target-shift family `U`:

```text
T_bar^U(S) := sup_{delta in U} sup_{f in Gamma}
              [E_delta(f)-(1/m)sum_e E_e(f)]
              = sup_{f in Gamma} h_U(g_f),
h_U(g) = sup_{delta in U} g(delta).
```

With the uniform residual bound, the inequality becomes

```text
T_bar^U(S) <= sup_f h_U(g_f) + 2 sup_f epsilon_f.
```

`T_bar^U` is a useful robust centered-excess quantity, but it is not silently
identified with Zhang et al.'s pairwise transfer measure or Moment Alignment's
multi-source center-of-mass construction. Passing from this average reference
to those endpoints requires a separate center/multi-source theorem. The
residual remains explicit.

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

## 4. Positive coverage theorem (conditional excess-risk corollary)

If `U` lies in the exposed source span and is contained in the ellipsoid induced
by the source exposure operator `A`,

```text
U subset {delta in S : delta^T A^dagger delta <= rho^2},
A g = (1/m)sum_e g(delta_e) delta_e,
```

then

```text
h_U(g_f) <= rho * sqrt(g_f(A g_f)),
T_bar^U(S) <= rho * sup_f sqrt(g_f(A g_f)) + 2 epsilon_repr.
```

In the exact affine representation,

```text
g_f(A g_f) = (1/m)sum_e g_f(delta_e)^2
```

is the variance of the source excess-risk responses. It is therefore not, in
general, the native V-REx statistic. Native V-REx uses `Var_e R_e(f)`; exact
identification with excess-risk variance requires the extra condition
`R_e^*` constant across environments. GroupDRO and MM-REx likewise act on raw
risks unless their objectives are explicitly redefined on excess risks. Their
support interpretations are conditional statements, not generic exact
translations of the native algorithms.

## 5. Native V-REx bound for the established transfer endpoint

To preserve the actual algorithm, retain raw risks. For every pair of source
environments and every predictor,

```text
|R_j(f)-R_i(f)| <= sqrt(2*m*Var_e R_e(f)).
```

Combining this elementary range bound with the existing convex-hull/pairwise
transfer theorem from Moment Alignment gives the conditional certificate

```text
T_Gamma(S||T)
 <= sqrt(m/2) * sup_f sqrt(Var_e R_e(f))
    + (1/2) * range_e R_e^*.
```

The second term is the correction for source optimum-risk heterogeneity. It
vanishes when `R_e^*` is constant, without changing the native V-REx objective.
This is the correct route from a real regularizer to an established transfer
measure; it is not a new transfer theorem and remains conditional on the
target-family assumptions of the imported result.

## 6. Blind-direction impossibility

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

## 7. What is actually new here

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

## 8. Proof status

Paper proofs: the affine excess-risk transfer identity with residuals, sharp
fiber minimality, exposed-span transfer bound, and blind-direction impossibility.
Lean checks: finite affine transfer algebra, finite support/max identities, and a
concrete blind-extension witness in `Stage13R2/Basic.lean`. Deterministic tests
cover positive and negative cases.

The next authorized task is a strict regularizer-to-transfer certificate gate.
It must use one declared target family at a time, expose all residual and
coverage assumptions, distinguish raw-risk objectives from excess-risk
representations, and compare against Zhang 2021, Hemati 2023, Chen 2025, Xu
2022, and Partial Transportability for Domain Generalization (NeurIPS 2024)
before any novelty claim.
