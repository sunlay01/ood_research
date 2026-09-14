# Stage 13R.1: Master-Functional Invariance and Nuisance Audit

## 1. Scope and decision

This gate tests whether the raw environmental-sensitivity functional is an
appropriate ideal OOD-regularization objective, rather than merely a coherent
risk derivative. The analysis is finite-dimensional and uses the same state
semantics as Stages 11R--13. No new regularizer, finite-sample theorem, or
benchmark is introduced.

**Decision: `FAIL-ESF-AS-INDEPENDENT-MASTER`.** Raw ESF is not invariant to
predictor-independent environment difficulty. The only canonical nuisance-
invariant correction, excess-risk ESF, is a path/differential representation
of the excess-risk transfer measure used by Moment Alignment. A pairwise-risk
quotient can remove the nuisance, but it yields relative/ranking sensitivity,
not an absolute target-risk certificate, and the fixed-tangent re-audit does
not produce a new common theorem for V-REx, GroupDRO, MMD, and ideal IRM.

This does not invalidate the Stage 12/13 local results. It removes ESF as an
independent inner master functional unless a future stage introduces a separate
scientific target (for example, explicitly relative-risk OOD guarantees).

## 2. Candidates

Raw ESF:

```text
S_raw(f;xi) = sup_{delta in D(xi)} D_xi R(f,xi)[delta].
```

Excess ESF:

```text
R*(xi) = inf_h R(h,xi),
E(f,xi) = R(f,xi) - R*(xi),
S_exc(f;xi) = sup_{delta in D(xi)} D_xi E(f,xi)[delta].
```

A formally valid third object is the quotient/anchor sensitivity

```text
Q_{f,h0}(xi) = R(f,xi) - R(h0,xi),
S_pair(f,h0;xi) = sup_{delta in D(xi)} D_xi Q_{f,h0}(xi)[delta].
```

It is invariant to `R -> R+c(xi)` and is predictor-specific, but it measures
relative performance against the declared anchor `h0`; it is not an absolute
target-risk certificate unless the anchor risk is controlled separately.

## 3. Gate A: invariance desiderata

| Transformation | Ideal master invariant? | Raw ESF | Excess ESF | Pairwise quotient |
|---|---:|---:|---:|---:|
| `R -> R+c(xi)` | Yes for predictor-independent task difficulty | **No**: `D_xiR` gains `D_xic` | **Yes** when `R*` shifts by the same `c` | **Yes**: `c` cancels exactly |
| `R -> a(xi)R`, `a>0` | No full invariance; covariance is the sensible law | gains `Da R + a DR` | same product-rule issue plus `R*` term | same product-rule issue |
| predictor reparameterization | Yes | Yes if the derivative is on the environment state | Yes | Yes |
| environment coordinate change `xi'=Txi` | Yes under paired dual/tangent transform | Yes, by Stage 11R covariance | Yes if `R*` is scalar | Yes |
| Bayes/irreducible-risk shift | Should not alter relative predictor quality | changes with the shift | cancels the common Bayes shift | cancels the common shift |

The additive-nuisance entry is a modeling requirement, not a universal claim
about every deployment objective. If the scientific goal is absolute service
loss (for example, a harsher environment legitimately incurs higher loss for
all predictors), retaining `c(xi)` can be appropriate. It is not appropriate
for a regularizer whose purpose is to select predictors robust to environment
changes while holding predictor-independent difficulty fixed. The Stage 13R
claim was the latter, so raw ESF fails the invariance test.

For positive rescaling, full invariance would discard real changes in units or
stakes. The correct requirement is covariance/equivariance: the tangent gauge
and any loss normalization must transform with the risk surface.

## 4. Gate B: nuisance decomposition and envelope conditions

Suppose

```text
R(f,xi) = c(xi) + Q(f,xi).
```

Then

```text
D_xi R(f,xi) = D_xi c(xi) + D_xi Q(f,xi),
```

and raw ESF sees the common covector `D_xi c`. Excess ESF removes it because,
under the explicit optimizer assumptions below,

```text
R*(xi) = c(xi) + Q*(xi),
E(f,xi) = Q(f,xi) - Q*(xi).
```

If `R` is continuously differentiable in `xi`, `F` is compact (or coercive),
the minimizer is unique and interior, and `R` is jointly regular enough for the
Danskin/envelope theorem, then

```text
D_xi R*(xi) = D_xi R(f*_xi,xi),
D_xi E(f,xi) = D_xi R(f,xi) - D_xi R(f*_xi,xi).
```

Without uniqueness, differentiability, or an appropriate directional Danskin
condition, `D_xi R*` may be set-valued; differentiating through an arbitrary
argmin is invalid. The theorem is therefore explicitly conditional.

## 5. Gate C: excess ESF versus Moment Alignment

Let `gamma` be absolutely continuous, with risks and `R*` differentiable along
the path. The scalar chain rule gives

```text
E(f,xi_T) - E(f,xi_S)
  = integral_0^1 D_xi E(f,gamma(t))[gamma_dot(t)] dt.
```

Taking a supremum over predictors after the endpoint difference yields exactly
the one-sided transfer quantity whenever the source and target reference
domains are the same as in Moment Alignment:

```text
sup_f [E_T(f) - E_S(f)] = T_Gamma(S || T).
```

Moment Alignment (Chen et al., UAI 2025, arXiv:2506.07378) defines precisely
this excess-risk difference, with multi-source center `mu*`, source/target
optima, and target-family assumptions. Its parameter-derivative Taylor bounds
are a way to upper-bound that endpoint quantity. Excess ESF is therefore a
differential/path representation of the same transfer target under the stated
regularity and reference-domain assumptions, not an independent OOD quantity.

The path geometry can be useful computationally, and a fixed predictor-level
ESF contains local information before taking the endpoint supremum. Neither
fact changes the equivalence of the endpoint excess-risk certificate.

## 6. Gate D: reclassification of the Stage 13R witness

For

```text
R(theta,xi)      = theta^2 + xi,
Rtilde(theta,xi) = theta^2 + 2*xi,
```

all positive-order parameter derivatives agree, while environment derivatives
are `1` and `2`. However, both optima are `theta*=0`, both excess risks are
`theta^2`, predictor rankings are identical, and regret/decision quality is
unchanged. The witness proves only **information separation**, not additional
OOD-regularization relevance. It is a valid nuisance stress test for raw ESF,
not a predictor-relevant novelty theorem.

There is no stronger predictor-relevant witness under the all-order scope. If
two smooth risk surfaces have identical `D_theta^k` for every order `k>=1` on a
connected parameter region, their difference is constant in `theta`; hence it
has the form `c(xi)` and cannot change rankings, argmin identity, or excess
risk. This is an impossibility result for strengthening the old witness
without weakening the Moment Alignment observable scope.

## 7. Gate F: the only surviving quotient candidate

The quotient of risk surfaces by

```text
R(.,xi) ~ R(.,xi) + c(xi) * 1_F
```

is represented by pairwise differences `Q_{f,h0}`. This object survives the
additive-nuisance invariance test and is not generally equal to excess ESF when
the anchor `h0` is fixed rather than the environment-wise Bayes optimizer.
Nevertheless, it has two hard limitations:

1. It certifies only relative risk/ranking motion. Absolute target risk still
   requires the uncontrolled anchor term `R(h0,xi_T)`.
2. V-REx and GroupDRO act on absolute source-risk vectors, so their native
   penalties do not control `S_pair` without adding a reference-risk statistic.
   MMD can upper-bound any fixed witness norm, and ideal IRM can restrict a
   mechanism fiber, but these are generic upper-bound/structural statements,
   not a new common theorem linking all four methods.

Thus the quotient is a legitimate future relative-risk module, not a surviving
independent absolute OOD master for this project.

## 8. Theorem/counterexample status

- **T1 additive nuisance:** proved exactly by the product rule; raw ESF gains
  `D c`, while excess and pairwise forms cancel the common term.
- **T2 excess identity:** proved under the stated Danskin/envelope assumptions;
  nonunique optimizer cases are explicitly excluded or require directional
  subgradients.
- **T3 equivalence:** proved at the endpoint/path level under differentiability
  and the same source/target reference convention as Moment Alignment.
- **T4 fixed tangent comparison:** completed in the companion fixed-tangent
  re-audit; no common independent certificate survives.
- **T5 predictor-relevant separation:** impossible under all-order parameter
  derivative equality; the original witness is correctly downgraded.

Only the finite algebraic identities are candidates for Lean checking. The
envelope theorem, path chain rule, and impossibility argument remain paper
proofs with deterministic witnesses.

## 9. Final gate decision

`FAIL-ESF-AS-INDEPENDENT-MASTER`.

Raw ESF fails the nuisance-invariance requirement; excess ESF collapses to the
Moment Alignment transfer target; and the pairwise quotient is relative-risk
only and lacks a new fixed-tangent cross-method theorem. Preserve the valid
Stage 12/13 support and translation results, but stop treating ESF as the
project's independent inner master functional.
