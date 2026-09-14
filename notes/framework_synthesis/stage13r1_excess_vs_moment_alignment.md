# Stage 13R.1: Excess ESF versus Moment Alignment

## Definitions

For a finite-dimensional environment state `xi`, let

```text
E(f,xi) = R(f,xi) - R*(xi),
R*(xi) = inf_h R(h,xi).
```

The excess environmental-sensitivity candidate is

```text
S_exc(f;xi) = sup_{delta in D(xi)} D_xi E(f,xi)[delta].
```

Moment Alignment (Chen, Si, Zhang & Zhao, UAI 2025; arXiv:2506.07378) uses the
one-sided multi-source transfer measure

```text
T_Gamma(S || T) = sup_f [(L_T(f)-L*_T) - (L_S(f)-L*_S)].
```

For multiple sources it replaces `S` by the center-of-mass domain `mu*` and
uses the corresponding source optimum, exactly as in Definition 3 of the
paper.

## Conditional envelope identity

Assume `R` is `C1` in `xi`, the predictor class is compact or coercive, the
minimizer `f*_xi` is unique and interior, and the hypotheses of the Danskin/
envelope theorem hold along the path under consideration. Then

```text
D_xi R*(xi) = D_xi R(f*_xi, xi),
D_xi E(f,xi) = D_xi R(f,xi) - D_xi R(f*_xi,xi).
```

The same statement with directional derivatives holds for a nonunique argmin
only after selecting the appropriate directional envelope; an arbitrary
measurable argmin cannot be differentiated silently.

## Path theorem

Let `gamma:[0,1]->V` be absolutely continuous, `gamma(0)=xi_S`,
`gamma(1)=xi_T`, and assume `t -> E(f,gamma(t))` is absolutely continuous.
Then

```text
E(f,xi_T) - E(f,xi_S)
 = integral_0^1 D_xi E(f,gamma(t))[gamma_dot(t)] dt.
```

Taking `sup_f` after evaluating the endpoint difference gives

```text
sup_f [E(f,xi_T)-E(f,xi_S)] = T_Gamma(S || T),
```

when `xi_S` denotes the same source reference (or Moment Alignment's `mu*`)
and `Gamma` is the same predictor class. The equality is simply the definition
of the transfer measure after replacing each endpoint excess risk by its path
integral.

## What is and is not new

Excess ESF can expose a local decomposition into path length and local excess
sensitivity, which may be useful for diagnostics. But as an endpoint target-risk
quantity it is mathematically equivalent to the transfer measure above. The
Moment Alignment parameter-side Taylor bounds and the ESF environment-side path
integral are different *proof coordinates* for the same excess-risk endpoint;
they are not two independent target quantities.

The distinction from raw ESF remains important: raw ESF responds to additive
environment-only difficulty, while excess ESF removes that nuisance. Once this
invariance correction is imposed, the original Stage 13R all-order separation
cannot support an independent novelty claim.

## Referee checks

- No differentiation through an arbitrary argmin is allowed.
- The equivalence requires the same source center, target reference, predictor
  class, and endpoint supremum as Moment Alignment.
- A path-dependent local bound can differ numerically from a Taylor bound, but
  endpoint equivalence means it is not a new transfer objective.
- Without target/path assumptions, neither formulation is source-only global
  control.

**Gate result:** `FAIL-NOVELTY-EXCESS-RISK-EQUIVALENCE` for Candidate B as an
independent master functional.
