# Minimal theorem attempt: source-only expected future-domain risk

This is a proof gate for the proposed route. It is deliberately smaller than a final DG theorem. Failure of its assumptions or bound should stop the route before adding a new representation.

## Setup

Let `P_1,...,P_m` be iid environments from a meta-distribution `Pi`. In environment `e`, observe `n` iid labeled examples. Let `ell(f(x),y)` lie in `[0,1]`, let `F` be a predictor class, and define

`R_P(f) = E_P ell(f(X),Y)` and `G_F = {g_f : P -> R_P(f) : f in F}`.

The primary target is

`Q_Pi(f) = E_{P~Pi} R_P(f)`.

This is expected risk on a fresh exchangeable domain. It is not the risk of an arbitrary fixed target.

## Candidate statement

Assume, for simplicity, equal domain sample size `n`. With probability at least `1-delta` over the sampled environments and examples, simultaneously for all `f in F`,

`Q_Pi(f) <= (1/m) sum_e hat R_e(f)`

`  + 2 Rad_m(G_F) + sqrt(log(4/delta)/(2m))`

`  + 2 Rad_n(ell o F) + sqrt(log(4m/delta)/(2n)).`

Here `Rad_m(G_F)` is the complexity of the class of domain-level risks evaluated on `m` environments, and `Rad_n(ell o F)` is a within-domain loss-class complexity. Constants can be replaced by the preferred VC/Rademacher formulation; the point is the decomposition, not this particular complexity convention.

## Proof sketch

For every `f`, add and subtract the source population average:

`Q_Pi(f) - (1/m) sum_e hat R_e(f)`

`= [Q_Pi(f) - (1/m) sum_e R_{P_e}(f)]`

`  + (1/m) sum_e [R_{P_e}(f)-hat R_e(f)].`

The first bracket is uniform convergence for `G_F` over iid environments. The second is bounded by the maximum within-domain uniform-convergence deviation, followed by a union bound over `m` domains. Applying the two standard bounded-loss Rademacher inequalities and a union bound yields the candidate display.

## Status of terms

- empirical source risk: `SOURCE-ESTIMABLE`;
- `Rad_m(G_F)`, `Rad_n(ell o F)`: `SOURCE-ESTIMABLE` or upper-bounded from class assumptions;
- `Pi` exchangeability and iid domain sampling: `ASSUMPTION-CONTROLLED`;
- expected future-domain target: source-only under the meta-law;
- single fresh-domain deviation `R_{P_T}(f)-Q_Pi(f)`: not removed by this theorem.

## Necessary extension for one target domain

For an independent `P_T~Pi`, a pointwise target-risk statement requires an additional upper-tail quantity, for example

`q_delta(f) = inf{t : Pr_{P~Pi}[R_P(f)-Q_Pi(f) <= t] >= 1-delta}`.

Then `R_{P_T}(f) <= Q_Pi(f)+q_delta(f)` with probability at least `1-delta` over the target draw. `q_delta` is family-controlled and generally not source-estimable without enough independent source domains or a tail model. Removing it would falsely turn expected future-domain control into arbitrary target control.

## Falsification / tightness checks

1. If `G_F` is too rich, the domain-level complexity is large or infinite; the theorem is then vacuous.
2. If source domains are not iid or representative of `Pi`, the first bracket has a coverage/misspecification remainder.
3. If two meta-distributions agree on the observed source domains but have different tail behavior, no source-only estimator can identify `q_delta` without additional structure.
4. A conditional-shift counterexample with equal source risks but different target labels leaves this theorem valid for `Q_Pi` only under the chosen `Pi`; it does not justify marginal alignment as a conditional guarantee.

## Gate decision

The theorem attempt supports advancing the domain-of-domains route for expected future-domain risk. It does not support a new universal representation or a pointwise arbitrary-target theorem. The next proof task is to make the complexity and coverage assumptions explicit for the intended hypothesis class, then compare the result with a declared DRO uncertainty set.
