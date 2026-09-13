# Adversarial attack on the bottom-up synthesis

This report treats the proposed structure as falsifiable. Evidence labels refer
to the repository ledger and the cited primary papers.

## Attack 1: marginal alignment is not conditional transport

Take binary `X,Y`, with every source environment having the same uniform `X` and
`Y=X`. Let two admissible target worlds share all source laws and all source
representation marginals, but let target world A have `Y=X` and target world B
have `Y=1-X`. The predictor `f(x)=x` has target risks `0` and `1`.

Thus MMD, CORAL, domain confusion, source risks and any representation retaining
only the source marginal can agree while target risk differs by one. P2 cannot
be compressed into P1, and a conditional residual is mandatory. This is a
`PROVED` counterexample pattern, consistent with Zhao2019.

## Attack 2: source-risk variance is not arbitrary-target control

Let source risks be identical for two predictors/worlds, or let all source
domains be identical. V-REx and ERM see the same risk vector. Define the unseen
target conditional adversarially. Target risks separate by a constant while all
source statistics agree. Therefore P1 needs a meta-law, finite-mixture family, or
an ambiguity term. It cannot be treated as a target mechanism.

## Attack 3: shared optimality does not identify causality without rank/coverage

In a restricted linear SCM with insufficient environment heterogeneity, both a
causal and a spurious direction can satisfy the shared-optimum condition. This is
the non-identifiability construction discussed by Rosenfeld2021 and Kamath2021.
Therefore P3 cannot be replaced by “low IRMv1 gradient” without explicit model,
rank, faithfulness and intervention assumptions.

## Attack 4: derivative covariance is not a sufficient state

Fishr retains a second-order statistic of per-example classifier gradients. There
is no injectivity theorem from this covariance to `P_T(Y|X)` or to the risk surface
outside the declared parameterization. Two worlds can match a finite set of
gradient covariances at a stationary point while differing away from that point.
The proposed universal P4 -> target-risk arrow is therefore `ABSENT`; P4 must be
kept as an algorithm-specific translation layer.

## Attack 5: complexity cannot replace a shift model

A zero stability defect or small spectral norm can coexist with a target label
reversal. The resulting source generalization theorem is valid but says nothing
about target membership. P5 is orthogonal and cannot be promoted to the OOD
primitive.

## Attack 6: DRO is exact only for its declared uncertainty set

The support theorem `R_T <= sup_{Q in U} R_Q` is sound if `P_T in U`. If the
metric/radius/support is misspecified, the theorem can be vacuous or simply not
apply. Calling `U` source-derived without an inclusion assumption hides the hard
scientific step.

## Attack result

The only surviving commonality is procedural and semantic:

1. preserve the exact training object;
2. translate it, if possible, to a typed bridge certificate;
3. declare the target family and query;
4. apply a theorem with explicit residuals;
5. return non-identifiability or misspecification when the bridge fails.

This is a framework architecture, not a lossless representation of all methods.
