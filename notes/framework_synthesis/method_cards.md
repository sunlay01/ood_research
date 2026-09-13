# Exact method cards: source-only OOD/DG

Status labels in this file are deliberately conservative:

- `EXACT`: the training object and the stated population object have the same semantics.
- `PROVED`: the cited source proves the displayed implication in its declared setting.
- `PROVED-UNDER-RESTRICTIONS`: a theorem exists, but only after structural/model restrictions.
- `CONJECTURAL / MOTIVATIONAL`: a useful interpretation without a general theorem for the exact method.
- `ABSENT`: no target-risk bridge is supplied by the cited result.

The common population setting is source laws `P_1,...,P_m`, hypothesis or predictor
`f`, loss `ell`, and an unobserved target law `P_T`. A target-family assumption is
never counted as source information.

## 1. ERM

- **Training object:** `argmin_f m^{-1} sum_e \hat R_e(f)`.
- **Population analogue:** `argmin_f m^{-1} sum_e R_e(f)`. `EXACT` as an objective-level analogue.
- **Proved consequence:** source population risk control plus a class-complexity term under standard concentration. `PROVED` for source generalization.
- **Restricted consequence:** expected future-domain risk can be controlled when source domains and target are sampled from a declared meta-law. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** none for an arbitrary unseen point target. `ABSENT` without a domain-generating law or target-family condition.
- **Failure mode:** low average source risk can hide a source minority or a target conditional reversal.

## 2. GroupDRO / finite-group robust risk

- **Training object:** `min_f max_e \hat R_e(f)`, or the equivalent simplex support `min_f max_{q in Delta_m} sum_e q_e \hat R_e(f)`.
- **Population analogue:** `sigma_{Delta_m}(r(f)) = max_e R_e(f)`, where `r(f)=(R_1(f),...,R_m(f))`. `EXACT` for observed finite groups.
- **Proved consequence:** uniform convergence transfers empirical group risks to population group risks; convex reweighting targets worst observed groups. `PROVED` in the finite-group setting (Sagawa2020).
- **Restricted consequence:** robust risk over a finite mixture of observed source laws. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** `R_T(f) <= sup_{Q in U} R_Q(f)` only if target membership `P_T in U` is an external assumption. `ABSENT` for arbitrary unseen domains.
- **Failure mode:** the deployment target is outside the group/mixture set, or has unseen conditional shift.

## 3. V-REx

- **Training object:** `m^{-1} sum_e \hat R_e(f) + lambda Var_e(\hat R_e(f))`.
- **Population analogue:** `\bar R(f)+lambda Var_e(R_e(f))` when the environment index is treated as a finite design. `EXACT` at the risk-vector level.
- **Proved consequence:** algebraic/toy-model extrapolation results and source-domain concentration. `PROVED-UNDER-RESTRICTIONS` (Krueger2021).
- **Restricted consequence:** under `P_e iid~Pi`, `g_f(P)=R_P(f)`, source variance informs an expected or tail future-domain query only together with a domain-level sampling theorem. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** variance alone does not identify `P_T(Y|X)` or an arbitrary point target. `ABSENT` in the general source-only setting.
- **Failure mode:** identical source risk vectors but different target laws, or a target outside the meta-law support.

## 4. Ideal IRM

- **Training object:** `min_{Phi,w} sum_e R_e(w o Phi)` subject to `w in argmin_{w'} R_e(w' o Phi)` for every source environment.
- **Population analogue:** the same shared-optimum constraint. `EXACT` for the ideal population principle (Arjovsky2019).
- **Proved consequence:** in restricted structural/linear models, sufficient heterogeneity and rank conditions can identify invariant directions. `PROVED-UNDER-RESTRICTIONS` (Kamath2021 and related analyses).
- **Restricted consequence:** an invariant conditional mechanism can transfer over a specified intervention family. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** requires an SCM or equivalent invariant conditional assumption, positivity/coverage, and model correctness. `ABSENT` for arbitrary worlds.
- **Failure mode:** noncausal or degenerate predictors satisfy shared optimality under weak environment variation.

## 5. IRMv1

- **Training object:** `mean_e R_e(w o Phi) + lambda sum_e ||grad_w R_e(w o Phi)||^2`, often evaluated at a fixed reference `w`.
- **Population analogue:** classifier-variable first-order stationarity. `EXACT` only as a derivative statistic; it is a surrogate for ideal IRM, not the constraint itself.
- **Proved consequence:** restricted population equivalences and, in Lai2024, a TV functional interpretation of the classifier-risk surface under smoothness/coarea assumptions. `PROVED-UNDER-RESTRICTIONS`.
- **Restricted consequence:** derivative-to-invariance or derivative-to-OOD conditions in specified function classes. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** empirical derivative -> population functional -> target risk needs concentration, stationarity and a declared environment family. `ABSENT` for arbitrary deep SGD.
- **Failure mode:** a small classifier gradient can coexist with a non-invariant conditional mechanism.

## 6. Fishr

- **Training object:** match per-example classifier-gradient covariances across environments, e.g. `sum_{e<e'} ||Cov_e(g)-Cov_{e'}(g)||^2`.
- **Population analogue:** equality or small discrepancy of gradient covariance laws. `EXACT` as an optimizer-statistic object.
- **Proved consequence:** the algorithm paper gives a causal/optimization motivation; a general exact-deep target-risk theorem is not established in the repository evidence. `CONJECTURAL / MOTIVATIONAL`.
- **Restricted consequence:** any covariance-to-conditional or covariance-to-risk-shift implication would be a new local theorem. `ABSENT` currently.
- **Target-risk bridge:** must identify a derivative-covariance sufficient statistic for a declared target family. No such general injectivity is known here.
- **Failure mode:** two worlds have matched gradient covariance but different target conditionals.

## 7. MMD

- **Training object:** label loss plus source representation discrepancy, e.g. `sum_{e<e'} MMD_k(P_e^Phi,P_{e'}^Phi)^2`.
- **Population analogue:** RKHS mean-embedding distance `||mu_k(P_e^Phi)-mu_k(P_{e'}^Phi)||_H`. `EXACT` for the discrepancy.
- **Proved consequence:** IPM/RKHS duality and concentration control the selected witness class. `PROVED`.
- **Restricted consequence:** target transfer follows from a source-target discrepancy bound plus joint/conditional error and a declared target family. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** marginal representation matching does not control target labels without conditional alignment. Zhao2019 gives the obstruction.
- **Failure mode:** identical feature marginals with opposite target label conditionals.

## 8. CORAL

- **Training object:** `sum_{e<e'} ||Cov(Phi#P_e)-Cov(Phi#P_{e'})||_F^2` plus label loss.
- **Population analogue:** equality of second feature moments. `EXACT` for the moment statistic.
- **Proved consequence:** moment concentration and, under a restricted feature/kernel class, an IPM bound. `PROVED-UNDER-RESTRICTIONS`.
- **Restricted consequence:** transfer requires moment-to-witness control and a conditional/joint-error term. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** no general implication from covariance equality to label-conditional equality. `ABSENT` without extra assumptions.
- **Failure mode:** equal covariance but different higher-order or conditional label structure.

## 9. DANN / domain-discriminator discrepancy

- **Training object:** minimize label loss while maximizing a domain-classifier loss over `Phi`.
- **Population analogue:** an `H`- or `HDeltaH`-type domain discrepancy induced by the discriminator class. `EXACT` only when the discriminator dual is explicitly fixed.
- **Proved consequence:** standard DA bounds retain a joint-error term and usually use target samples. `PROVED` for that DA setting.
- **Restricted consequence:** a source-only theorem needs a source-defined target family and a discriminator witness inclusion. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** source domain separability is not target conditional alignment. `ABSENT` in arbitrary shifted-target DG.
- **Failure mode:** domain confusion removes predictive information or aligns incompatible labels.

## 10. ICP / conditional mechanism invariance

- **Training object:** select a covariate subset `S` for which `P_e(Y|X_S)` is invariant across environments, usually via conditional-independence tests.
- **Population analogue:** a common conditional kernel `K(y|x_S)` across the source/intervention family. `EXACT` under the SCM semantics.
- **Proved consequence:** stable intervention prediction when the causal and intervention assumptions hold. `PROVED-UNDER-RESTRICTIONS` (Peters2016).
- **Restricted consequence:** identified parent sets and target risk under covered interventions. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** graph, faithfulness/coverage, positivity and intervention-family assumptions are external.
- **Failure mode:** multiple conditionals are compatible with finite source environments, or an intervention changes the supposedly stable mechanism.

## 11. Anchor regression

- **Training object:** in a linear model, residual risk decomposed into anchor-explained and anchor-orthogonal parts, commonly `E[(I-P_A)r]^2 + gamma E[P_A r]^2`.
- **Population analogue:** sensitivity of residuals to the observed anchor and a primal-dual shift family. `EXACT` in the linear anchor model.
- **Proved consequence:** robustness over the declared anchor-shift family via a primal-dual theorem. `PROVED-UNDER-RESTRICTIONS` (Rothenhaeusler2021).
- **Restricted consequence:** target risk control for shifts represented by the anchor covariance geometry.
- **Target-risk bridge:** target membership in the anchor family is external; arbitrary shifts are outside the theorem.
- **Failure mode:** omitted shift directions or nonlinear mechanisms not encoded by anchors.

## 12. Wasserstein and f-divergence DRO

- **Training object:** `min_f sup_{Q: d(Q, P_hat)<=rho} E_Q ell_f`, or a finite-group/f-divergence reweighting support function.
- **Population analogue:** `sigma_U(ell_f)=sup_{Q in U} R_Q(f)`. `EXACT` for the chosen uncertainty set.
- **Proved consequence:** transport/convex duality and uniform convergence yield robust risk guarantees over `U`. `PROVED` under metric, support, radius and integrability assumptions (Esfahani2018; Duchi2021).
- **Restricted consequence:** deployment risk if `P_T in U`. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** the central scientific claim is target-set coverage, not the dual objective itself.
- **Failure mode:** radius/metric misspecification or a target conditional outside `U`.

## 13. Norm / margin / stability control

- **Training object:** spectral/path/norm constraints, margins, or a replace-one stability coefficient.
- **Population analogue:** a restricted hypothesis class or algorithm sensitivity. `EXACT` as a class/stability object.
- **Proved consequence:** same-distribution source generalization. `PROVED` (Bousquet2002; Bartlett2017).
- **Restricted consequence:** can enter an OOD bound only as the complexity/estimation term after a separate shift theorem. `PROVED-UNDER-RESTRICTIONS`.
- **Target-risk bridge:** capacity control alone says nothing about which targets are admissible. `ABSENT`.
- **Failure mode:** a stable, low-norm predictor can be confidently wrong under conditional shift.

## 14. PAC-Bayes / information complexity

- **Training object:** posterior `Q` with source Gibbs risk plus `KL(Q||P)` (or information density / mutual information for a stochastic learner).
- **Population analogue:** posterior risk and a complexity radius. `EXACT` for the posterior-level theorem.
- **Proved consequence:** change-of-measure bounds for randomized predictors; Liu2024/2025 supply OOD information-theoretic variants under their shift model. `PROVED-UNDER-RESTRICTIONS`.
- **Restricted consequence:** stochastic optimization can be analyzed through its information leakage, but this is not the exact deterministic IRMv1/Fishr objective.
- **Target-risk bridge:** target disagreement, information radius and shift assumptions remain explicit.
- **Failure mode:** an operationally unavailable prior/radius or a shift outside the assumed information model.

## 15. MLDG / bilevel meta-learning

- **Training object:** `L_meta-test(theta - alpha grad L_meta-train(theta))`, with source environments split into meta-train/meta-test.
- **Population analogue:** an update map and cross-environment loss. `EXACT` at the bilevel objective level.
- **Proved consequence:** Taylor/linear/convex analyses explain first-order cross-environment terms. `PROVED-UNDER-RESTRICTIONS`.
- **Restricted consequence:** expected future-domain risk only under a domain meta-law or explicit held-out-domain model.
- **Target-risk bridge:** exact deep optimization dynamics and arbitrary target shifts are not represented. `ABSENT` generally.
- **Failure mode:** meta-environment split is unrepresentative, or the one-step surrogate has a large optimization/curvature error.

## Bottom-up comparison

The exact objects fall into distinct semantic levels:

1. `r(f)=(R_1(f),...,R_m(f))` and support functions: ERM, V-REx, GroupDRO.
2. Pushforward distribution witnesses: MMD, CORAL, DANN.
3. Conditional mechanisms and intervention responses: ICP, anchor regression, ideal IRM.
4. Derivative/objective functionals: IRMv1, Fishr, MLDG.
5. Complexity/posterior controls: norm, stability, PAC-Bayes.

These levels can feed a common target-risk theorem only after a local translation
to a certificate with the same semantics. The table is therefore evidence for a
modular framework, not for a single universal primitive.
