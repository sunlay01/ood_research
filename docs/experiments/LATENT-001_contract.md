# LATENT-001 Frozen Experiment Contract

Status: `DESIGN_GATE pending`. This file is frozen after the first
`DESIGN_GATE=PASS`; later changes require a new experiment ID.

## Question

For ERM, L1, L2, scalar-scale IRMv1, Gaussian-RBF MMD, CORAL, shared-head
gradient alignment, and shared-head Hessian alignment, determine:

1. which source-supervised semantic latent component is suppressed;
2. which covered task-preserving target shift consequently improves;
3. which operator, source-design, semantic, interaction, or optimization blind
   direction causes failure;
4. whether a predeclared componentwise population risk bound is available.

The candidate direct sum is a hypothesis, not an assumption:

\[
\mathcal Z=\mathcal Z_{task}\oplus\mathcal Z_{relation}\oplus
\mathcal Z_{mean}\oplus\mathcal Z_{covariance}\oplus\mathcal Z_{residual}.
\]

## Data-Generating Process

Use independent standard Gaussian innovations and fixed dimensions
`d_C=3`, `d_U=d_rel=d_mean=d_cov=3`, `d_noise=3`:

\[
Y=\beta^TC+\epsilon_Y,\quad U=LC+\xi,
\]
\[
A_e^{rel}=R_eC+\eta_r,\quad A_e^{mean}=\mu_e+\eta_m,
\quad A_e^{cov}=S_e\eta_c,\quad N=\eta_n.
\]

The input concatenates all five blocks. Source environments form a factorial
baseline plus signed interventions on axes 1 and 2 for relation, mean, and
covariance. Axis 3 is absent from source intervention labels and reserved as an
unseen target direction. The task equation and `beta` never change.

Frozen target labels:

- `covered_interpolation`;
- `covered_extrapolation`;
- `unseen_relation_axis3`;
- `relation_sign_flip`;
- `mean_axis1`;
- `covariance_axis2`;
- `compound_covered`.

No target quantity may enter training, lambda selection, whitening, operator
regression, residualization order, semantic naming, or budget selection.

## Model and Methods

The model is `Z=BX`, `f=w^TZ+b`. Main latent dimension is 15 so that the
linear encoder has full capacity; bottleneck dimensions 4 and 2 are negative
controls. Every method uses the same seed-specific source samples,
initialization, optimizer steps, and lambda grid. ERM is rerun for each seed and
capacity and paired with every regularized model.

Objectives are exactly source MSE plus lambda times:

- L1/L2: joint encoder/head parameter norm;
- IRMv1: mean squared environment-wise derivative under scalar output scaling;
- MMD: mean pairwise Gaussian-RBF MMD of latent distributions;
- CORAL: mean pairwise latent covariance Frobenius discrepancy;
- gradient alignment: variance of shared-head risk gradients;
- Hessian alignment: variance of shared-head risk Hessians.

## Source-Only Semantic Estimator

Split every source environment into estimator folds A/B. For each trained
encoder, whiten latent values on fold A's source-mixture non-degenerate support.
Regress fold-A environment moments on the known signed factorial source design:

- task raw range: common label-predictive `Cov(Z,Y)` intercept;
- relation raw range: coefficients of `Cov(Z,Y|e)` on relation design;
- mean raw range: coefficients of `E[Z|e]` on mean design;
- covariance raw range: spectral ranges of coefficients of `Cov(Z|e)` on
  covariance design.

Residualize ranges in the frozen order task, relation, mean, covariance; the
orthogonal support complement is residual. Projectors are mapped back to the
original latent inner product for risk attribution. Fold B evaluates operator
transfer and moment response. Repeat with A/B exchanged and aggregate without
using target data.

Required validity diagnostics:

- symmetry, idempotence, pairwise orthogonality, support completeness;
- cross-fit operator transfer;
- principal-angle recovery against encoder images of known generating blocks;
- source intervention-label permutation (recovery must disappear);
- all legal semantic orders (conclusions must not depend materially on order);
- seed/lambda stability and bottleneck collapse/mixing.

If raw ranges overlap, oracle recovery is weak, permutation does not destroy
recovery, or conclusions change with order, mark the affected method/component
`SEMANTIC_DECOMPOSITION_NOT_IDENTIFIED`.

## Risk Accounting and Bound

For `w_k=P_k w`, compute source and target latent second moments and exactly
account for:

\[
R_T-R_S=\sum_k(w_k^T\Delta\Sigma_Zw_k-2w_k^T\Delta c_{ZY})
+2\sum_{k<l}w_k^T\Delta\Sigma_Zw_l.
\]

Interactions are reported directly and split equally between their two
components for Shapley-style presentation. The sum must close to the directly
evaluated population or empirical risk gap within numerical tolerance.

Predeclared shift budgets are derived from the SCM intervention limits, before
evaluating target instances:

\[
|R_T-R_S|\leq\sum_k(2\rho_{c,k}\|w_k\|+
\rho_{\Sigma,kk}\|w_k\|^2)+2\sum_{k<l}
\rho_{\Sigma,kl}\|w_k\|\|w_l\|.
\]

An Omega-to-component claim additionally requires an independently verified
operator bridge. Otherwise its status is `NO OMEGA-TO-COMPONENT BOUND`.

## Registered Decision Rule

A method controls a shift type only if, for a lambda selected solely by source
risk plus the method's source objective:

- its own source operator decreases;
- corresponding effective head energy decreases at least 25%;
- covered-target transport improves at least 25% relative to paired ERM;
- source-risk cost is at most 10% of paired ERM target excess;
- at least 8 of 10 seeds agree in direction.

Failure labels are `operator blind`, `source-design blind`, `semantic mixing`,
`task-signal collapse`, `cross-subspace interaction`, `optimization
instability`, and `out-of-family target`.

## Execution Gates

`DESIGN_GATE -> MVP_GATE -> CODE_GATE -> ten-seed run -> RESULT_GATE`.
An independent `codex exec --ephemeral --sandbox read-only` instance issues the
verdict. `VETO` stops execution. `REVISE_ONCE` permits one bounded repair; a
second failure at the same gate stops the experiment. ARS is not exposed in the
current environment, so this local independent supervisor is the recorded
fallback.
