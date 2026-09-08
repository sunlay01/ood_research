# C010: ERM--IRMv1 Operator, Coverage, Blindness, and Error Accounting

## Status

`PARTIAL / PROVED_IN_SCALAR_POPULATION_SUBCLASS / HIGH-RISK_LITERATURE_COLLISION`

## Scope

This Claim is limited to population squared loss in the partially observed scalar Gaussian SCM:

\[
C\sim N(0,1),\quad U=LC+\xi,\quad Y=\beta C+\epsilon_Y,\quad A=rC+\eta.
\]

It assumes centered nuisance, common source nuisance variance `sigma_A^2`, and a zero-intercept affine predictor `f_w=uU+aA`. The target may change nuisance relation, mean, and covariance, while the task equation remains fixed. It is not a theorem about arbitrary neural IRMv1 implementations.

## ERM

### Operator: `exact equality`

Population ERM imposes only mixture stationarity:

\[
\nabla R_S(w)=\sum_e\pi_e\nabla R_e(w)=0.
\]

It does not impose `nabla R_e(w)=0`, radial optimality, or a nuisance coefficient constraint in every environment. In the two-source diagnostic `r=(0.8,0.3)`, ERM has zero mixture gradient while its two scale derivatives are `+0.05640` and `-0.05640`; the tangential gradient norm is nonzero in both environments.

### Restricted positive control: `exact calculation`

For centered symmetric source relations `(-r,+r)` with common nuisance variance, the source moment is block diagonal between `U` and `A`. Population ERM has `a=0`, hence its risk is unchanged under any relation, nuisance-mean, or nuisance-covariance intervention in this SCM. This is a source-symmetry result, not an ERM OOD guarantee in general.

### Blindness: `counterexample`

When source mixture moments make `A` predictive, ERM uses `a != 0`. C002-IRM supplies an especially strong instance: source ERM is also an IRMv1 zero, yet its target causal excess is `127/48` at an allowed relation sign flip.

## Standard Scalar-Scale IRMv1

### Actual operator: `exact equality`

For feature moment `Sigma_e`, feature-label moment `c_e`, and `g_e(w)=2(Sigma_e w-c_e)`, standard IRMv1 is

\[
\Omega_{\rm IRMv1}(w)=\frac1m\sum_e [w^Tg_e(w)]^2
{}=\frac4m\sum_e [w^T(\Sigma_ew-c_e)]^2.
\]

Thus it observes only the radial projection of each environment gradient onto the current predictor ray. The tangent component \((I-ww^T/\|w\|^2)g_e\) is not in this operator. It is neither full-gradient matching nor covariance alignment.

### Relation-response identity: `exact equality`

Write the unscaled radial response as

\[
q(r)=\mathbb E[(f_w(X)-Y)f_w(X)]=q_0+q_1r+q_2r^2,
\]

where

\[
\begin{aligned}
q_0&=(L^2+\sigma_\xi^2)u^2-L\beta u+\sigma_A^2a^2,\\
q_1&=a(2Lu-\beta),\\
q_2&=a^2.
\end{aligned}
\]

The IRMv1 derivative is `2q(r)`. Let `V_S` have source rows `(1,r_e,r_e^2)` and let `kappa_V=sigma_min(V_S)`.

### Controlled component and finite-penalty bridge: `conditional theorem`

If there are at least three distinct centered source relations, then `V_S` has full column rank and

\[
a^2=q_2\leq\frac{\sqrt{m\,\Omega_{\rm IRMv1}(w)}}{2\kappa_V}.
\]

This identifies the scalar effective nuisance coefficient as the controlled component. It is a source-design theorem, not a generic statement that IRMv1 controls all nuisance directions.

For a target changing only the nuisance mechanism, let `b_0` be the C001 residual with the nuisance coefficient set to zero and `Delta M=M_T-M_S`. If `A_Omega` is the square root of the preceding bound, then

\[
|R_T(w)-R_S(w)|\leq \|\Delta M\|_{\rm op}\left(2\|b_0\|A_\Omega+A_\Omega^2\right).
\]

Consequently,

\[
R_T(w)-R_C^*=R_S(w)-R_C^*+[R_T(w)-R_S(w)]
\]

is a conditional population generalization-error accounting. The first term is not controlled by IRMv1; observation error, source fit, and target geometry remain explicit.

### Zero-penalty positive result: `conditional theorem`

Under the same full-rank relation design, `Omega_IRMv1=0` gives `a=0`. Then `q_0=0` implies either `u=0` or

\[
u=\frac{L\beta}{L^2+\sigma_\xi^2}.
\]

Assuming `L beta != 0`, the strict source-fit condition `R_S(w)<R_S(0)` eliminates the zero predictor. The remaining `U`-only predictor is exactly insensitive to every allowed relation, nuisance-mean, and nuisance-covariance shift, although it still has irreducible task-observation excess relative to direct observation of `C`.

### Blind boundary: `counterexample`

With only two relation points, `V_S` has rank at most two and cannot identify the three response coefficients. The existing C002-IRM sources `r=(0.7,-0.1)` have a nonzero-nuisance zero-penalty branch `(u,a)=(1/4,5/6)`, which is also source ERM. Its failure is not due to the gradient-response observability diagnostic being zero; it is a scalar radial operator/design blind branch.

This condition is sufficient, not necessary. It does not imply every two-environment IRMv1 setting fails, nor that every three-environment neural IRMv1 run is robust.

## Exact Error Accounting

For a source-target pair, `risk_transport_components` fixes a sequential hybrid convention:

\[
R_T-R_S=E_{\rm relation}+E_{\rm mean}+E_{\rm covariance}.
\]

The terms telescope exactly; their individual values depend on the fixed order of replacing relation, then mean, then covariance. For any pre-specified orthogonal nuisance projector `P`,

\[
R_T-R_S=E^{\rm ctrl}(P)+E^{\rm blind}(I-P)+E^{\rm interaction}(P,I-P)
\]

is also exact. C010 uses `P=I` only after the full-rank scalar design proves that the entire scalar nuisance coefficient is controlled. In the two-source blind case, no source-only projector is inferred from IRMv1.

## Linear Representation Track

For `Z=BX` and `f=w^TZ`, define `theta=B^Tw`. Under any invertible latent transform `T`,

\[
B\mapsto TB,\qquad w\mapsto T^{-T}w,
\]

`theta`, risk, IRMv1 penalty, and the effective nuisance coefficient `w^TB_A=theta_A` are unchanged. Therefore no C010 conclusion may use `B_A` alone. A rank-deficient encoder can still retain nonzero `theta_A`; hiding a coordinate in a representation is not a nuisance-sensitivity result.

## Verification and Literature Boundary

`tests/test_intervention_linear.py` verifies all identities, the two-source blind branch, the three-source bridge, the transport bound, and representation reparameterization invariance. `python -m ood_repr_reg.erm_irmv1_report` emits the population audit.

Kamath et al. (2021) already establishes practical IRMv1 failure in simple population settings. Lai and Wang (2024) also reinterpret IRMv1's objective. This Claim is an internal mechanism decomposition and a source-design condition; it is not a novelty claim until a theorem-by-theorem collision audit against those and moment-alignment work is complete.
