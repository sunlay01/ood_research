# How OOD/DG bounds are proved

## Proof-language map

The recurring architecture is

`target risk` -> `exact decomposition or change-of-measure` -> `discrepancy/robustness/complexity` -> `empirical concentration` -> `oracle or approximation term`.

### 1. Domain-adaptation discrepancy

Ben-David et al. (2010, Theorem 2) and Mansour et al. (2009) use the triangle inequality on risks and an `H Delta H` (or loss-class discrepancy) term. In a common 0-1-loss form,

`R_T(h) <= R_S(h) + 1/2 d_{HDeltaH}(S,T) + lambda*`,

where `lambda* = min_h R_S(h)+R_T(h)`. The discrepancy is estimable only with target samples; `lambda*` is an oracle joint-error term. This is DA, not source-only DG.

### 2. Source-only DG / domain-of-domains

Blanchard et al. (2011) and Blanchard et al. (2021) treat domains as draws from a meta-distribution or define an admissible family. A typical route is (i) uniform convergence over environments for source risks and (ii) concentration of a domain-level statistic, yielding an unseen-domain or worst-family guarantee. The price is explicit assumptions on how source environments cover the meta-distribution; no-free-lunch results show these assumptions cannot be omitted.

### 3. Uniform convergence and complexity

VC/Rademacher/Gaussian complexity, symmetrization, contraction, and Bernstein/Hoeffding concentration control `sup_{f in F}|R(f)-Rhat(f)|`. Shift enters by applying the argument per environment, weighting environments, or adding a discrepancy class. These tools naturally analyze ERM, norm-bounded nets, finite-group DRO, and RKHS estimators, but not optimizer trajectories unless stability is separately proved.

### 4. PAC-Bayes

Change of measure plus a KL variational inequality gives a high-probability bound on Gibbs risk. DA versions add a domain-disagreement/IPM term (Germain et al. 2020). Rivasplata et al. (2018) is a separate PAC-Bayes + algorithmic-stability result, not a DA bound. Source-only DG requires a prior/domain model or a meta-distribution; the posterior is a certificate, not automatically the output of a deep optimizer. The KL and disagreement terms are source-estimable only under the chosen domain formalism.

### 5. Stability

Uniform stability bounds algorithmic generalization on the *same* distribution. For shifted domains one must add a change-of-distribution term; SGD stability alone does not control `R_T-R_S`. Regularization can improve stability, but the theorem remains distribution- and loss-dependent (Bousquet & Elisseeff 2002; Hardt et al. 2016).

### 6. Norm/margin/spectral complexity

Bartlett et al. (2017) and Neyshabur et al. (2018) bound source generalization using products of spectral norms, Frobenius norms, path norms, and margins. These bounds control capacity, not which target distributions are admissible. DG-specific use therefore combines them with a shift/discrepancy or robust-risk assumption.

### 7. RKHS/IPM

MMD is the RKHS norm of a difference of kernel mean embeddings (Gretton et al. 2012). Bounded kernels give concentration at rate `O(1/sqrt(n))`; conditional mean embeddings and RKHS norm control yield tractable discrepancy and estimation terms. The target-risk bound still has a joint/approximation term unless conditional invariance is assumed.

### 8. DRO

DRO bounds start with `sup_{Q in U(P)} R_Q(f)`. Convex duality turns f-divergence or Wasserstein balls into a regularized empirical objective (Duchi & Namkoong 2021; Esfahani & Kuhn 2018; Sinha et al. 2018). Generalization then controls empirical robust risk uniformly over `f` and the uncertainty set. The uncertainty set, metric, radius, and support assumptions are the shift model; changing them changes the theorem.

### 9. Impossibility/lower bounds

Zhao et al. (2019) show that marginal feature alignment can conflict with label predictability; Johansson et al. (2019) make the representation-identifiability issue explicit. DG impossibility constructions (e.g. Gulrajani & Lopez-Paz 2021's discussion and subsequent theory) show that arbitrary unseen-domain behavior cannot be inferred from finite source domains. These results explain why every non-vacuous source-only theorem contains a restricted family, causal invariance, coverage, or oracle term.

## Algorithm-specific results

The literature separates three cases:

* **Exact and tractable:** convex ERM/DRO, finite-group robust risk, and RKHS estimators. The training objective and theorem object coincide up to empirical concentration.
* **Restricted exact:** linear-Gaussian IRM, linear risk extrapolation, or anchor regression. The algorithm is analyzed under a small model class and explicit shift equations.
* **Surrogate/idealized:** deep IRMv1, Fishr, MLDG, and representation matching. Papers analyze a population constraint, Taylor surrogate, or induced function class; optimizer details are assumed exact or absorbed as error.

## Proof anatomy template (representative)

For Ben-David DA: target quantity is `R_T(h)`; Step 1 adds/subtracts `R_S(h')`; Step 2 takes a supremum over `h,h'` to obtain `HDeltaH`; Step 3 estimates the discrepancy with a domain classifier; Step 4 bounds source empirical risk by VC concentration; Step 5 leaves `lambda*` as irreducible joint error. Target samples are required for the discrepancy.

For finite-group DRO: target is `max_e R_e(f)`; Step 1 replace population group risks by empirical risks; Step 2 uniform-convergence over `F` (and, for learned weights, over the simplex); Step 3 optimize the empirical robust objective; Step 4 retain group coverage and approximation terms. No target data is needed when the desired target is one of the observed groups; unseen-domain claims need a family assumption.

For RKHS MMD: target is a risk difference or an IPM; Step 1 use the RKHS reproducing property; Step 2 bound mean-embedding estimation by bounded-kernel concentration; Step 3 apply loss-class/RKHS norm generalization; Step 4 retain conditional mismatch/joint error. Target data is needed for DA MMD, but not for source-source DG penalties.

For Wasserstein DRO: target is worst-case risk over `U`; Step 1 dualize the inner supremum; Step 2 control empirical dual objective uniformly; Step 3 select radius from concentration; Step 4 retain approximation from the chosen ball. Source-only operation is natural, but only for shifts inside `U`.

For REx/V-REx (Krueger et al. 2021, Sec. 2--3): Step 1 define an extrapolated environment as an affine combination of source risks; Step 2 maximize the affine risk over a bounded coefficient set (MM-REx); Step 3 observe that the resulting quadratic penalty is proportional to the variance of the source risk vector (V-REx); Step 4 in the linear-SEM analysis (Theorem 1) identify conditions under which the extrapolation objective selects invariant/causal directions. The slack is the restriction to a linear SEM and a chosen extrapolation set; the theorem is not a distribution-free finite-sample deep-network bound.

For the INV framework (Shui et al. 2022, Proposition 1 and Theorem 1): Step 1 impose feature-conditional invariance and bounded loss; Step 2 express test balanced error as a nearest-source conditional shift term plus a representation-dependent contraction term; Step 3 bound the latter through total variation/Jensen--Shannon and a Lipschitz or Jacobian control on the representation; Step 4 add finite-source estimation. The target-family and bounded-loss assumptions are explicit, and the bound becomes vacuous when the representation's TV contraction coefficient is large. This is a genuine unified invariance framework, but not an exact analysis of a deep optimizer.

For IRM-TV (Lai & Wang 2024, Theorems 3.1--3.11): Step 1 treat `R(w o Phi, rho)` as a function of classifier variable `w`; Step 2 identify the IRMv1 gradient norm as the TV-`ell_2` variation of that risk; Step 3 use variational/TV conditions to characterize global OOD objectives and admissible environment sets; Step 4 derive conditions for TV-`ell_1` or minimax variants to generalize. The price is functional regularity, measure/coarea conditions, and an environment-family assumption; the result is not a generic empirical-process theorem for SGD on deep IRMv1.

## What is source-estimable?

Source risks, source-source risk variance, source feature moments, RKHS discrepancies between sources, and empirical robust objectives are source-estimable. Target discrepancies, target risk, and joint error are not source-estimable without assumptions or target samples. Optimization error is usually an additional abstract term rather than a derived quantity.
