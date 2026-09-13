# Frozen Explorer Cards

These cards were produced independently before synthesis. They are intentionally compact and preserve disagreements.

## A. Risk / Function-Space

- **Core:** represent a predictor by the domain-risk function `g_f(P)=R_P(f)` and source risk vector; attach typed modules rather than forcing a lossless unifier.
- **Target:** expected fresh-domain risk under `P~Pi`, or worst-case risk in declared `U(S)`; arbitrary targets are excluded.
- **Bridges:** ERM/V-REx/finite GroupDRO are exact risk-level functionals; IPM, IRM, IRMv1, Fishr and MLDG require translation or population abstraction.
- **Decomposition:** within-domain estimation + source-to-meta deviation + fresh-domain fluctuation; robust sibling uses a support function.
- **Failure:** shortcut and causal predictors can have identical source risks/statistics but opposite target risk.
- **Minimal theorem:** two-level uniform convergence with an explicit domain complexity and fresh-domain tail term.

## B. Conditional / Causal

- **Core:** model `X=(C,S)` and a stable `P_e(Y|C)` or SCM/intervention family; feature marginal alignment is not conditional invariance.
- **Target:** `Q_Pi` or a declared causal/DRO family with overlap, faithfulness and intervention coverage.
- **Bridges:** ICP/anchor/ideal IRM are theorem-level only under structural assumptions; IRMv1 and Fishr are surrogates with derivative and optimization remainders.
- **Decomposition:** covariate shift term plus conditional-label shift term; the latter is zero only under verified invariance.
- **Failure:** equal source risks and marginal moments coexist with opposite target conditionals (Zhao-style construction).
- **Minimal theorem:** source-only two-level bound plus explicit conditional remainder or SCM identification lemma.

## C. Discrepancy / IPM / Operator

- **Core:** typed signed-measure contrasts: marginal RKHS means, conditional operators, derivative-score operators, and robust support functions.
- **Target:** source-defined operator uncertainty set around the source hull, with separate marginal and conditional radii.
- **Bridges:** RKHS/IPM duality and DRO support are exact for declared classes; gradient/Fishr-to-conditional bridges are open outside restricted smooth models.
- **Decomposition:** source risk plus dual norm times target radius and source operator diameter, plus conditional, approximation and optimization slacks.
- **Failure:** marginal operators can be identical while conditional labels flip.
- **Minimal theorem:** RKHS loss-class target bound with explicit `rho_m + rho_cond` and a regularizer-to-radius lemma.

## D. Robust / Uncertainty-Set

- **Core:** declare `U(S)` first and target `Q_U(f)=sup_{Q in U(S)}R_Q(f)`; separate this from expected-domain risk.
- **Objects:** source risk vector and support function `sup_w w^T R(f)` for finite mixtures, or f-divergence/Wasserstein balls.
- **Bridges:** convex/transport duality is exact conditional on metric, radius and support; IRMv1/Fishr do not define `U` without a separate theorem.
- **Minimal theorem:** uniform robust estimation bound via simplex Lipschitzness and per-domain Rademacher concentration.
- **Failure:** out-of-set target misspecification is uncontrolled; large radius makes the bound vacuous.

## E. Variational / Functional-Translation

- **Core:** treat every regularizer as a functional on population risks or their derivatives, and prove objective-to-functional translations before invoking OOD bounds.
- **Objects:** `J(f)`, risk variance, classifier-gradient field `q_e(f)`, gradient covariance `K_e(f)`, IPM seminorms, and robust support function.
- **Bridges:** IRMv1-to-TV functional is known under smoothness/coarea-style assumptions; V-REx and DRO have direct variational forms; Fishr and MLDG translations remain conjectural or restricted.
- **Target:** a typed outer error decomposition; no claim that all functionals induce the same structural property.
- **Failure:** parameterization-dependent derivatives can be equal across worlds while target conditional mechanisms differ.
- **Minimal theorem:** a restricted linear/smooth objective-to-functional lemma with explicit derivative-class concentration and optimization error.
