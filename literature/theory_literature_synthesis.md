# Theory literature synthesis

## 1. Mathematical families

The literature supports a structural taxonomy: (i) environment-risk consistency (`Var_e R_e`, worst-group risk); (ii) invariant optimality/conditional mechanisms; (iii) optimizer statistics (gradient covariance); (iv) feature moments and IPMs; (v) RKHS norms/mean embeddings; (vi) capacity, margin, spectral and Jacobian control; (vii) DRO/minimax uncertainty sets; (viii) causal invariance and anchor shifts; (ix) bilevel/meta objectives; and (x) posterior/PAC-Bayes complexity. Several are alternative estimators of the same high-level shift principle, but they are not interchangeable in a theorem.

Recent verified additions sharpen the map rather than overturn it: Lai & Wang (2024) reinterpret the IRM gradient penalty as a TV functional; Wang et al. (2024) give a training-domain-count lower-bound perspective; Cao & Chen (2024) use PAC-Bayes for Mixup-induced extrapolation; Shui et al. (2022) provide an explicit INV framework for several invariance notions. These papers all state additional structure instead of claiming assumption-free source-only transfer.

## 2. Dominant proof architectures

`risk decomposition -> discrepancy / complexity / robust dual / stability -> concentration -> oracle or approximation term` is the dominant pattern. DA uses `source + discrepancy + joint error`; DG replaces target discrepancy by a meta-distribution or admissible-family assumption; DRO replaces discrepancy by a chosen uncertainty set; RKHS and norm methods make the empirical-to-population step explicit. Stability controls sample perturbations, not domain shift, unless paired with a shift term.

## 3. Tractability ranking

Most natural: ERM, finite-group DRO, convex regularized ERM, RKHS/IPM penalties, linear anchor and risk-variance models. Intermediate: deep norm/margin control and representation matching, where class complexity is manageable but conditional alignment is not. Hard: bilevel/meta-learning. Usually idealized or restricted: IRMv1 and Fishr, because their objects are derivatives/covariances of an inner optimizer and require derivative-class concentration, smoothness, and optimization assumptions.

## 4. IRMv1 answer

IRMv1 differentiates empirical/population risk with respect to the classifier variable `w` and penalizes the squared gradient at a fixed scale. Reviewed theory generally studies ideal shared-optimality constraints, linear/structural models, consistency/identifiability, or counterexamples. The exact deep training dynamics are not the object of a broadly applicable finite-sample source-only target-risk theorem in this corpus. The right conclusion is therefore “gradient penalties are normally abstracted upward,” not “IRM has no theory.”

## 5. How abstraction level is chosen

Authors repeatedly: analyze population rather than optimizer trajectories; replace a deep model by `f in F`; restrict to linear/convex/RKHS settings; assume exact optimization or add an error; and retain an oracle joint/approximation term. When no defensible target family exists, they prove impossibility instead of claiming transfer.

## 6. Candidate levels for our future theory search (no new theorem proposed)

| Level | Objects | Expresses | Cannot express cleanly | Tractable bounds | Source-only natural? |
|---|---|---|---|---|---|
| A risk-vector | `(R_e(f))`, dispersion, weights | ERM, V-REx, GroupDRO | gradients, feature conditionals | concentration over environments | with meta-distribution/coverage |
| B function + discrepancy | `F`, loss/IPM discrepancy | MMD, CORAL, DA/DG penalties | optimizer dynamics | VC/Rademacher/RKHS + discrepancy | only with source family assumption |
| C robust uncertainty | `U(P_S)`, `sup_Q R_Q` | DRO, group/moment robustness | methods lacking a shift set | duality + robust uniform convergence | yes conditional on U |
| D RKHS/operator | kernel mean/conditional operators, norm | MMD, kernel DG, moment penalties | deep causal/optimizer details | Hilbert-space concentration | yes for source-defined family |
| E causal conditional | invariant `P(Y|X_S)` / SCM family | ICP, anchor, stable mechanisms | arbitrary alignment and IRMv1 dynamics | finite candidate tests + structural results | yes under interventions/coverage |

These are candidate *representation levels*, not a proposed unified theorem. The evidence favors a layered system with explicit fidelity labels and unavoidable oracle/misspecification terms.
