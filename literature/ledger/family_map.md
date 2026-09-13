# OOD/DG theory family map

This map is the first retrieval layer. Read it before the paper ledger; each family records the mathematical language and its tractability price, not an algorithm name.

## Environment-risk / risk-vector
- Core object: `(R_1(f),...,R_E(f))`, variance, weighted or worst-group risk.
- Typical target: observed-group max risk or extrapolation proxy.
- Proof tool: concentration over environments, convex weighting, algebraic risk decompositions.
- Tractable: ERM, V-REx, GroupDRO and finite-group robust objectives.
- Cost/exclusion: finite-group coverage; no representation of arbitrary unseen domains, gradients, or conditional mechanisms.
- Papers: Vapnik1998; Krueger2021; Sagawa2020; Blanchard2021.

## Discrepancy / IPM / RKHS
- Core object: loss-class discrepancy, `HΔH`, MMD, kernel mean embedding, or feature moments.
- Typical target: target risk or source-target risk gap plus joint/approximation error.
- Proof tool: triangle inequality, RKHS duality, reproducing property, concentration.
- Tractable: DA discrepancy bounds, MMD, CORAL and domain-classifier penalties.
- Cost/exclusion: target samples or an explicit source-family assumption; alignment need not imply conditional-label alignment.
- Papers: BenDavid2010; Mansour2009; Gretton2012; Sun2016; Ganin2016; Zhao2019.

## Causal / conditional invariance
- Core object: invariant `P(Y|X_S)` or an SCM/intervention family.
- Typical target: stable causal predictor risk under specified interventions.
- Proof tool: conditional independence, d-separation, identifiability and finite candidate testing.
- Tractable: ICP, anchor regression, causal invariant representation arguments.
- Cost/exclusion: graph, intervention, faithfulness/coverage assumptions; arbitrary alignment and optimizer statistics are outside scope.
- Papers: Peters2016; Rothenhaeusler2021; WangVeitch2022.

## DRO / uncertainty sets
- Core object: `sup_{Q in U(P)} R_Q(f)` with f-divergence or Wasserstein `U`.
- Typical target: robust population or empirical worst-case risk.
- Proof tool: convex duality, transport duality, uniform convergence.
- Tractable: Wasserstein DRO, f-divergence/group DRO, robust adversarial training.
- Cost/exclusion: metric, radius, support and ambiguity-set correctness; no guarantee outside `U`.
- Papers: Esfahani2018; Sinha2018; Duchi2021; Sagawa2020.

## PAC-Bayes / information complexity
- Core object: posterior `Q`, prior `P`, KL or mutual information/information density.
- Typical target: Gibbs/source-to-target risk or OOD risk gap.
- Proof tool: change of measure, variational inequality, concentration and data processing.
- Tractable: posterior-level transfer, stochastic learners and SGLD analyses.
- Cost/exclusion: prior/information-radius and shift terms may be oracle or non-operational; not exact deep optimizer coverage.
- Papers: Germain2020; Rivasplata2018; Cao2024Mixup; Liu2024InfoOOD; Liu2025InfoSGLD.

## Stability / norm and margin complexity
- Core object: uniform stability, spectral/Frobenius/path norms and margins.
- Typical target: same-distribution source generalization, optionally combined with shift term.
- Proof tool: replace-one stability, symmetrization, contraction and concentration.
- Tractable: regularized ERM, spectrally normalized networks, randomized stable algorithms.
- Cost/exclusion: capacity or stability says nothing about admissible target distributions by itself.
- Papers: Bousquet2002; Bartlett2017; Miyato2018; Rivasplata2018.

## Invariance objectives and latent decompositions
- Core object: population invariance slot `INV`, direct-sum latent spaces, invariant/spurious/variant components.
- Typical target: unseen-domain target risk with localized shift terms.
- Proof tool: strong data processing/TV contraction, unique decomposition, localized discrepancy and concentration.
- Tractable: representation alignment, domain augmentation and selected IRM-like population constraints.
- Cost/exclusion: bounded loss, latent structure, deterministic invariant labels or conditional-density assumptions; not exact arbitrary deep training.
- Papers: Arjovsky2019; Kamath2021; Shui2022; Wang2026TriSpace.

## Variational / TV objective translation
- Core object: total variation of risk as a function of classifier variable.
- Typical target: population OOD conditions or minimax-TV objectives.
- Proof tool: variational analysis, coarea/regularity arguments and measure-theoretic duality.
- Tractable: an explicit functional interpretation of IRMv1's classifier-gradient penalty.
- Cost/exclusion: translation is not a finite-sample theorem for deep SGD and does not unify Fishr, GroupDRO or generic DRO.
- Paper: Lai2024.

## Bilevel / meta-learning
- Core object: update map and meta-train/meta-test loss.
- Typical target: cross-environment validation or approximate DG objective.
- Proof tool: Taylor expansion, linear/convex analysis and stationary-point arguments.
- Tractable: MLDG-style one/few-step meta objectives.
- Cost/exclusion: optimizer dependence and second-order terms; exact deep finite-sample guarantees are uncommon.
- Paper: Li2018.

## Lower bounds / impossibility / calibration
- Core object: non-identifiability, domain-count uncertainty, calibration/density-ratio function classes.
- Typical target: minimax lower bound, impossibility, or risk under shift beyond covariate shift.
- Proof tool: indistinguishable constructions, minimax probability, multicalibration and change of measure.
- Tractable: tests of whether a proposed representation can support a non-vacuous theorem.
- Cost/exclusion: these papers do not supply an upper-bound algorithmic unifier.
- Papers: Rosenfeld2021; Gulrajani2021; Wang2024Lost; Wu2024Multicalibration.
