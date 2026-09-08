# 3B Response Modules Report

## Verdict

`3B-LOW-RANK-BUT-NONSEMANTIC`

The primary object is the shift-column response `u_s = H_S^(-1/2) g_s` and
its intrinsic Gram matrix. Discovery did not use mechanism labels, exposure,
regularizer identity, or target-risk labels. Labels enter only in the
post-hoc audit.

## Algebraic results

- Main relevant response rank: `5`.
- Main Gram rank: `5`.
- Blind selected module count: `6`.
- Rotation mean assignment ARI: `1.0000`.
- Nuisance module-count stability: `True`.
- One-to-two shortcut response-rank increase: `True`.
- Bootstrap co-membership stability: `0.8302`.
- Post-hoc ARI/NMI (not used in selection): `0.2284` / `0.5300`.
- Held-out mixed reconstruction median: `1.989e-18`.
- Held-out mixed reconstruction pass: `True`.
- First-order-null probes filtered: `True`.

## Classification

The benchmark has a low-dimensional response geometry and robust predictor
coordinate invariance, but blind assignments do not meet the pre-registered
semantic alignment threshold. This is a `LOW-RANK-BUT-NONSEMANTIC` result,
not evidence for a causal module decomposition. The selected count is an
estimator diagnostic and is not a latent factor count.

## Formal status

The Lean extension proves abstract direct-sum uniqueness, overlap
non-identifiability, linear response additivity, and Gram invariance under an
inner-product-preserving linear map. Numerical ranks and cluster assignments
remain `diagnostic`.

## Interpretation boundary

The result supports stable risk-response subspaces in this controlled
population benchmark. It does not identify arbitrary latent semantic truth.
The identical-image and partial-overlap counterexamples are explicit
non-identifiability controls. Module count is not a latent factor count, and
no causal or regularizer-control conclusion is made.

## Scope limits

First-order-null nuisance shifts are filtered from primary discovery and are
not silently treated as semantic modules. Mixed shifts are validated with
moment-level additive construction; nonlinear parameterization interaction is
not forced into an additive claim.
