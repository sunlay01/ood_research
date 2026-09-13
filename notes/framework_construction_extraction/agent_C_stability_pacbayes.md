# Agent C — Algorithmic stability and PAC-Bayes

## External search and selected works

Selected foundational works: Bousquet & Elisseeff (2002), *Stability and Generalization*; Hardt, Recht & Singer (2016), *Train Faster, Generalize Better: Stability of Stochastic Gradient Descent*, DOI [10.48550/arXiv.1509.01240](https://doi.org/10.48550/arXiv.1509.01240); McAllester (1999), *PAC-Bayesian Bounds for the Generalization Error*; Germain et al. (2020), PAC-Bayes review. Chosen because they introduce reusable intermediate properties and generic theorems.

## Framework cards

### Uniform/expected stability
- **Problem:** objective complexity alone does not explain algorithm-dependent generalization.
- **Primitives:** algorithm `A`, sample replacement operator, loss and hypothesis space.
- **Intermediate:** replace-one stability `beta` (uniform, hypothesis, or on-average).
- **Backbone:** stability-to-generalization theorem; regularization/SGD lemmas bound `beta`, then the generic theorem gives expected or high-probability risk gap.
- **Modularity:** any algorithm with a stability lemma reuses the same outer theorem.
- **Assumptions:** bounded/Lipschitz loss, convexity or smoothness for SGD, step-size and iteration controls.
- **Failure:** stability is not a shift model; it controls source-distribution generalization only.

### PAC-Bayes
- **Primitives:** prior `P`, posterior `Q`, data-dependent empirical risk, KL divergence.
- **Intermediate:** change-of-measure complexity `KL(Q||P)` plus exponential moment bound.
- **Backbone:** PAC-Bayes inequality converts empirical Gibbs risk and KL into population risk.
- **Algorithm-specific bridge:** a learner or posterior construction supplies `Q`; the outer theorem remains unchanged.
- **Fidelity cost:** stochastic/randomized predictor and prior choice; target shift requires an additional transfer term.

## Construction lessons

Stability and PAC-Bayes demonstrate a clean three-layer design: algorithm-specific lemma, generic intermediate certificate, reusable risk theorem. The certificate is useful because it is decision-relevant, compositional, and estimable or upper-bounded. Structural assumptions and finite-sample concentration are stated separately. For OOD, the transferable principle is modularity; source stability or KL cannot silently become target-shift identification.

## Cross-field summary

Recurring patterns: choose a property invariant across algorithms; prove local algorithm-to-property lemmas; reserve one backbone theorem for risk. Primitive objects are selected for closure under perturbation/change of measure. Impossibility is represented by large or unbounded certificate. Safe transfer: demand an algorithm-specific bridge and keep target-family terms outside the certificate.
