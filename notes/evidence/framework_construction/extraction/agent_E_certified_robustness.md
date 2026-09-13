# Agent E — Certified adversarial robustness

## External search and selected works

Selected: Cohen, Rosenfeld & Kolter (2019), *Certified Adversarial Robustness via Randomized Smoothing*, DOI [10.48550/arXiv.1902.02918](https://doi.org/10.48550/arXiv.1902.02918); Wong & Kolter (2018), *Provable Defense against Adversarial Examples via the Convex Outer Adversarial Polytope*, DOI [10.48550/arXiv.1711.00851](https://doi.org/10.48550/arXiv.1711.00851); Salman et al. (2019), *A Convex Relaxation Barrier to Tight Robustness Verification*. Chosen for contrasting probabilistic and deterministic certificates.

## Framework cards

### Randomized smoothing
- **Problem:** certify a perturbation radius for a black-box/non-smooth classifier.
- **Primitives:** perturbation set/norm, base classifier, smoothing distribution.
- **Intermediate:** class-probability margin under the smoothed classifier.
- **Backbone:** Neyman–Pearson lemma converts probability bounds into a certified radius; Monte Carlo confidence intervals make the certificate computable.
- **Fidelity:** certificate is exact for the smoothed classifier, not the unsmoothed model at every point.
- **Failure:** loose confidence bounds, distribution mismatch, or radius outside the smoothing geometry.

### Convex relaxation certificates
- **Primitives:** network constraints and perturbation polytope.
- **Intermediate:** upper/lower bounds from a convex outer polytope or linear relaxation.
- **Backbone:** if relaxed margin is positive, no allowed perturbation changes the label.
- **Fidelity cost:** relaxation gap; tighter certificates cost computation.

## Construction lessons

Certification frameworks introduce a certificate that is sufficient for a concrete guarantee, then make uncertainty geometry and computability explicit. They distinguish validity from tightness and report relaxation/statistical confidence separately. The transferable lesson is a certificate sufficiency theorem with a visible gap term; OOD should not call a diagnostic statistic a certificate unless an implication is proved.

## Cross-field summary

Recurring patterns: perturbation model first; certificate object second; generic certificate theorem; confidence/relaxation layer last. Primitive objects are chosen for sound implication to the query. Impossibility appears as zero/negative margin or relaxation barrier. Do not transfer smoothing radii to domain shift without a justified perturbation metric.
