# Agent A — OOD/DG native framework anatomy

## Selected works

Shui, Wang & Gagné (2022), *A Principled Approach to Domain Generalization*; Wang & Veitch (2022), *A General Framework for Invariant Representation Learning*; Lai & Wang (2024), IRM-TV; Wang et al. (2026), Tri-Space; Rothenhäusler et al. (2021), Anchor Regression. Repository evidence: `notes/evidence/ood_bridges/synthesis/validated_bridge_chains.md` and cited detailed cards.

## Framework cards

### Shui et al. 2022
- **Problem:** bound unseen-domain error under invariant latent conditional structure.
- **Primitives:** source/target latent conditionals, representation channel, total variation, nearest-source radius.
- **Intermediate:** Dobrushin contraction coefficient and conditional-TV defect `kappa`.
- **Backbone:** exact decomposition + strong data processing/contraction yields `BER_T <= average BER_S + kappa + alpha_TV(Phi) epsilon`.
- **Boundary:** requires latent invariance and a source-nearest target family; does not cover arbitrary OOD.

### Wang & Veitch 2022 / Anchor Regression
- **Problem:** stable prediction under specified interventions.
- **Primitives:** SCM/anchor variables, residual covariance and intervention class.
- **Intermediate:** a primal-dual robustness functional.
- **Backbone:** robust-risk equivalence under linear SCM assumptions.
- **Boundary:** graph, linearity and intervention coverage are part of the framework, not cosmetic assumptions.

### Lai & Wang 2024
- **Problem:** interpret IRMv1 mathematically.
- **Primitives:** risk as a function of classifier variable and regularity class.
- **Intermediate:** total-variation functional obtained by variational translation.
- **Backbone:** functional theorem, not exact deep optimizer analysis.
- **Boundary:** coarea/smoothness conditions and translation gap.

### Tri-Space
- **Problem:** separate invariant, spurious and variant latent contributions.
- **Primitives:** direct-sum latent spaces and structural label mechanism.
- **Intermediate:** localized discrepancy terms.
- **Backbone:** unique decomposition plus target-risk bound.
- **Boundary:** latent identifiability and deterministic/restricted labels.

## Construction lessons

Native OOD frameworks become frameworks when they specify a scientific environment family, a minimal representation preserving the claimed mechanism, a structural theorem, and explicit residuals. Their extensions add new assumptions or a new lemma rather than merely appending another statistic. The main fidelity cost is that these assumptions narrow the admissible target set.

## Cross-field summary

Recurring architecture patterns: target/query first; mechanism or family next; exact decomposition; theorem backbone; residual accounting. Distinctive pattern: target-family and contraction terms are first-class. Primitive objects are chosen because they make the stated shift mathematically visible. Identifiability is encoded by invariance/rank or explicit counterexamples. Estimation and optimization are separate from structural claims. Do not copy latent notation into OOD without preserving its assumptions; transfer the discipline of an explicit target family and residual.
