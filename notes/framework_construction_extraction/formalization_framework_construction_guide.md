# Formalization framework-construction guide

This dossier synthesizes the six cross-domain extraction reports together with the OOD bridge evidence. It is a construction guide, not a final OOD framework.

## I. Recurring anatomy

Mature frameworks repeatedly follow: scientific uncertainty/model → primitive representation → structural relations → intermediate certificate or identification object → backbone theorem → query/risk, with separate estimation and computation layers. Strong frameworks also specify a failure output: non-identifiability, least-favorable instance, relaxation gap, target discrepancy, or confidence interval.

## II. Choosing primitive objects

Objects are chosen because they preserve decision/query-relevant information, define the uncertainty class, admit a complete calculus, or support a reusable theorem. Causal graphs preserve intervention semantics; stability coefficients preserve replace-one sensitivity; KL supports change of measure; contamination neighborhoods define adversaries; certificates are sufficient statistics for a concrete robustness query. Elegance or shared notation alone is not evidence.

## III. Structural, estimation, computation

Successful theories separate structural validity (SCM, neighborhood, perturbation set, hypothesis class), finite-sample estimation (concentration, confidence intervals, Rademacher/KL terms), and optimization/certification computation (solver error, relaxation gap, Monte Carlo error). Blending these layers obscures what is actually proved.

## IV. Intermediate-object design principles

Useful intermediates have a sound implication to the target, compose across algorithms, expose assumptions, and admit a falsification or tightness test. They may be a stability coefficient, deficiency, KL radius, least-favorable distribution, support function, contraction factor, or probability margin. Each is role-specific; a discrepancy is not an identifiability defect and a certificate is not merely a diagnostic statistic.

## V. From statistic to framework

A single statistic becomes a framework only when it has a declared domain of validity, relations to primitives, a generic theorem, an estimation/computation interface, and an impossibility or gap semantics. A collection of quantities without these relations is a taxonomy. A representation system additionally defines mappings for new algorithms and queries.

## VI. Backbone theorem taxonomy

| Backbone | Construction role |
|---|---|
| completeness/identifiability | decides whether a query is determined and returns failure otherwise |
| stability-to-risk | lets many algorithms plug in through local stability lemmas |
| change-of-measure/PAC-Bayes | converts empirical object plus complexity into population risk |
| minimax/least-favorable | turns uncertainty neighborhoods into worst-case guarantees |
| certificate theorem | converts a sufficient statistic into a radius/error guarantee |
| duality/support | turns robust objectives into computable primal/dual forms |
| contraction | reduces raw shift after a structural channel assumption |
| decomposition | isolates distinct shift, approximation and irreducible terms |
| experiment comparison | formalizes information preservation for all decisions |

## VII. Modularity

The strongest modular shape is `algorithm-specific object Ω → local lemma → generic certificate/bridge B → backbone theorem → query`. Multiple algorithms share the B→risk theorem only if B has the same semantics, not merely the same symbol. Algorithm-specific translation, target uncertainty, and optimization errors remain explicit.

## VIII. Information preservation and abstraction

Decision theory supplies the strict test: if two representations are equivalent up to small deficiency, every downstream decision risk is close; otherwise the discarded distinction can matter. Causal transportability preserves intervention-relevant structure and returns non-identifiability when absent. For OOD, abstraction should quotient only information proven irrelevant to the declared target family; marginal alignment cannot quotient conditional labels without a theorem.

## IX. Robustness formalization patterns

Contamination neighborhoods, DRO balls, adversarial perturbation sets and OOD target families share the logic “declare uncertainty set → define worst-case/query risk → derive support/minimax/certificate theorem.” They differ in geometry and scientific meaning: contamination models mixture corruption, DRO uses divergence/transport, adversarial certification uses local perturbations, and OOD families encode domain mechanisms or coverage. Radius misspecification is a first-class error in all cases.

## X. Algorithm-to-theory translation

Exact translations include finite-group support objectives, fixed-set DRO, and smoothed-classifier probability certificates. Stability/SGD, empirical PAC-Bayes posteriors, IRMv1-TV and MLDG require local lemmas or variational/Taylor replacements. Every lossy translation needs a named approximation, stationarity, relaxation or solver term.

## XI. Mathematical-friendliness ranking

Most useful proof enablers, in combination, are: decomposition and duality first; then identifiability/completeness and contraction; then convexity/minimax; then concentration/finite complexity; finally variational representations. Elegance without a sound implication or finite certificate is insufficient.

## XII. Adequacy checklist for later GPT-6 design

- Scientific fidelity: does the object still describe the claimed domain shift?
- Information sufficiency: can source-indistinguishable worlds have different target risk?
- Translation: can actual algorithms map in with explicit fidelity labels?
- Backbone: is there one reusable structural theorem?
- Modularity: can a held-out algorithm enter via a local lemma?
- Estimation: are all empirical terms measurable and finite?
- Computation: are optimization, confidence and relaxation gaps explicit?
- Impossibility: is non-identifiability represented as a result, not hidden?
- Out-of-sample test: does a new regularizer reuse the theorem?
- Novelty: is there a theorem-level distinction beyond renamed notation?

## XIII. Anti-patterns

Reject `L+lambda Omega` syntax as a framework, tuple-of-everything representations, framework-first reasoning, target-first trivialization, generic concentration mislabeled as OOD theory, unproved algorithm-to-property bridges, hidden target information, silent optimization abstraction, universal notation over incompatible assumptions, and frameworks with no completeness/falsification notion.

## XIV. Cross-field construction lessons

| Field | Primitive | Intermediate | Backbone | Final target | Main lesson |
|---|---|---|---|---|---|
| OOD/DG | environments, risks, SCM/target family | discrepancy, mechanism, contraction, uncertainty set | transfer/concentration/contraction | target or robust risk | keep target-family and conditional residuals visible |
| causal transportability | DAG/SCM/selection diagram | identifiable functional/defect | complete do-calculus | transported causal query | query-first, explicit failure |
| stability | algorithm/sample/loss | stability coefficient | stability-to-generalization | population risk | local algorithm lemma + generic theorem |
| PAC-Bayes | prior/posterior | KL/information radius | change of measure | Gibbs/population risk | separate learner representation from class complexity |
| robust statistics | nominal law/neighborhood | least-favorable law/modulus | minimax theorem | worst-case risk | formalize uncertainty before estimator |
| decision theory | experiment/decision rule | deficiency/randomization | comparison theorem | all decision risks | preserve decision-relevant information |
| certified robustness | perturbation set/model | margin/probability/relaxation certificate | certificate implication | robust radius/error | soundness, tightness and computability are separate |
| DRO | metric/divergence set | support/dual variable | primal-dual equivalence | worst-case risk | geometry and misspecification are assumptions |

## INSTRUCTIONS FOR GPT-6 FRAMEWORK DESIGN

GPT-6 will receive two evidence bases:

**Evidence A — OOD bridge evidence:** concrete `Ω_j → B_{j,1} → … → R_T` chains from the bridge-extraction round.

**Evidence B — framework-construction evidence:** how mature fields choose primitives, structural relations, theorem backbones, abstraction layers, error terms and falsification criteria.

Use both evidence bases to generate and compare candidate architectures. First identify which bridge objects and construction patterns are genuinely shared; only then propose a candidate. Require an out-of-sample algorithm test, an information-sufficiency counterexample, and explicit estimation/computation layers.

Do not copy an adjacent-field object merely because its framework is elegant. Transfer the construction principle, not automatically the mathematical object. Do not treat a target-family assumption, causal graph, KL term, stability coefficient or robustness radius as interchangeable without proving semantic equivalence.

Final status: **CONSTRUCTION-GUIDE-READY**.
