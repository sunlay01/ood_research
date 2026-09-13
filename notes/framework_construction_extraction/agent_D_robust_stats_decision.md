# Agent D — Robust statistics and decision theory

## External search and selected works

Crossref search returned Huber, *Robust Statistics* (1983), DOI [10.1137/1025032](https://doi.org/10.1137/1025032); Chen, Gao & Ren, *A General Decision Theory for Huber's ε-Contamination Model* (2016); Le Cam/Torgersen, *Comparison of Statistical Experiments* (1991/1992). These were selected for explicit neighborhoods, least-favorable distributions, minimax risk, and information comparison.

## Framework cards

### Huber contamination / robust decision theory
- **Pre-formalization difficulty:** “robust to outliers” lacks a quantified adversary and risk criterion.
- **Primitives:** nominal distribution `P`, contamination neighborhood `(1-epsilon)P + epsilon Q`, decision rule and loss.
- **Intermediate:** least-favorable distribution, modulus of continuity, or minimax testing problem.
- **Backbone:** minimax risk and optimal decision rule over the neighborhood; least-favorable distributions reduce an infinite uncertainty class to a hard boundary instance.
- **Assumptions:** contamination fraction, parameter class, loss, identifiability and regularity.
- **Failure:** misspecified radius or parameter class; minimax risk can be large and informative estimation impossible.

### Statistical experiments / deficiency
- **Primitives:** experiment (Markov kernel from parameter to observations), decision rules, risk functions.
- **Intermediate:** deficiency / randomization distance between experiments.
- **Backbone:** comparison theorem: small deficiency means every decision problem in one experiment can be simulated in the other with small excess risk.
- **Information loss:** deficiency formalizes what a representation discards for all downstream decisions, stronger than matching one statistic.

## Construction lessons

Robust statistics turns a verbal uncertainty claim into a neighborhood before choosing an estimator. Decision theory then asks whether an information channel preserves all relevant risks, not merely whether it looks similar. Least-favorable and deficiency objects are useful because they support minimax and equivalence theorems. Transferable lesson: define the uncertainty and decision query first; do not use a convenient norm as a substitute for a scientific neighborhood.

## Cross-field summary

Recurring patterns: explicit uncertainty set; least-favorable boundary; minimax/equivalence theorem; impossibility when information is insufficient. Unique pattern: universal downstream-risk comparison through deficiency. Computation is separate from minimax validity. Do not copy contamination neighborhoods into OOD without a justified contamination mechanism.
