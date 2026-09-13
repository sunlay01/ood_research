# Agent B — Causal transportability and identifiability

## External search and selected works

Crossref search (2026-09-13) for “transportability causal inference” returned: Pearl & Bareinboim, *Transportability of Causal Effects: Completeness Results* (2011), DOI [10.1097/EDE.0b013e3182254b8f](https://doi.org/10.1097/EDE.0b013e3182254b8f); Bareinboim & Pearl, *Causal Transportability with Limited Experiments* (2013), DOI [10.1609/aaai.v27i1.8692](https://doi.org/10.1609/aaai.v27i1.8692); Peters et al. (2016) ICP. Selected for explicit graph/query/identification separation and completeness.

## Framework cards

### Transportability / selection diagrams
- **Pre-formalization difficulty:** source and target differ structurally, so ordinary observational adjustment is not enough.
- **Primitives:** causal DAG/SCM, selection nodes encoding domain differences, source/target distributions, causal query.
- **Relations:** graph separation, do-calculus transformations, selection-node constraints.
- **Intermediate:** an identifiable functional of available source/target observations; if none exists, a non-identifiability certificate.
- **Backbone:** sound and complete transportability algorithm: every identifiable query is reducible to admissible source/target components, and failure indicates lack of identification.
- **Assumptions:** scientific graph, intervention semantics, positivity/availability; not merely statistical similarity.
- **Generalization:** new queries plug into the same identification calculus.

### ICP
- **Primitive choice:** candidate parent sets and conditional laws `P(Y|X_S)` preserve causal mechanism rather than marginals.
- **Intermediate:** intersection of accepted invariant sets.
- **Backbone:** under SCM, faithfulness and intervention coverage, selected set contains causal parents and supports stable prediction.
- **Boundary:** weak heterogeneity, hidden confounding, or failed coverage destroys identification.

## Construction lessons

Causal frameworks are query-first: they separate available information, structural model, target query and identification algorithm. They define insufficiency positively through completeness/failure, not by hiding it in an error term. The key transferable lesson is to make “what is invariant” and “what differs across domains” explicit before selecting statistics. Do not import DAGs into OOD when the claimed environment mechanism is not testable or specified.

## Cross-field comparison

Recurring patterns: explicit scientific model; query-specific target; completeness or impossibility; modular identification calculus. Unique pattern: graphical representation preserves intervention semantics. Algorithm-specific content is the identification procedure; estimation comes later. Safe transfer is the query-first and failure-certificate discipline, not automatic causal interpretation of alignment.
