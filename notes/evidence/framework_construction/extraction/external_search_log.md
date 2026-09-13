# External literature search log (2026-09-13)

Targeted Crossref API searches were run for causal transportability, algorithmic stability, PAC-Bayes, Huber contamination, certified robustness, and statistical-experiment comparison. Results were used to select depth-oriented framework papers rather than application volume.

| title | authors | year / venue | reason selected | construction value | source |
|---|---|---|---|---|---|
| Transportability and Causal Generalization | Pearl & Bareinboim | 2011, Epidemiology | explicit source/target transport calculus | selection diagrams, complete identification | [DOI](https://doi.org/10.1097/EDE.0b013e3182254b8f) |
| Causal Transportability with Limited Experiments | Bareinboim & Pearl | 2013, AAAI | limited-information identification | query-first completeness/failure | [DOI](https://doi.org/10.1609/aaai.v27i1.8692) |
| Stability and Generalization | Bousquet & Elisseeff | 2002, JMLR | canonical algorithm-independent certificate | stability-to-risk modular theorem | [JMLR](https://www.jmlr.org/papers/v2/bousquet02a.html) |
| Train Faster, Generalize Better | Hardt, Recht & Singer | 2016, ICML | algorithm-to-stability translation | SGD local lemma + generic stability bound | [arXiv](https://arxiv.org/abs/1509.01240) |
| PAC-Bayesian Bounds for the Generalization Error | McAllester | 1999, COLT | reusable change-of-measure framework | prior/posterior/KL backbone | [DBLP](https://dblp.org/rec/conf/colt/McAllester99.html) |
| Robust Statistics | Huber | 1983, Wiley/SIAM edition | formal contamination uncertainty | neighborhoods and least-favorable laws | [DOI](https://doi.org/10.1137/1025032) |
| Comparison of Statistical Experiments | Torgersen | 1991, Cambridge | information-preservation criterion | deficiency/randomization equivalence | [DOI](https://doi.org/10.1017/CBO9780511666353) |
| Certified Adversarial Robustness via Randomized Smoothing | Cohen, Rosenfeld & Kolter | 2019, ICML | certificate architecture | probability margin → certified radius | [arXiv](https://arxiv.org/abs/1902.02918) |
| Provable Defense against Adversarial Examples via Convex Outer Adversarial Polytope | Wong & Kolter | 2018, ICML | deterministic contrast | relaxation certificate and gap | [arXiv](https://arxiv.org/abs/1711.00851) |
| Generalization Bounds: Perspectives from Information Theory and PAC-Bayes | Alquier | 2025, Foundations and Trends | current synthesis check | information-radius modularity | [DOI](https://doi.org/10.1561/2200000112) |
| Partial Transportability for Domain Generalization | (Crossref record) | 2024 | partial-identification analogue | interval output under nontransportability | [DOI](https://doi.org/10.52202/079017-4376) |

The network calls returned metadata successfully for the DOI records above; no claim depends on search-result snippets alone. Existing repository PDFs remain the primary evidence for OOD-native claims.
