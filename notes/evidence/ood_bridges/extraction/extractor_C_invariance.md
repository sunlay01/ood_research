# Extractor C — Invariance / causal identification

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| Arjovsky2019 | shared optimal classifier constraint | `simultaneous risk minimizer -> invariant predictor candidate` | DG objective | POPULATION-ABSTRACTION | no finite/deep guarantee |
| Kamath2021 | ideal IRM in restricted linear models | `gradient/optimality constraints -> rank/diversity identifiability -> causal solution` | unseen-domain risk | EQUIVALENT-UNDER-ASSUMPTIONS | rank and heterogeneity |
| Peters2016 | invariant conditional tests | `P_e(Y|X_S) invariant -> causal parent set identification -> intervention risk` | causal intervention risk | EXACT under SCM | faithfulness/coverage |
| Rothenhaeusler2021 | anchor regression penalty | `anchor residual geometry -> primal-dual robustness` | shift-family risk | EXACT restricted linear | anchor/SCM family |
| Rosenfeld2021 | finite-environment IRM analysis | `finite source constraints -> non-identifiability counterexample` | lower bound | LOWER-BOUND | indistinguishable mechanisms |

Repeated objects: conditional mechanism, simultaneous optimum, rank/diversity, intervention family, identifiability defect. These are not equivalent to marginal IPMs.
