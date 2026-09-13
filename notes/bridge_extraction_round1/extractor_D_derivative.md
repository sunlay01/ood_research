# Extractor D — Derivative / variational / optimizer objects

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| Lai2024 | IRMv1 classifier-gradient penalty | `gradient penalty -> TV variation functional (variational translation) -> OOD condition` | functional OOD condition | FUNCTIONAL-TRANSLATION | smoothness/coarea; no deep SGD theorem |
| Arjovsky/Kamath | classifier stationarity | `population derivative/optimality -> restricted identifiability` | DG risk | POPULATION-ABSTRACTION | derivative class and rank |
| Rame2022/Fishr | gradient variance matching | `per-example gradient covariance ->? invariant prediction` | motivation | MOTIVATIONAL-ONLY / SURROGATE | no proved covariance-to-conditional bridge |
| Li2018 MLDG | meta-train update map | `bilevel update -> Taylor cross-environment loss` | DG proxy | SURROGATE | optimization and Taylor remainder |

Repeated objects: derivative functional, variational translation, stationarity, optimizer remainder. The central regularizer-to-target arrow is unproved for Fishr and unrestricted deep IRMv1.
