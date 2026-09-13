# Extractor B — Risk-level regularization

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| Krueger2021 | mean risk + `Var_e R_e` | `risk vector -> dispersion control -> extrapolation claim under restricted domain model` | DG risk proxy | RELAXATION / restricted theory | unseen-domain law and variance-to-tail gap |
| Sagawa2020 | `max_e R_e` | `risk vector -> simplex support function -> worst observed-group risk` | robust group risk | EXACT | no unseen-group coverage |
| Blanchard2011/2021 | domain risk function | `g_f(P)=R_P(f) -> concentration over Pi -> fresh-domain expected risk` | expected future-domain risk | POPULATION-ABSTRACTION | meta-law tail / coverage |

Repeated objects: risk vector, support function, domain-level function class, concentration. V-REx does not prove conditional invariance; GroupDRO proves only the declared group family.
