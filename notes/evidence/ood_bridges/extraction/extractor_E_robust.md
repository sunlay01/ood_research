# Extractor E — DRO / robust shift sets

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| Duchi2021 | f-divergence robust objective | `sup_{Q in U} R_Q -> convex dual/reweighting -> robust risk bound` | worst-case U risk | EXACT under U | radius/support misspecification |
| Esfahani2018 | Wasserstein DRO | `transport ball -> Kantorovich dual -> regularized robust risk` | worst-case U risk | EXACT under metric | metric/radius/support |
| Sinha2018 | adversarial/Wasserstein penalty | `transport perturbation -> Lipschitz regularization -> robust risk` | robust risk | EXACT restricted | Lipschitz and ball assumptions |
| Rothenhaeusler2021 | anchor penalty | `anchor residual -> dual uncertainty set -> shift robustness` | intervention family | EXACT restricted | SCM/anchor coverage |

Repeated objects: uncertainty set, support function, dual variable, transport/divergence radius. DRO does not imply robustness outside the declared set.
