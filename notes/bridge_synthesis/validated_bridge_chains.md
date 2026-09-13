# Validated bridge chains

Validation uses the ledger, detailed cards and the extractor reports. Labels distinguish theorem-level edges from interpretation.

| Paper | Validated chain | decisive edge | status |
|---|---|---|---|
| Ben-David 2010 | `HΔH discrepancy -> loss-class gap -> target risk bound + lambda*` | discrepancy inequality | THEOREM |
| Mansour 2009 | `loss-class discrepancy -> transfer triangle -> target risk + cross-domain oracle` | discrepancy inequality | THEOREM |
| Gretton 2012 | `RKHS mean embedding distance -> IPM expectation gap` | RKHS duality | THEOREM |
| Zhao 2019 | `marginal alignment -> conditional mismatch counterexample` | lower bound/counterexample | THEOREM |
| Shui 2022 | `INV/TV source term -> Dobrushin contraction -> unseen-domain BER` | contraction + conditional invariance | THEOREM under assumptions |
| Krueger 2021 | `risk vector variance -> observed-domain dispersion / extrapolation proxy` | risk algebra + restricted assumptions | RELAXATION |
| Sagawa 2020 | `risk vector -> simplex support max_e risk` | support function + uniform convergence | THEOREM |
| Blanchard 2011/2021 | `domain-risk function g_f(P) -> concentration over Pi` | domain-level empirical process | THEOREM under meta-law |
| Arjovsky 2019 | `simultaneous optimum -> invariant predictor objective` | population constraint | POPULATION ABSTRACTION |
| Kamath 2021 | `stationarity constraints -> rank/diversity identifiability` | rank argument | RESTRICTED THEOREM |
| Peters 2016 | `conditional invariance -> causal parent identification -> intervention risk` | SCM identifiability | THEOREM under SCM |
| Lai 2024 | `IRMv1 classifier gradient -> TV functional` | variational translation | FUNCTIONAL TRANSLATION |
| Duchi 2021 / Esfahani 2018 | `uncertainty set -> dual support/reweighting -> robust risk` | convex/transport duality | THEOREM under set assumptions |
| Wang 2026 Tri-Space | `latent direct-sum decomposition -> localized discrepancy -> target risk` | unique decomposition | POPULATION ABSTRACTION |

No extractor supplied a valid theorem-level edge from Fishr covariance or deep MLDG updates to conditional target risk; those remain surrogate/motivational.
