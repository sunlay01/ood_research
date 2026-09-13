# Required mathematical properties

These properties are inferred from the obstruction map before proposing any new representation.

| Obstruction | Required property | Cheapest existing source |
|---|---|---|
| Unknown future-domain population | Domain-level sampling law with concentration over environments | `Blanchard2011`, `Blanchard2021` |
| Explicit worst-case shift | Dual uncertainty-set representation | `Duchi2021`, `Esfahani2018` |
| Conditional-label mismatch | Conditional invariance or a localized conditional discrepancy | `Peters2016`, `Shui2022`, `Zhao2019` as failure test |
| Invariant/variant entanglement | A justified decomposition or contraction | `Wang2026TriSpace`, `Shui2022` |
| Target-discrepancy oracle | Source-defined family term or an explicit irreducible oracle term | domain-of-domains or DRO; DA cannot remove it |
| Fresh-domain tail risk | Quantile, variance, or robust supremum over `R_P(f)` | meta-distribution tail term or DRO |
| Finite-sample estimation | Uniform convergence at the domain and within-domain levels | Rademacher/VC concentration in `Blanchard2011`, `Blanchard2021` |
| Algorithm-specific gradient objective | Functional translation plus derivative-class control | `Lai2024` gives translation; derivative theorem remains open |
| Optimizer error | Explicit `epsilon_opt` additive term | standard population abstraction; no microscopic bridge assumed |

## No new representation is forced

The required properties partition into two existing languages:

1. `Pi` plus domain-level risk class `G_F` for expected future-domain risk.
2. `U(S)` plus robust risk for a declared worst-case target.

Conditional invariance and latent decomposition are optional refinements when the scientific shift claim requires them. They are not missing base capabilities. A proposed new representation would need to outperform these choices on a named obstruction, not merely combine their notation.
