# Algorithm Mapping and Fidelity Ledger

| Method | Theoretical object | Fidelity | Bridge | Error term / failure |
|---|---|---|---|---|
| ERM | mean source risk | EXACT | two-level concentration | no target-family identification |
| V-REx | mean + source risk variance | EXACT at population risk level | controls observed risk dispersion only | no conditional guarantee |
| GroupDRO | simplex support `max_e R_e` | EXACT for finite groups | support-function robustness | unseen-group misspecification |
| MMD | RKHS marginal mean discrepancy | EXACT for fixed kernel | IPM duality | marginal alignment misses labels |
| CORAL | covariance-operator discrepancy | FUNCTIONAL-TRANSLATION | second-moment bound | higher moments/conditionals omitted |
| DANN | discriminator/loss-class IPM | FUNCTIONAL-TRANSLATION | H-divergence theorem with target/oracle terms | not source-only in classic DA |
| Ideal IRM | shared risk minimizer | EQUIVALENT-UNDER-ASSUMPTIONS | conditional/SCM identification | rank/heterogeneity assumptions |
| IRMv1 | classifier gradient penalty | FUNCTIONAL-TRANSLATION | TV/variational result or restricted model | derivative + optimization gap |
| Fishr | covariance of per-example gradients | SURROGATE | no general conditional bridge | parameterization dependence |
| f-/Wasserstein DRO | support over declared uncertainty set | EXACT under set assumptions | convex/transport duality | radius/metric/support misspecification |
| MLDG | meta-train/meta-test update map | SURROGATE | Taylor/convex restricted analysis | deep bilevel dynamics not covered |

All theorem statements must label source-estimable, assumption-controlled, target-dependent/oracle, irreducible, translation and optimization terms separately.
