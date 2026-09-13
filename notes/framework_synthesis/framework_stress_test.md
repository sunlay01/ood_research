# Framework stress test

The test reconstructs each method as

`training object -> population property -> target-relevant state -> R_T`.

The labels are `EXACT`, `PROVED`, `PROVED-UNDER-RESTRICTIONS`, `CONJECTURAL`,
and `ABSENT`.

| Method | Training -> population | Population -> target state | Target state -> `R_T` | Main exposed gap |
|---|---|---|---|---|
| ERM | EXACT | PROVED only under meta-law/coverage | PROVED under that query | arbitrary target absent |
| V-REx | EXACT risk variance | PROVED-UNDER-RESTRICTIONS under `Pi` | PROVED-UNDER-RESTRICTIONS | variance is not conditional control |
| GroupDRO | EXACT finite support | PROVED for observed groups / declared `U` | PROVED if `P_T in U` | target inclusion |
| MMD | EXACT RKHS witness | PROVED to marginal witness radius | PROVED with conditional/joint residual | marginal-only implication false |
| CORAL | EXACT moments | PROVED-UNDER-RESTRICTIONS moment -> witness | PROVED-UNDER-RESTRICTIONS | higher-order/conditional mismatch |
| DANN | EXACT only after fixed discriminator dual | PROVED for DA discrepancy | PROVED with joint error | source-only target family |
| ICP | EXACT conditional test | PROVED-UNDER-RESTRICTIONS stable mechanism | PROVED-UNDER-RESTRICTIONS | graph/coverage |
| Anchor | EXACT linear shift response | PROVED-UNDER-RESTRICTIONS anchor family | PROVED-UNDER-RESTRICTIONS | family misspecification |
| ideal IRM | EXACT shared optimum | PROVED-UNDER-RESTRICTIONS invariant mechanism | PROVED-UNDER-RESTRICTIONS | rank/realizability |
| IRMv1 | EXACT derivative statistic | PROVED-UNDER-RESTRICTIONS TV translation | PROVED-UNDER-RESTRICTIONS only in restricted functional model | finite/deep bridge |
| Fishr | EXACT gradient covariance | ABSENT general covariance-to-certificate lemma | ABSENT | bottleneck candidate |
| Wasserstein/f-DRO | EXACT support over `U` | PROVED robust theorem | PROVED if target in `U` | radius/metric |
| norm/stability | EXACT complexity object | PROVED source estimation | ABSENT without shift bridge | not target semantics |
| PAC-Bayes/info | EXACT posterior/information theorem | PROVED-UNDER-RESTRICTIONS info shift model | PROVED for declared query | radius/prior operationality |
| MLDG | EXACT update-map objective | PROVED-UNDER-RESTRICTIONS Taylor/convex | PROVED-UNDER-RESTRICTIONS | optimizer error |

## Held-out method test: Fishr

The framework does not silently absorb Fishr into “invariance”. It accepts the
gradient covariance as `OBJECTIVE-FUNCTIONAL`, then stops at

`Cov(g_e) matching -> ? typed certificate`.

This is a valid framework behavior: it exposes a missing theorem rather than
inventing a target-risk implication. The cheapest discriminating probe is a
restricted linear model where covariance can be calculated exactly and compared
with conditional-risk shift; failure there kills the proposed bridge.

## Two-world test

The source observation map identifies the same source laws in the label-reversal
construction. MMD/CORAL and risk-vector certificates are equal, but target risk is
0 versus 1. Candidate framework 01 therefore returns
`SOURCE-INSUFFICIENT / NON-IDENTIFIABLE` instead of a false bound.

## Stress-test result

Candidate 01 explains the exact behavior of finite-group risk, discrepancy,
conditional mechanism and robust methods. It can represent optimizer methods but
does not certify them without local translation theorems. That asymmetry is a
feature required by the evidence, not a defect to be hidden by a larger tuple.
