# Source-only estimability

| object | classification | implementation note |
|---|---|---|
| source moments and source risk | directly source-estimable | exact population moments in Phase I |
| `w_ref`, `H_R`, `B_R` | directly source-estimable | exact quadratic head derivatives |
| finite pseudo-response target | source-domain holdout | leave one source domain out; it is not a target domain |
| local interpolation tangent | source-domain holdout / family model | needs coordinates and more than two domains; documented, not selected |
| meta-gradient surrogate | source-estimable | can approximate a component of response matching, but is not exact here |
| `O_S` | source-estimable only relative to declared family | not inferred from arbitrary unlabeled source data |
| `A_rec` | requires a declared family model or a source pseudo-target estimator | exact oracle is post-hoc diagnostic; the implemented estimate is a finite source contrast |
| true target risk, target labels, target operator | requires target information -> prohibited | only generated after source-only fitting for held-out evaluation |
| `Pi` | source-estimable through exact head solution | never a free matrix in the learner |
| `rho_slack` and `R_info` | declared-family/post-hoc | not used for beta or checkpoint selection |

With only two source domains, the finite contrast has high variance and no independent tangent fit. With more source domains, LOO gives a source-only validation analogue but remains a finite approximation to the family operator. The experiment therefore separates oracle, estimated, and matched-norm random directions.

## Competing estimators for the required response

| estimator | assumptions | bias | variance | coordinates? | Hessian? | two domains? | cost | closest analogue |
|---|---|---|---|---|---|---|---|---|
| finite source-optimum difference | nearby source domain is a useful pseudo-shift and quadratic head is identifiable | finite-shift curvature bias | high with few domains | no | only for whitening/audit | yes | one solve per domain | MLDG pseudo-test split |
| local tangent fit | declared source coordinates and enough independent source domains | model/tangent misspecification | depends on design conditioning | yes | optional | no | least-squares plus head solves | local transferability estimation |
| meta-gradient surrogate | one-step Taylor approximation is adequate | optimizer-step and higher-order bias | minibatch-gradient variance | no | implicit or approximated | yes | gradient steps | MLDG/Fish |

The implemented estimator is the first one because the exact Gaussian probe can audit its finite displacement without adding an approximation. This choice does not imply it is statistically preferable outside the controlled setting.
