# Formalization adequacy audit

This audit starts from the scientific problem: connect a concrete source-only OOD/DG regularizer to a declared target-risk quantity. It does not assume a new representation.

| Family | Scientific object / target | Preserved and discarded information | Natural algorithms | Source-only uncertainty / non-identifiability | Classification |
|---|---|---|---|---|---|
| Discrepancy / IPM | source-target distinguishability; transfer risk plus joint error | preserves a chosen witness class; discards orthogonal and conditional-label directions | MMD, CORAL, DANN, HDeltaH | classic DA needs target data; source-only requires a target family and retains `lambda*` or conditional remainder | ADEQUATE-WITH-ALGORITHM-LOCAL-LEMMA |
| Conditional / causal invariance | stable `P(Y|C)` or intervention risk | preserves mechanism/intervention semantics; discards worlds outside SCM/coverage | ICP, anchor, ideal IRM | explicit SCM/intervention family; failure is non-identifiability | ADEQUATE-WITH-ALGORITHM-LOCAL-LEMMA |
| Robust / uncertainty set | `sup_{Q in U} R_Q` | preserves declared shift geometry; discards out-of-set targets | GroupDRO, f-DRO, Wasserstein DRO | uncertainty set is explicit; misspecification is visible | ADEQUATE-AS-IS for its target |
| Identifiability / rank | whether a stable predictor/mechanism is determined | preserves distinctions used by rank/graph assumptions; can reject identification | ideal IRM, ICP, transportability | represents failure through lower bounds or empty identified set | ADEQUATE-WITH-ALGORITHM-LOCAL-LEMMA |
| Contraction / localization | how raw shift is reduced before risk | preserves channel/decomposition structure; loses shifts outside the channel | INV/TV, Tri-Space | target radius and contraction coefficient remain assumption-controlled | ADEQUATE-WITH-ALGORITHM-LOCAL-LEMMA |
| Meta-domain | expected risk on `P~Pi` | preserves domain population; discards arbitrary out-of-support targets and point-target tails | ERM/V-REx/GroupDRO as source learners | source-only concentration over `Pi`; fresh-domain quantile remains | ADEQUATE-AS-IS for expected-domain target |
| Variational / functional translation | maps practical penalty to population functional | preserves only translated functional; discards optimizer details unless proved | IRMv1, restricted MLDG, derivative methods | translation and optimization errors must be explicit | ADEQUATE-WITH-ALGORITHM-LOCAL-LEMMA |

## Audit conclusion

Existing families already express the relevant target quantities. The missing work is local: prove an algorithm-specific empirical-to-population and regularizer-to-bridge lemma, then reuse an existing transfer, conditional, robust, or meta-domain theorem. A representation change is justified only if a chosen target query cannot be stated with the existing bridge object; the current evidence does not establish that failure.

The main structural obstruction is information insufficiency, not notation: equal source risks or marginal discrepancies can hide a target conditional flip. The correct response is an explicit conditional/joint-error or target-family term, not an automatic new tuple of statistics.
