# Current theory landscape

This is a structural map of the authoritative ledger, not a paper-by-paper survey. The literature separates the theorem target from the representation used to make that target tractable.

## Main target classes

| Target | Information regime | What is estimable | Unavoidable price |
|---|---|---|---|
| `R_T(f)` or `R_T(f)-R_S(f)` | DA, target samples available | source risk and target discrepancy | target-dependent discrepancy and joint-error term (`BenDavid2010`, `Mansour2009`) |
| `E_{P_T~Pi} R_T(f)` | source-only DG, source domains iid from a meta-law `Pi` | within-domain and between-domain empirical risks | meta-distribution/coverage assumption; fresh-domain variance |
| `sup_{P in U(S)} R_P(f)` | source-only robust learning | empirical robust objective under a declared `U` | uncertainty-set metric/radius misspecification (`Duchi2021`, `Esfahani2018`) |
| intervention-robust risk | causal/anchor shift family | source conditional mechanism or anchor residual | SCM, intervention and coverage assumptions (`Peters2016`, `Rothenhaeusler2021`) |
| lower bound/impossibility | arbitrary or weakly specified source-only shift | indistinguishable source worlds | non-identifiability (`Rosenfeld2021`, `Zhao2019`, `Wang2024Lost`) |

## Representation families and proof operations

- **Domain-of-domains / risk-vector:** environment-level concentration and risk aggregation. This is the natural language for expected future-domain risk and finite-group risk. It captures ERM, V-REx and GroupDRO at different fidelity levels, but cannot identify arbitrary conditional shifts.
- **DRO uncertainty sets:** convex or transport duality turns a worst-case distribution problem into a tractable weighted or regularized objective. It gives the cleanest source-only robust target when `U(S)` is scientifically declared; its guarantee is conditional on `U`.
- **Discrepancy/IPM/RKHS:** triangle inequalities and concentration provide modular transfer bounds. Classic versions are DA because they require target samples and retain a joint-error term. Source-only use requires a source-defined family or an assumption replacing target discrepancy.
- **Causal/conditional invariance:** conditional independence and intervention structure can identify stable mechanisms. The gain is interpretability and a target family aligned with a causal claim; the cost is strong structural assumptions.
- **Latent decomposition / contraction:** `Shui2022` and `Wang2026TriSpace` show how contraction or a direct-sum decomposition localizes a shift term. These are useful refinements inside a stated family, not assumption-free unifiers.
- **TV translation:** `Lai2024` gives IRMv1's classifier-gradient penalty a variational TV reading. This is an objective-to-theory translation and does not cover Fishr, GroupDRO or generic DRO.
- **PAC-Bayes/information/stability:** these control posterior or learner complexity and can be attached to a shift term. They do not define the admissible target family by themselves.

## Recurring proof architecture

The robust pattern is

`target quantity -> same-level decomposition -> shift/family term + complexity/estimation term -> concentration or duality -> oracle/misspecification remainder`.

The strongest source-only results make the target family explicit before selecting a representation. Theorems that omit that step either become DA bounds with target data, robust bounds conditional on a chosen uncertainty set, or impossibility results.

## Decision-relevant conclusion

The existing representation system is sufficient for the next theory step. The project should develop a domain-level concentration theorem for an explicit source-only meta-distribution target, with a separate DRO formulation for worst-case robustness. A new cross-method representation would duplicate the current coverage while adding assumption cost and no missing proof capability.
