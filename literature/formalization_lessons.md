> **Status: HISTORICAL / INTERMEDIATE RESEARCH NOTE.** Retained for provenance. Do not read by default; use `literature/ledger/` as the current authoritative entry point.

# Formalization lessons

1. Pick the theorem target first. A source-risk bound, unseen-domain expectation, robust supremum, and target risk are different quantities.
2. Make the shift model explicit. DA discrepancy bounds use target samples; source-only DG uses a meta-distribution, causal/interventional family, coverage condition, or an uncertainty set.
3. Abstract the optimizer unless it is part of the claim. Most papers analyze the population minimizer or an induced function class and append an optimization error; only restricted convex/linear settings analyze the algorithm closely.
4. Preserve algorithm identity in a side column. “`min L+lambda Omega`” is syntactic unification, not theoretical unification. Record whether the theorem is exact, equivalent, a relaxation, or motivational.
5. Use impossibility results as representation tests. If a framework has no place for joint error, identifiability, or family misspecification, it cannot support a non-vacuous DG theorem.
6. A realistic future search has layered abstractions: (A) risk-vector/environment statistics; (B) function class plus discrepancy; (C) robust uncertainty set; (D) RKHS/operator specialization; (E) causal conditional invariance. A single layer should not be expected to encode optimizer-level quantities exactly.
