# Framework decision

## Decision

**SMALL-FAMILY-OF-BRIDGES.** The extracted literature does not support one universal bridge object. It supports a small proof-role family:

1. discrepancy/IPM seminorms;
2. uncertainty-set support functionals;
3. identifiability/mechanism defects;
4. contraction/localization coefficients;
5. algorithm-to-population translation maps;
6. statistical complexity terms.

These are linked by theorem-specific edges, not by a single scalar representation. Any future framework must be induced by this family and preserve irreducible terms.

## Coverage

Exact: discrepancy transfer, support-function robustness, two-level concentration, and selected conditional decompositions. Translated or restricted: IRMv1, Fishr, MLDG, latent decompositions. Excluded: arbitrary targets and assumption-free optimizer-to-risk claims.

## Next theorem

Prove a conditional decomposition theorem with a source-estimable witness class and explicit `rho_cond`, then test whether one concrete regularizer supplies a non-vacuous penalty-to-radius lemma. Keep a matching channel-swap lower bound.
