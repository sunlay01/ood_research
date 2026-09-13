# Route A — Existing discrepancy/IPM theorem plus local lemma

- **Problem instance:** P2, MMD/CORAL to target transfer risk.
- **Formalization:** fixed RKHS/loss-class IPM and conditional change-of-measure decomposition.
- **Reuse/change:** reuse existing IPM theorem; no framework change.
- **Representation:** chosen witness class for marginal loss sections plus an explicit conditional discrepancy.
- **Required result:** empirical MMD/CORAL to population marginal-radius bound; conditional radius remains separate.
- **Assumptions:** bounded kernel/features, finite witness complexity, overlap, declared target family.
- **Source-estimable:** empirical risk and source-source discrepancy.
- **Assumption-controlled:** target radius, witness inclusion and conditional coverage.
- **Irreducible:** conditional mismatch and out-of-family misspecification.
- **Failure case:** aligned marginals with flipped target labels.
- **Solved if successful:** a faithful source-only transfer theorem for one representation regularizer under an explicit target family.
