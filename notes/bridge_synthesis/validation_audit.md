# Validation audit for extracted chains

## Procedure

Each chain was checked against the compact ledger, available detailed cards, and (where the card gave a decisive pointer) the original PDF theorem/proposition. The audit treats motivation as non-proof and keeps target-dependent quantities visible.

## Corrections and confirmations

- Ben-David/Mansour: the discrepancy edge is valid, but the final bound retains target discrepancy and an ideal joint-error term.
- Gretton MMD: the paper proves the RKHS mean/IPM identity and concentration; inserting a prediction-loss witness is an additional assumption, not a theorem about arbitrary classifiers.
- Zhao: the conditional-label term and aligned-marginal counterexample are essential; no marginal-alignment-to-target-risk implication was accepted.
- Shui: the contraction result requires latent conditional invariance, a nearest-source/raw-TV radius, and a Dobrushin coefficient; it is not a generic MMD theorem.
- Krueger/V-REx: risk variance is recorded as a risk-profile relaxation; no unconditional unseen-domain bridge was inferred.
- Arjovsky/Kamath: ideal IRM and restricted rank identifiability are separated from practical IRMv1.
- Lai: the gradient-to-TV edge is labeled functional translation and not deep-SGD fidelity.
- Duchi/Esfahani: robust duality is exact only for the declared uncertainty geometry and support.
- Tri-Space/InfoOOD/Multicalibration: their decompositions/information/calibration objects remain family-specific.

## Rejected claims

No chain is recorded that proves `Fishr covariance -> invariant conditional -> arbitrary target risk`, `MMD -> conditional label alignment`, or `IRMv1 empirical gradient -> exact deep optimizer target theorem`. These would require new evidence.
