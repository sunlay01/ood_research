# Stage 13R.1 Novelty Audit

## Search scope

The audit revisited the primary Moment Alignment paper and searched the
repository ledger and recent primary literature for additive-risk nuisance
quotients, Bayes-risk/regret derivatives, differential transferability,
pairwise/ranking risk geometry, local excess-risk sensitivity, tangent-space
DG, local DRO, robust statistics, decision-theoretic regret, and
optimal-recovery formulations of risk differences.

## Classification

| Structure | Classification | Reason |
|---|---|---|
| raw `sup D_xi R` | `COMPONENT-LEVEL` | local robustness/support and distributional sensitivity already contain this pattern |
| excess `D_xi(R-R*)` | `EQUIVALENT-AFTER-REPARAMETERIZATION` | endpoint path integral is the Moment Alignment excess-risk transfer target under shared references |
| quotient modulo `c(xi) 1_F` | `SPECIAL-CASE` / `COMPONENT-LEVEL` | standard centering, regret, or pairwise-risk construction; relative rather than absolute risk |
| fixed Euclidean tangent support | `SPECIAL-CASE` | ordinary dual norm/support geometry |
| complete Stage 13R ESF package | `NO-EXACT-OVERLAP-FOUND` | no checked source stated the exact same end-to-end organization, but this is negative evidence only |

The absence of an exact end-to-end hit is not sufficient for novelty because the
surviving nuisance-invariant candidate either coincides with the established
excess-risk transfer target or becomes a standard pairwise/regret object that
does not yield the required absolute OOD certificate.

## Conclusion

No independent novelty claim survives this invariance gate. Preserve the
support/quotient machinery as components of the Stage 12/13 theory, but do not
present raw ESF as a new master functional. Any future relative-risk quotient
would require a new, separately justified target and theorem gate.
