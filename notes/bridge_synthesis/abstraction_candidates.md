# Abstraction candidates after bridge extraction

## Candidate A — Generic seminorm/IPM

`D_G(P,Q)=sup_{g in G}|E_P g-E_Q g|` explains HΔH, loss-class discrepancy and RKHS MMD. It does not subsume conditional invariance, rank identifiability, or DRO uncertainty semantics; those need extra objects.

## Candidate B — Support/uncertainty functional

`sigma_U(ell_f)=sup_{Q in U} E_Q ell_f` explains GroupDRO, f-DRO and Wasserstein DRO. It is not the same as an IPM: the set geometry is a target-family assumption, not merely a witness class.

## Candidate C — Identifiability defect

The distance from a learned solution to the set of predictors/mechanisms compatible with all environments explains IRM/ICP/rank arguments and lower bounds. It cannot be estimated from marginal moments alone.

## Candidate D — Contraction/localization coefficient

TV/Dobrushin coefficients and localized latent discrepancies explain how a raw environment shift is reduced before entering target risk. They require a conditional channel or decomposition and therefore do not form a universal bridge.

## Candidate E — Translation operator

Objective-to-functional maps explain IRMv1-TV and restricted MLDG analyses. They are parameterization and optimization dependent, so they cannot be identified with the target-risk bridge.

**Assessment:** the evidence supports a small family of proof-role objects (seminorms, support functionals, identifiability defects, contractions, translation maps), not one common bridge object.
