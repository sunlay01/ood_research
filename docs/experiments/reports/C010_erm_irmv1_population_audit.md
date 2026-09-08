# CAL-002: ERM--IRMv1 Population Mechanism Audit

## Purpose

Executable sanity check for C010 in the centered scalar Gaussian SCM. This is not a benchmark experiment or a finite-sample guarantee.

## Results

`PYTHONPATH=src python -m ood_repr_reg.erm_irmv1_report` reports:

- **ERM mixture blindness:** with source relations `(0.8, 0.3)`, ERM has mixture-stationarity residual `0`, but scale derivatives `(+0.05640, -0.05640)`, nonzero tangent gradient norms, and IRMv1 penalty `0.003181`.
- **ERM restricted positive control:** symmetric source relations `(-0.8,0.8)` give ERM `(0,0.8,0)` and exactly zero transport to a target changing relation, nuisance mean, and nuisance variance.
- **Three-relation IRMv1 positive control:** source relations `(-0.6,0.1,0.8)` have quadratic-design smallest singular value `0.37312`. The nontrivial U-only zero has source risk `0.21`, below the zero predictor's `1.01`, zero IRMv1 penalty, zero nuisance bound, and zero relation/mean/covariance transport.
- **Finite-penalty bridge:** for `(u,a)=(0.45,0.35)`, actual `a^2=0.1225`, design-based bound `0.47381`, absolute target transport `0.3885`, and the conditional transport upper bound `2.71214`.

## Interpretation

The positive result controls only the effective nuisance coefficient under a specific centered scalar source design. The remaining target causal excess of the U-only predictor is `0.2`, caused by partial observation of `C`, not by nuisance transport. The finite-penalty bound is loose in this toy setting but has the required direction and retains target geometry explicitly.

## Verification

`PYTHONPATH=src python -m pytest -q` passed `40` tests after this audit. The unit tests additionally recompute exact telescoping, projected accounting, the two-source blind branch, and representation reparameterization invariance.
