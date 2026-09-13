# Extractor F — Latent / information / calibration formalizations

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| Wang2026TriSpace | invariant/spurious/variant latent decomposition | `unique direct-sum decomposition -> localized discrepancy terms -> target risk` | DG target risk | POPULATION-ABSTRACTION | deterministic labels and latent assumptions |
| Liu2024InfoOOD | information regularization | `information density/radius -> stochastic generalization bound` | OOD risk | SURROGATE | information model and shift term |
| Wu2024Multicalibration | calibration constraints | `group calibration -> density-ratio weighted risk control` | calibrated OOD risk | EXACT under coverage | calibration coverage |

Repeated objects: latent decomposition, information radius, density-ratio/calibration constraint. These solve particular obstructions and are not interchangeable with HΔH discrepancy or DRO support.

## Local bridge summary

The corpus repeatedly uses five proof roles: distinguishability, robust target-set definition, conditional/mechanism identification, contraction/localization, and statistical estimation. No common scalar bridge is established across all families.
