# CMNIST Round-3 Geometry Bridge Report

## Verdict

`CMNIST-GEOMETRY-PASS`

This is an empirical finite-sample bridge for a frozen learned
representation and a trainable squared-loss linear head. It is not a proof of
the Gaussian population theorems on neural networks.

## Main findings

- Five ERM encoder seeds were trained with the existing CMNIST generator.
- The feature bank was reduced to the empirical non-degenerate support; no
  damping was added to the head Hessian.
- `A` step stability: `True`; `O_S` step stability: `True`.
- Hidden mechanism alpha mean: `7.27674`.
- Exposed mechanism alpha mean: `4.20123`.
- The hidden-to-exposed decrease is a source-exposure diagnostic, not a
  causal identification or target-risk theorem.

## Method summary

The recoverable residual is defined as
`E = A_recoverable + Pi O_S`, where `A_recoverable = A(I-P_ker(O_S))`.
The irreducible response `A_irreducible = A P_ker(O_S)` is used only for the
information floor and spectral-slack diagnostic. The maximum reconstruction
residual `||A-A_irreducible-A_recoverable||` over method rows is
`2.12e-15`.

| family | method | mean ||z0|| | mean ||Pi O|| | mean ||E|| | mean affine regret | mean FD error |
|---|---|---:|---:|---:|---:|---:|
| declared_source_target_coupled_correlation | ERM | 0 | 2.6957 | 11.009 | 62.692 | 4.37e-06 |
| declared_source_target_coupled_correlation | L2 | 0.66495 | 2.8853 | 11.137 | 65.213 | 2.81e-06 |
| declared_source_target_coupled_correlation | IRMV1 | 0.012063 | 2.7041 | 11.01 | 62.778 | 3.93e-06 |
| declared_source_target_coupled_correlation | VREX | 0.0013271 | 2.6815 | 11.003 | 62.642 | 3.94e-06 |
| irrelevant_source_diversity | ERM | 0 | 3.3934 | 11.147 | 90.739 | 6.94e-05 |
| irrelevant_source_diversity | L2 | 0.66495 | 3.3642 | 11.289 | 93.583 | 6.55e-05 |
| irrelevant_source_diversity | IRMV1 | 0.012063 | 3.3962 | 11.148 | 90.825 | 6.91e-05 |
| irrelevant_source_diversity | VREX | 0.0013271 | 3.3879 | 11.142 | 90.699 | 6.96e-05 |
| mechanism_defined_exposed | ERM | 0 | 4.6691 | 13.395 | 100.57 | 1.45e-05 |
| mechanism_defined_exposed | L2 | 0.66495 | 4.9975 | 13.64 | 105.06 | 9.31e-06 |
| mechanism_defined_exposed | IRMV1 | 0.012063 | 4.6503 | 13.392 | 100.63 | 1.47e-05 |
| mechanism_defined_exposed | VREX | 0.0013271 | 4.6436 | 13.383 | 100.43 | 1.42e-05 |
| mechanism_defined_hidden | ERM | 0 | 2.6957 | 11.009 | 89.189 | 4.37e-06 |
| mechanism_defined_hidden | L2 | 0.66495 | 2.8853 | 11.137 | 91.898 | 2.81e-06 |
| mechanism_defined_hidden | IRMV1 | 0.012063 | 2.7041 | 11.01 | 89.291 | 3.93e-06 |
| mechanism_defined_hidden | VREX | 0.0013271 | 2.6815 | 11.003 | 89.151 | 3.94e-06 |

All target accuracy values are post-hoc context only. The source state and
operator construction read no target risk, semantic label, cluster label or
regularizer geometry. `ift_fd_rows.csv` contains the per-direction,
per-step audit; `method_comparison.csv` contains method rows; and
`operator_snapshots.npz` contains the A/O matrices.

## Interpretation boundary

The result supports or rejects applicability only for the stated frozen
representation, empirical moments, tangent family and metric. It does not
establish a finite-sample guarantee, causal mechanism identification,
universal DG theorem, or target-risk lower bound. Full-CNN cross-entropy and
the prior CMNIST probe remain separate tracks.
