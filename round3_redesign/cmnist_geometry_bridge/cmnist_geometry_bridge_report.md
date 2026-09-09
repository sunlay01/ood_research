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
| irrelevant_source_diversity | ERM | 0 | 3.3934 | 11.147 | 90.739 | 6.94e-05 |
| irrelevant_source_diversity | L2 | 0.66495 | 3.3642 | 11.289 | 93.583 | 0.000125 |
| irrelevant_source_diversity | IRMV1 | 0.012063 | 3.3962 | 11.148 | 90.825 | 0.000352 |
| irrelevant_source_diversity | VREX | 0.0013271 | 3.3879 | 11.142 | 90.699 | 0.000333 |
| mechanism_defined_exposed | ERM | 0 | 4.6691 | 13.395 | 100.57 | 1.45e-05 |
| mechanism_defined_exposed | L2 | 0.66495 | 4.9975 | 13.64 | 105.06 | 7.27e-05 |
| mechanism_defined_exposed | IRMV1 | 0.012063 | 4.6503 | 13.392 | 100.63 | 0.000339 |
| mechanism_defined_exposed | VREX | 0.0013271 | 4.6436 | 13.383 | 100.43 | 0.000288 |
| mechanism_defined_hidden | ERM | 0 | 2.6957 | 11.009 | 89.189 | 4.37e-06 |
| mechanism_defined_hidden | L2 | 0.66495 | 2.8853 | 11.137 | 91.898 | 0.000112 |
| mechanism_defined_hidden | IRMV1 | 0.012063 | 2.7041 | 11.01 | 89.291 | 0.000501 |
| mechanism_defined_hidden | VREX | 0.0013271 | 2.6815 | 11.003 | 89.151 | 0.00034 |
| source_induced | ERM | 0 | 2.6957 | 11.009 | 62.692 | 4.37e-06 |
| source_induced | L2 | 0.66495 | 2.8853 | 11.137 | 65.213 | 0.000112 |
| source_induced | IRMV1 | 0.012063 | 2.7041 | 11.01 | 62.778 | 0.000501 |
| source_induced | VREX | 0.0013271 | 2.6815 | 11.003 | 62.642 | 0.00034 |

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
