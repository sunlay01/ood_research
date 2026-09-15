# Controlled synthetic OOD failure-regime audit

## Aim

This experiment tests whether readout-limited, representation-limited, and
functionally contaminated OOD failures are empirically separable. It does not
claim a theory or a universal taxonomy.

## Factorial design

32 configurations, 5 seeds each:

- `rho`: `low=(0.65,0.55)` or `high=(0.95,0.85)` source spurious agreement;
- `k`: `linear`, with `y=sign(u0)`, or `nonlinear`, with `y=sign(u0*u1)`;
- `sigma_c`: `0.0` or `0.2` core-label flip probability;
- `d_z`: `2` or `8` representation width;
- optimizer: `sgd` or `adam`.

The target spurious agreement is fixed at `0.10`. Inputs are a fixed random
orthogonal mixing of `(u0,u1,s,n0,n1)`. The model is
`5 -> 32 -> d_z -> 1`, with ReLU before the representation and a linear head.

## Frozen diagnostics

All diagnostics use a separately generated balanced probe distribution with
spurious feature independent of the label. `R_actual` is target BCE risk.
`R_head` is target BCE after retraining only a linear head on frozen features;
`R_probe` is target BCE after retraining a nonlinear probe on frozen features;
`R_core` is target BCE of a nonlinear probe given the true core `u`.

`G_use = R_actual - R_head`, `G_repr = R_probe - R_core`.
`C_pred` is the mean squared difference between the nonlinear representation
probe's logits on pairs with identical `u` and independently resampled `s,n`.

Pre-registered diagnostic thresholds are `G_use >= 0.10`, `G_repr >= 0.10`,
and `C_pred >= 0.01`; threshold labels are descriptive only.

## Interventions and falsification gates

Head repair is the frozen-representation oracle-head intervention. Full repair
is a source-only retraining intervention on balanced spurious features using
the same architecture and optimizer. A readout regime must show head repair
improvement; a representation regime must show little head improvement but a
full-retrain improvement. Contamination is only considered distinct if `C_pred`
adds separation beyond the two gaps and nuisance counterfactual prediction
changes are reduced by balanced retraining.

The experiment is rejected as a regime theory if metrics are smooth and
collinear, if clusters are explained only by optimizer identity, or if the
diagnostic gaps fail to predict the corresponding intervention.

## Reproducibility

CPU, torch deterministic mode, seeds `10..14`, checkpoints at `0,50,100,200,500`.
Target labels are read only for post-hoc diagnostics and intervention scoring.
