# Controlled synthetic OOD failure-regime audit

## Aim

This experiment tests whether usage/readout and representation failures form
distinct, factor-dependent continuous axes, while auditing whether nuisance
counterfactual sensitivity is an independent and repairable axis. It does not
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
`C_prob` is the mean squared difference between nonlinear probe probabilities
on pairs with identical `u` and independently resampled `s,n`; raw-logit
`C_pred` is recorded only as a non-scale-invariant comparison. `head_repair_gain`
uses an independently generated balanced repair set and independent probe seed.

The primary analysis uses seed-level 2x2 interaction contrasts, not thresholds:
`rho x sigma_c -> G_use`, `k x optimizer -> G_repr`, and `k x d_z -> G_repr`.
An absolute seed-level z-score of 2 is an exploratory stability flag, not a
claim of inferential significance.

## Interventions and falsification gates

Head repair is a frozen-representation intervention evaluated on a separate
balanced repair set. Full repair is a source-only retraining intervention on
balanced spurious features using the same architecture and optimizer.
Contamination remains unresolved unless probability-space sensitivity predicts
an independently specified nuisance-removal intervention and improves target
risk; a reduction in sensitivity alone is insufficient.

The experiment is not treated as a regime theory if continuous metrics are
replaced by priority labels, if clusters are explained only by optimizer
identity, or if diagnostics and repairs are circularly defined.

## Reproducibility

CPU, torch deterministic mode, seeds `10..14`, checkpoints at `0,50,100,200,500`.
Target labels are read only for post-hoc diagnostics and intervention scoring.
