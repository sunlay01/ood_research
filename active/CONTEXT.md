# TASK3-CMNIST-LOCAL-RESPONSE Context

Current question:

- Does curvature-aware local head-gradient response disagreement produce useful end-to-end CMNIST DG training signal beyond ERM, IRMv1, V-REx, and the same unpreconditioned gradient-alignment penalty?

Frozen inputs needed:

- `T-3A-A`, `T-3C-PI`, `T-3D-IDENTIFIABILITY`, `T-3E-INFO-FLOOR`, `T-3E-SPECTRAL-SLACK`, `T-3E-AFFINE-REGRET` motivate the local source-risk metric but are not used as an online `A_rec`, `E`, or slack estimator.
- `R-REPAIR-GATE`, `R-TASK1-MECHANISM`, and `R-TASK2-SHARP` remain frozen evidence.

Required CMNIST protocol:

- Binary label is `digit >= 5`.
- Source correlations are `[0.9, 0.8]`; OOD target correlation is `0.1`.
- Train the existing small CNN end-to-end; do not freeze the representation.
- Primary comparison is `LOCAL_RESPONSE` versus `UNPRECONDITIONED_GRAD_ALIGN`.
- Target accuracy/loss are evaluation-only and cannot select beta, checkpoint, architecture, duration, source batches, method, or ablation.

Known exclusions:

- Do not restore or use `TASK3R-ALGORITHMIZATION` as active evidence.
- Do not estimate `A_rec` from source optima or force `E/rho_slack` into training.
- Do not claim causal identification, target-risk lower bounds, finite-sample guarantees, universal DG, or algorithmic novelty without prior-art support.

Relevant code entrypoints:

- `configs/cmnist_vis_001_main.json`
- `src/ood_repr_reg/cmnist_feature_probe.py`
- `src/ood_repr_reg/run_cmnist_feature_probe.py`
- `src/ood_repr_reg/cmnist_geometry_bridge.py`

Historical reopen: none.
