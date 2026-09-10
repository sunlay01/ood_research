# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND Context

Current diagnostic question:

- Can the ERM-vs-IRMv1 OOD gap in corrected CPU-minimal ColoredMNIST be separated into coverage, separability, and selection bottlenecks?
- This is not an algorithm task and does not execute source-side identification or optimization geometry.

Trusted inputs:

- Corrected CPU-minimal config: `configs/task3_cmnist_cpu_minimal.json`.
- Corrected ERM/IRMv1 checkpoints: `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoints/`.
- Checkpoint manifest: `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv`.
- Corrected probe construction: `src/ood_repr_reg/task3_cmnist_counterfactual_audit/`.

First-round interventions:

- Coverage: frozen encoder plus deterministic ridge probes.
- Separability: remove estimated color-response subspace ranks `[0,1,2,4,8,16,32,64]`.
- Selection: freeze encoder and retrain head-only ERM, head-only IRMv1, and diagnostic oracle clean heads.

Forbidden interpretation:

- No new regularizer or algorithm claim.
- No source-identifiability claim.
- No causal or additive decomposition claim.
- No frozen theory validation.
- No D/E execution.

Canonical state remains unchanged; completion writes only `active/STATE_DELTA.md`.
