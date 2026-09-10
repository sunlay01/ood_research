# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT Context

Current diagnostic question:

- In the corrected CPU-minimal ColoredMNIST pipeline, ERM target accuracy is near `0.11` and IRMv1 target accuracy is near `0.67` on seeds `10..14`.
- This task asks what differs between those trained models: encoder representation content, final-head use of color, or both.

Required source of truth:

- Corrected CPU-minimal config: `configs/task3_cmnist_cpu_minimal.json`.
- Corrected data/model/trainer package: `src/ood_repr_reg/task3_cmnist_cpu_minimal/`.
- Corrected reference run table: `round3_redesign/task3_cmnist_cpu_minimal/results/main_runs.csv`.

Runtime path:

- Reconstruct final `ERM` and `IRMv1` models for seeds `10..14` only if exact checkpoints are absent.
- Reconciliation must match existing source/target/color-agreement metrics with tolerance `1e-6`.
- Diagnostics use the full held-out target split post-hoc and never use target metrics for training or selection.

Forbidden interpretation:

- No causality claim.
- No source-identifiability claim.
- No new objective or algorithm claim.
- No frozen theory validation.
- No novelty claim.
- No restoration of old CMNIST empirical results.

Historical reopen: old CMNIST feature-probe code may be inspected for formulas only; old data construction, training, checkpoints, and results are not runtime inputs.
