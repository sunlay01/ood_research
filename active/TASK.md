# TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK

task_id: `TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK`

goal: `Extend the modular CMNIST A/O/Pi survey from ERM, IRMv1, VREX, and CORAL to Fishr, MLDG, weight-nuclear, feature-nuclear, and stable-rank diagnostics.`

state_write_authorized: false

scientific_status: validation / falsification gate only

methods: ERM, IRMv1, VREX, CORAL, FISHR, MLDG, WEIGHT_NUCLEAR, FEATURE_NUCLEAR

primary seeds: 10, 11, 12, 13, 14

inputs: `configs/task3_aopi_multimethod_mechanism_survey.json`, corrected CPU-minimal data/model code, existing modular survey algorithms, and the accepted ERM/IRM reconstruction manifest.

hard constraints:

- Preserve CMNIST data/model semantics, R5 smooth displacement world, A/O geometry, functional banks, normalization, and blind grouping semantics.
- Keep one algorithm file per runnable method; training and Pi_full continuation call algorithm-owned interfaces.
- Do not add OURS, HA, CMA, target tuning, target method selection, target grouping, or canonical-state edits.
- Admit new methods to A/O/Pi only through source-only training and smooth-continuation fidelity.
- Overall scientific interpretation is capped at `ALGORITHM-PANEL-EXPANSION-PARTIAL`.

outputs: `round3_redesign/task3_aopi_multimethod_mechanism_survey/`

Interpretation ceiling: descriptive response differences only; no semantic mechanism recovery, causal claim, theory validation, new algorithm claim, or universal DG taxonomy.
