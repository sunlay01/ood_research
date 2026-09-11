# TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL

task_id: `TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL`

goal: `Run a common-harness CMNIST spectral and flatness mechanism panel extending the modular A/O/Pi survey.`

state_write_authorized: false

scientific_status: validation / falsification gate only

runnable methods: ERM, IRMv1, VREX, CORAL, FISHR, MLDG, SPECTRAL_NORM_REG, SPECTRAL_REG_2024, SVB_ORTHDNN, STABLE_RANK_NORM, SAM, ASAM

deferred candidates: SVD_SPARSE, FAD, DISAM

legacy-only rank probes: WEIGHT_NUCLEAR, FEATURE_NUCLEAR

primary seeds: 10, 11, 12, 13, 14

hard constraints:

- Preserve CMNIST data/model semantics, R5 smooth displacement world, A/O definitions, functional banks, normalization, and blind grouping.
- Existing ERM, IRMv1, VREX, CORAL, FISHR, and MLDG methods remain frozen.
- Use one algorithm file per runnable method; training and Pi_full continuation call algorithm-owned interfaces.
- Do not use target data for training, tuning, variant selection, method inclusion, normalization, or grouping.
- This is common-budget controlled mechanism comparison, not paper benchmark reproduction.
- Overall scientific verdict cannot be PASS.

outputs: `round3_redesign/task3_aopi_multimethod_mechanism_survey/`

Interpretation ceiling: descriptive spectral, flatness, and A/O/Pi associations only; no causal low-rank/flatness claim, new algorithm claim, or theory validation.
