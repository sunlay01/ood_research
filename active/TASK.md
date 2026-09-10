# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND

task_id: `TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND`

goal: `Test whether first-round OOD capability bottlenecks can be separated on corrected CPU-minimal ColoredMNIST using controlled A/B/C interventions.`

state_write_authorized: false

scientific_status: diagnostic / capability verification only

allowed experiments:

- `A_COVERAGE`
- `B_SEPARABILITY`
- `C_SELECTION`

forbidden experiments:

- `D_SOURCE_SIDE_IDENTIFICATION`
- `E_OPTIMIZATION_RESPONSE_ABILITY`

methods / encoders:

- `ERM`
- `IRMv1`

primary seeds:

- `10`
- `11`
- `12`
- `13`
- `14`

inputs:

- `configs/task3_cmnist_cpu_minimal.json`
- `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv`
- `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoints/`
- corrected CPU-minimal data/model code
- corrected counterfactual probe construction

hard constraints:

- Load existing corrected ERM/IRMv1 checkpoints; do not retrain encoders.
- Verify checkpoint SHA256, config SHA256, parameter hash, model architecture, and existing ERM/IRMv1 target gap before capability analysis.
- Write preregistration before capability metrics.
- Freeze ridge value `1e-3` and color-subspace ranks `[0,1,2,4,8,16,32,64]`.
- Do not add new regularizers, algorithms, datasets, methods, hyperparameter sweeps, target tuning, or canonical state updates.

outputs:

- `round3_redesign/ood_capability_decomposition/`
- `src/ood_repr_reg/task3_ood_capability_decomposition/`
- `src/ood_repr_reg/run_task3_ood_capability_decomposition.py`
- `tests/test_ood_capability_decomposition.py`
- `active/STATE_DELTA.md`

completion verdict enum:

- `FIRST-ROUND-CAPABILITY-ISOLATED`
- `FIRST-ROUND-CAPABILITY-PARTIAL`
- `FIRST-ROUND-CAPABILITY-INCONCLUSIVE`
- `FIRST-ROUND-AUDIT-INVALID`

Historical reopen: none for empirical evidence; this task uses corrected checkpoint artifacts only.
