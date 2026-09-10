# TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT

task_id: `TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT`

goal: `Audit frozen-encoder, trainable-final-head A/O/Pi response geometry on corrected CPU-minimal ColoredMNIST.`

state_write_authorized: false

scientific_status: validation / falsification gate only

methods:

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

hard constraints:

- Freeze the nonlinear encoder and audit only the 65-dimensional final head.
- Verify checkpoint identity and write preregistration before audit metrics.
- Source fitting, `H_S`, `O_S`, and `Pi` are source-only; held-out target data defines A only post-hoc.
- Do not add algorithms, regularizers, semantic clustering, target tuning, or canonical-state edits.

outputs:

- `round3_redesign/task3_aopi_cmnist_reinstantiation/`
- `src/ood_repr_reg/task3_aopi_cmnist_reinstantiation/`
- `src/ood_repr_reg/run_task3_aopi_cmnist_reinstantiation.py`
- `tests/test_task3_aopi_cmnist_reinstantiation.py`
- `active/STATE_DELTA.md`

completion verdict enum:

- `AOPI-REINSTANTIATION-PASS`
- `AOPI-REINSTANTIATION-PARTIAL`
- `AOPI-REINSTANTIATION-FAIL`
- `AOPI-AUDIT-INVALID`

Interpretation ceiling: frozen-encoder head-block local response audit only.
