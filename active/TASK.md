# TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR

task_id: `TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR`

goal: `Repair the CMNIST A/O/Pi audit with smooth tangent expectations, corrected A/O, secondary Pi_head, and primary finite-time Pi_full.`

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

- Preserve the old audit as historical evidence; do not overwrite it.
- Use exactly three primary tangent directions and smooth weighted expectations.
- Treat frozen-head Pi as secondary and full-network finite-time response as primary.
- Verify checkpoint identity and write preregistration before audit metrics.
- Source fitting, `H_S`, `O_S`, and `Pi` are source-only; held-out target data defines A only post-hoc.
- Do not add algorithms, regularizers, semantic clustering, target tuning, or canonical-state edits.

outputs:

- `round3_redesign/task3_aopi_cmnist_reinstantiation_repair/`
- `src/ood_repr_reg/task3_aopi_cmnist_reinstantiation_repair/`
- `src/ood_repr_reg/run_task3_aopi_cmnist_reinstantiation_repair.py`
- `tests/test_task3_aopi_cmnist_reinstantiation_repair.py`
- `active/STATE_DELTA.md`

completion verdict enum:

- `AOPI-REPAIR-PASS`
- `AOPI-REPAIR-PARTIAL`
- `AOPI-REPAIR-FAIL`
- `AOPI-REPAIR-INVALID`

Interpretation ceiling: validity repair only; no new algorithm, theory validation, causal claim, or universal DG claim.
