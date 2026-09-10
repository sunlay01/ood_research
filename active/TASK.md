# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT

task_id: `TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT`

goal: `Port counterfactual color diagnostics onto the corrected CPU-minimal ColoredMNIST ERM/IRMv1 runs and decompose the observed target-accuracy gap into representation content versus final-head color usage.`

state_write_authorized: false

scientific_status: diagnostic / evidence-cleanup only

allowed methods:

- `ERM`
- `IRMv1`

primary seeds:

- `10`
- `11`
- `12`
- `13`
- `14`

allowed files:

- `active/TASK.md`
- `active/CONTEXT.md`
- `active/STATE_DELTA.md`
- `src/ood_repr_reg/task3_cmnist_counterfactual_audit/`
- `src/ood_repr_reg/run_task3_cmnist_counterfactual_audit.py`
- `tests/test_task3_cmnist_counterfactual_audit.py`
- `round3_redesign/task3_cmnist_counterfactual_audit/`

hard constraints:

- Use `configs/task3_cmnist_cpu_minimal.json` as the single source of truth.
- Use corrected CPU-minimal `build_task3_data`, `build_model_from_config`, and `train_one_method` only.
- If checkpoints are absent, reconstruct only ERM/IRMv1 seeds `10..14` and reconcile against existing corrected `main_runs.csv` with `1e-6` tolerance.
- Build counterfactuals from held-out target images after training; original target color is not used for intervention construction.
- Do not add or evaluate GRAD, LOCAL_RESPONSE, IGA, Fish, Fishr, V-REx, CORAL, MLDG, new objectives, hyperparameter sweeps, or target tuning.
- Do not modify `CURRENT_STATE.md` or canonical state registries.

completion verdict enum:

- `REPRESENTATION-DOMINANT`
- `HEAD-USAGE-DOMINANT`
- `MIXED-DECOMPOSITION`
- `DESCRIPTIVE-INCONCLUSIVE`
- `AUDIT-INVALID`

Historical reopen: old CMNIST diagnostic code may be read for mathematical reference only; old empirical results/checkpoints are not evidence for this task.
