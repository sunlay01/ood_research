# TASK3-BASELINE-FIDELITY-RECOVERY

task_id: `TASK3-BASELINE-FIDELITY-RECOVERY`

goal: `Recover and audit ERM, IRMv1, IGA, and Fish baseline identities before any Task 3 scientific verdict is interpreted.`

state_write_authorized: false

required_result_ids:

- `R-REPAIR-GATE`
- `R-TASK1-MECHANISM`
- `R-TASK2-SHARP`

allowed_initial_files:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `active/CONTEXT.md`
- `active/TASK.md`
- pinned upstream baseline code under `artifacts/baseline_fidelity/upstreams/`

allowed_code_roots:

- `src/ood_repr_reg/task3_baseline_fidelity/`
- `src/ood_repr_reg/run_task3_baseline_fidelity.py`
- `tests/test_baseline_fidelity.py`

historical_reopen_policy:

- Use `AGENTS.md`; record `REOPEN_REASON:` before reopening history.
- Do not use Task 3 local-response outputs as scientific evidence in this gate.

success_gate:

- Pinned upstream commits are recorded.
- Facebook IRM Colored MNIST recovery is preserved or rerun.
- IGA is full-network gradient alignment, not head-only gradient variance.
- Fish uses clone, sequential inner-domain updates, carried inner optimizer state, and outer interpolation.
- `HEAD_GRADIENT_VARIANCE_SURROGATE` is explicitly blocked from being reported as IGA or Fish.
- Final verdict is one of `BASELINE-FIDELITY-PASS`, `BASELINE-FIDELITY-PARTIAL`, or `BASELINE-FIDELITY-FAIL`.

required_outputs:

- `round3_redesign/task3_baseline_fidelity/`
- `round3_redesign/task3_baseline_fidelity/results/`
- `active/STATE_DELTA.md`

state_delta_policy:

- Canonical state is not edited; proposed changes go to `active/STATE_DELTA.md`.
