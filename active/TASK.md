# TASK3-CMNIST-LOCAL-RESPONSE

task_id: `TASK3-CMNIST-LOCAL-RESPONSE`

goal: `Test whether curvature-aware local cross-environment head-gradient response geometry improves end-to-end CMNIST DG training beyond ERM, IRMv1, V-REx, and unpreconditioned gradient alignment.`

state_write_authorized: false

required_theorem_ids:

- `T-3A-A`
- `T-3C-PI`
- `T-3D-IDENTIFIABILITY`
- `T-3E-INFO-FLOOR`
- `T-3E-SPECTRAL-SLACK`
- `T-3E-AFFINE-REGRET`

required_result_ids:

- `R-REPAIR-GATE`
- `R-TASK1-MECHANISM`
- `R-TASK2-SHARP`

required_decision_ids:

- `D-ACTUAL-VS-COMMON-BASE-SEPARATION`
- `D-SHARP-SLACK-REPLACES-E0`
- `D-REG-EXPOSURE-CONTAINMENT-REJECTED`

allowed_initial_files:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `active/CONTEXT.md`
- `active/TASK.md`
- referenced registry entries
- current CMNIST data/model source files

allowed_code_roots:

- `src/ood_repr_reg/task3_cmnist_local_response/`
- `src/ood_repr_reg/run_task3_cmnist_local_response.py`
- `tests/test_task3_cmnist_local_response.py`

historical_reopen_policy:

- Use `AGENTS.md`; record `REOPEN_REASON:` before reopening history.
- Do not read old Task3R results as evidence for this task.

success_gate:

- Preregistered design is written before new LOCAL_RESPONSE target outcomes.
- CMNIST training is end-to-end, not frozen-head.
- Source-only selection is used for beta and checkpoint choice.
- Verdict is one of `TASK3-CMNIST-SUPPORT`, `TASK3-CMNIST-PARTIAL`, or `TASK3-CMNIST-FAIL`.

required_outputs:

- `round3_redesign/task3_cmnist_local_response/`
- `active/STATE_DELTA.md`

state_delta_policy:

- Canonical state is not edited; proposed changes go to `active/STATE_DELTA.md`.
