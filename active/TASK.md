# TASK3R-ALGORITHMIZATION

task_id: `TASK3R-ALGORITHMIZATION`

goal: `Derive and audit one source-only pseudo-response objective in the exact Gaussian/quadratic setting.`

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

allowed_code_roots:

- `src/ood_repr_reg/task3r_algorithmization/`
- `src/ood_repr_reg/run_task3r_algorithmization.py`
- `tests/test_task3r_algorithmization.py`

historical_reopen_policy:

- Use `AGENTS.md`; record `REOPEN_REASON:` before reopening history.
- Do not reconstruct old rounds.

success_gate:

- Prior-work bridge is complete before outcome generation.
- Source-only exact Gaussian probe and leakage tests pass.
- Verdict is one of `TASK3R-ALGORITHM-SUPPORT`, `TASK3R-ALGORITHM-PARTIAL`, or `TASK3R-ALGORITHM-FAIL`.

required_outputs:

- `round3_redesign/task3r_algorithmization/`
- `active/STATE_DELTA.md`

state_delta_policy:

- Canonical state is not edited; proposed changes go to `active/STATE_DELTA.md`.
