# Task Template

task_id: `<required>`

goal: `<one sentence>`

state_write_authorized: `false`

required_theorem_ids:

- `<T-ID>`

required_result_ids:

- `<R-ID>`

required_decision_ids:

- `<D-ID>`

allowed_initial_files:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `active/CONTEXT.md`
- `active/TASK.md`

allowed_code_roots:

- `<source-or-test-root>`

historical_reopen_policy:

- Use `AGENTS.md` reopen policy.
- Record `REOPEN_REASON:` before opening frozen or historical artifacts.
- Do not reconstruct project history.

success_gate:

- `<tests, reports, or evidence required>`

required_outputs:

- `<paths or artifact classes>`

state_delta_policy:

- If `state_write_authorized: false`, write proposed state changes to
  `active/STATE_DELTA.md` instead of editing canonical state.
