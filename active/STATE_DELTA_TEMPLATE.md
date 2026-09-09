# State Delta Template

Use this file when `state_write_authorized: false` and a task produces proposed
changes to canonical state.

task_id: `<required>`

state_write_authorized: `false`

proposed_changes:

- target_file: `<CURRENT_STATE.md | theorem/result/decision registry | open_questions>`
- change_type: `<add | update | supersede | reject>`
- summary: `<one sentence>`
- evidence_path: `<artifact path>`
- validation: `<tests or checks>`

not_applied_reason:

- Canonical state writes were not authorized for this task.

historical_reopen:

- path: `<path or none>`
- REOPEN_REASON: `<one sentence or none>`
- did_it_change_canonical_state: `no`
