# Agent Boot Protocol

This file is mandatory boot context for agents working in this repository.

## 1. Default Reading Order

1. Read `AGENTS.md`.
2. Read `CURRENT_STATE.md`.
3. Read `active/CONTEXT.md` if an active task exists.
4. Read `active/TASK.md` if an active task exists.
5. Open theorem, result, and decision registry entries only as referenced.
6. Open directly relevant source and test files for the active implementation.

Do not begin with repository-wide research-history search. The normal research
state boot context should be at most these files plus task-specific source/test
files.

## 2. Authority Hierarchy

Current state authority is:

1. `CURRENT_STATE.md`
2. `docs/state/THEOREM_REGISTRY.md`
3. `docs/state/RESULT_REGISTRY.json`
4. `docs/state/DECISION_LOG.md`
5. `docs/state/FILE_STATUS.json`
6. Historical reports only when explicitly referenced or reopened under policy

`README.md`, `CONTEXT_MANAGEMENT.md`, `PROJECT_PROMPT_CN.md`, and
`docs/research/research_state.md` are compatibility or landing files. They must
not be treated as competing current-state authorities.

## 3. Historical Reopen Policy

Historical or superseded artifacts may be opened only if one of these applies:

- A current regression contradicts the registry.
- The active task explicitly modifies that theorem or result.
- Exact provenance is required.
- The registry identifies the artifact as canonical proof or evidence.
- A dependency cannot be resolved from the authoritative layer.

Before opening one, record:

```text
REOPEN_REASON: <one sentence>
```

Without a reopen reason, do not open historical files merely for safety.

## 4. State Write Policy

Future agents must not silently promote their own output to project truth.

- If `active/TASK.md` has `state_write_authorized: true`, canonical state files
  may be updated and the exact changes must be recorded.
- If `state_write_authorized: false`, write proposed updates to
  `active/STATE_DELTA.md` instead.
- Routine code bug fixes should not update scientific state unless they change a
  theorem, frozen result, or readiness gate.

Canonical state files are:

- `CURRENT_STATE.md`
- `docs/state/THEOREM_REGISTRY.md`
- `docs/state/RESULT_REGISTRY.json`
- `docs/state/DECISION_LOG.md`
- `docs/research/open_questions.md`

## 5. Current Boundaries

- Do not start Task 3 unless explicitly requested by a task file or user request.
- Do not reinterpret Task 1 or Task 2 results while performing state management.
- Do not use historical semantic-latent or Round-2 quotient files as the current
  mainline.
- Do not restore, reconstruct, summarize, or cite deleted `MECH-001/C011`
  material.
- Do not delete, mass-move, or rewrite history to make old reports look current.

## 6. Validation

Run `python scripts/check_project_state.py` after editing canonical state. Fast
tests for this layer live in `tests/test_project_state.py`.
