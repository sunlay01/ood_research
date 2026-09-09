# Context Management

This file is retained as a compatibility pointer. It is not a separate research
state document.

## Boot Policy

Use the root boot protocol in [AGENTS.md](AGENTS.md):

1. Read `AGENTS.md`.
2. Read `CURRENT_STATE.md`.
3. Read `active/CONTEXT.md` and `active/TASK.md` only if they exist.
4. Open theorem, result, and decision registry entries only as referenced.
5. Open directly relevant source and test files for the active implementation.

Do not begin a task by reconstructing research history from old reports. The
historical corpus is evidence and provenance, not default current-state input.

## State Writes

Future tasks may update canonical state only when their `active/TASK.md` sets
`state_write_authorized: true`. Otherwise they must write proposed state changes
to `active/STATE_DELTA.md`.
