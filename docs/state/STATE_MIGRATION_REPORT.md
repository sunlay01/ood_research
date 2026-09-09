# State Migration Report

## A. Pre-Migration Conflict

The following authority-looking files were stale before this migration:

| file | stale issue | migration action |
| --- | --- | --- |
| `README.md` | Presented semantic-latent decomposition as current. | Converted to landing page. |
| `CONTEXT_MANAGEMENT.md` | Listed old state files as default truth and LATENT-001 as active. | Converted to boot-policy pointer. |
| `PROJECT_PROMPT_CN.md` | Hard-coded old semantic-latent workflow. | Converted to stable collaboration policy. |
| `docs/research/research_state.md` | Maintained a competing current-state summary. | Converted to compatibility pointer. |
| `docs/research/open_questions.md` | Contained old LATENT-era queue. | Migrated to Task 3/4/5 only. |

Historical reports and round directories were not deleted or moved.

## B. New Authority Hierarchy

Default boot sequence:

1. `AGENTS.md`
2. `CURRENT_STATE.md`
3. `active/CONTEXT.md` if an active task exists
4. `active/TASK.md` if an active task exists
5. Referenced theorem/result/decision registry entries only
6. Directly relevant source/test files

Single source of truth for current state: `CURRENT_STATE.md`.

## C. Current Scientific Snapshot

| object/theorem | status | canonical ID | evidence |
| --- | --- | --- | --- |
| 3A source-whitened response `A` | FROZEN | `T-3A-A` | `R-LEGACY-HIDDEN-U`, `R-SOURCE-INDUCED` |
| 3C exact IFT response `Pi` | FROZEN | `T-3C-PI` | `R-TASK1-MECHANISM` |
| 3C common-base attribution | FROZEN | `T-3C-COMMON-BASE` | `R-TASK1-MECHANISM` |
| 3D source identifiability | FROZEN | `T-3D-IDENTIFIABILITY` | `R-LEGACY-HIDDEN-U`, `R-SOURCE-INDUCED` |
| 3E information floor | FROZEN | `T-3E-INFO-FLOOR` | `R-TASK2-SHARP` |
| 3E spectral slack | FROZEN | `T-3E-SPECTRAL-SLACK` | `R-TASK2-SHARP` |
| 3E full affine regret | FROZEN | `T-3E-AFFINE-REGRET` | `R-TASK2-SHARP` |
| old `E=0` necessity | SUPERSEDED | `T-OLD-E0-SHARP` | `D-SHARP-SLACK-REPLACES-E0` |
| regularizer/exposure containment necessity | REJECTED | `T-REG-EXPOSURE-CONTAINMENT` | `D-REG-EXPOSURE-CONTAINMENT-REJECTED` |

Task 1 and Task 2 are completed. Task 3 applicability is next and was not
executed by this migration.

## D. Historical Classifications

| path class | status | default_read |
| --- | --- | --- |
| `round1/**` | HISTORICAL | false |
| `round2/**` | HISTORICAL | false |
| `round3/**` | HISTORICAL | false |
| `round3_redesign/**` | FROZEN_EVIDENCE | false |
| `docs/research/research_state.md` | DO_NOT_READ_BY_DEFAULT | false |
| `MECH-001/C011` | DO_NOT_RESTORE | false |

Repair-evidence JSON/CSV files remain accessible through `RESULT_REGISTRY.json`.

## E. Context Reduction

Default boot context when no active task exists:

| files | line count | byte count | approximate token count |
| ---: | ---: | ---: | ---: |
| 2 | 188 | 7477 | 1869 |

Before migration, agents were directed toward multiple old current-state files
and then into historical report reconstruction. After migration, history is
opened only by explicit registry pointer or `REOPEN_REASON`.

## F. Validation

Executed validation commands:

```text
python scripts/check_project_state.py                  # ok=true, 0 errors
PYTHONPATH=src pytest -q tests/test_project_state.py   # 5 passed
PYTHONPATH=src pytest -q                               # 264 passed
```

The checker validates required files, JSON registries, unique theorem and
decision IDs, artifact paths, stale-authority claims, Task 3 next-state wording,
and the do-not-restore marker.

## G. Remaining Ambiguity

No substantial historical reopening was performed. The migration relies on the
repair evidence artifacts at commit `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`
and the current-state facts supplied by the migration task.

Historical reopen: none.

## H. Final Verdict

`STATE-MANAGEMENT-PASS`

## Final Questions

1. Single source of truth: `CURRENT_STATE.md`.
2. New sessions read first: `AGENTS.md`, then `CURRENT_STATE.md`.
3. Default boot context: 2 files when no active task exists.
4. Not read by default: `round1/**`, `round2/**`, `round3/**`, `round3_redesign/**`, and stale compatibility files.
5. Historical reopening requires a recorded `REOPEN_REASON` under `AGENTS.md`.
6. Theorem status is in `docs/state/THEOREM_REGISTRY.md`.
7. Numerical artifacts are located through `docs/state/RESULT_REGISTRY.json`.
8. Killed ideas are protected by `DECISION_LOG.md` and `FILE_STATUS.json`.
9. Future state changes require `state_write_authorized: true`.
10. Unapproved state changes go to `active/STATE_DELTA.md`.
11. Stale files are listed in section A.
12. LATENT-001 is clearly historical.
13. The current `A,O_S,Pi,E,S_slack` spine is represented in `CURRENT_STATE.md` and `THEOREM_REGISTRY.md`.
14. Task 1 and Task 2 are recorded completed.
15. Task 3 is recorded as next but not executed.
16. Current state can be reconstructed without Round 1 / Round 2 archaeology.
17. No history was deleted.
18. The new state system is compact relative to the historical corpus.
