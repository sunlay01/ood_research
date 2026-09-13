# Algorithm branch migration map

The repository is intentionally kept import-compatible. Existing source and result paths are not mass-moved; the directories below provide stable scientific routing and point to canonical implementations.

| Scientific role | Canonical implementation/artifacts | Branch routing |
|---|---|---|
| A/O/Pi response survey | `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/`; `round3_redesign/task3_aopi_multimethod_mechanism_survey/` | `mechanism_representations/aopi/` |
| Full A/O geometry and schedule-transfer audit | `src/ood_repr_reg/run_aopi_full_geometry_predictive_audit.py`; `round3_redesign/semantic_mechanism_bridge/aopi_full_geometry_*` | `mechanism_representations/aopi/` and `reports/audits/` |
| C/K factorial interventions | `src/ood_repr_reg/run_cmnist_ck_factorial.py`; `round3_redesign/method_agnostic_mechanism/ck_factorial/` | `mechanism_representations/ck_factorial/` |
| Functional response geometry | `full_response.py`, `functional_banks.py`, `smooth_world5.py` | `mechanism_representations/response_geometry/` and `infrastructure/functional_banks/` |
| Trajectory probes | `run_vrex_*`, `round3_redesign/vrex_*`, trajectory reports | `mechanism_representations/trajectory/` |
| Memory/mechanism discovery | `src/ood_repr_reg/algorithm_mechanism/`, `algorithm_mechanism_discovery.py`, related artifacts | `mechanism_representations/memory/` |
| Checkpoint bundles and deterministic continuation | `method_trainer.py`, `full_response.py`, `tests/test_aopi_checkpoint_bundle.py` | `infrastructure/checkpoint_bundles/` and `infrastructure/deterministic_continuation/` |
| Semantic perturbation and banks | `smooth_world5.py`, `functional_banks.py`, semantic bridge artifacts | `infrastructure/perturbation_probes/` and `infrastructure/functional_banks/` |
| CMNIST/synthetic experiments | `round3_redesign/task3_*`, CMNIST runners, synthetic audits | `experiments/cmnist/` and `experiments/synthetic/` |
| Matched forks | `matched_fork_cmnist.py`, A/O predictive audits | `experiments/matched_forks/` |
| Reports and negative audits | `round3_redesign/semantic_mechanism_bridge/`, `method_agnostic_mechanism/`, trajectory reports | `reports/positive/`, `reports/negative/`, `reports/audits/` |

The routing directories contain pointers and scope notes, not duplicate implementations. This avoids breaking imports, changing hashes, or rewriting historical provenance. Files that are intentionally left in their existing paths are listed in `archive/legacy_scripts/README.md` and remain the source of truth until a future migration is separately reviewed.
