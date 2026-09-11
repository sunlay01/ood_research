# Proposed State Delta

Task: `TASK-AOPI-MULTIMETHOD-MECHANISM-SURVEY`

Old audit: `AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH`

Superseded repair: `b6c9eaf` is invalid because base probabilities were added twice.

Survey verdict: `MULTIMETHOD-MECHANISM-SURVEY-PARTIAL`

All four methods (`ERM`, `IRMv1`, `VREX`, `CORAL`) and five seeds completed. V-REx has been repaired from weak-REx to the official-style source-risk-gap objective with `lambda=10000`, anneal `100`, Adam reset, and post-anneal whole-loss rescale. R5 world gates passed with `rank(A)=5`, `rank(O)=3`, and `O e3 = O e5 = 0`; full-response replay matched all 660 rows; blind grouping used seed-resampled signatures with silhouette `0.3881` and mean ARI `1.0`.

Post-repair mean target accuracy is `IRMv1=66.91%`, `VREX=55.39%`, `ERM=10.98%`, and `CORAL=10.96%`. V-REx is materially repaired relative to the weak variant, but this run still records it as descriptive survey evidence rather than a literature-grade baseline claim.

Algorithm structure has been refactored so survey method semantics live in dedicated files under `task3_aopi_multimethod_mechanism_survey/algorithms/` (`erm.py`, `irmv1.py`, `vrex.py`, `coral.py`) with a formula-free `registry.py`. Training and `Pi_full` continuation both dispatch through the same algorithm interface.

No canonical state changes are proposed. This is descriptive evidence only; no semantic mechanism recovery, theory validation, algorithm claim, or Task 3 promotion is authorized by this survey alone.
