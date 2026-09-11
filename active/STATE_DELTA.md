# Proposed State Delta

Task: `TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK`

Verdict: `ALGORITHM-PANEL-EXPANSION-PARTIAL`

The modular CMNIST A/O/Pi survey has been expanded from four methods to eight runnable methods: `ERM`, `IRMv1`, `VREX`, `CORAL`, `FISHR`, `MLDG`, `WEIGHT_NUCLEAR`, and `FEATURE_NUCLEAR`. `STABLE_RANK` remains diagnostic-only and is not registered as a learner.

Implementation changes:

- Algorithm semantics now live in one file per method under `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/algorithms/`.
- `AlgorithmState`, `StepResult`, and `SmoothStepResult` were added; both training and `Pi_full` continuation call algorithm-owned step interfaces.
- Obsolete centralized compatibility shim `method_objectives.py` was removed after all imports migrated.
- Dynamic method-code generation replaced the old four-method signature map.
- New outputs were added: `results/new_method_admission.csv` and `results/weight_spectrum_diagnostics.csv`.

Expanded run summary:

- All eight methods and five seeds completed with no runner errors.
- Geometry gates passed: each model row has `rank(A)=5`, `rank(O)=3`, and `O e3 = O e5 = 0`.
- `Pi_full` replay completed for all admitted methods: `1320` response rows.
- Blind grouping remained stable: `STAGE4-PASS`, silhouette `0.3807331450`, bootstrap mean ARI `1.0`.
- Row counts: `method_fidelity=40`, `method_performance=40`, `geometry_A=440`, `geometry_O=440`, `pi_full=1320`, `normalized_response=1320`, `mechanism_signatures=11`.

Mean target accuracy across seeds `10..14`:

- `IRMv1`: `66.91%`
- `FISHR`: `55.88%`
- `VREX`: `55.31%`
- `MLDG`: `11.22%`
- `FEATURE_NUCLEAR`: `11.01%`
- `ERM`: `10.98%`
- `CORAL`: `10.93%`
- `WEIGHT_NUCLEAR`: `10.75%`

Interpretation remains descriptive only. This expansion does not establish semantic mechanism recovery, causal feature recovery, a universal DG taxonomy, a new algorithm, or theory validation. No canonical state changes are proposed.
