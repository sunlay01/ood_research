# Proposed State Delta

Task: `TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL`

Verdict: `SPECTRAL-FLATNESS-PANEL-PARTIAL`

The modular CMNIST A/O/Pi survey has been extended into a common-budget spectral and flatness panel. The run is descriptive only: it does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, paper benchmark reproduction, a new algorithm, theory validation, low rank as a cause of OOD, flatness as a cause of OOD, or a universal DG taxonomy.

Implementation changes:

- Added one-file algorithm modules for `SPECTRAL_NORM_REG`, `SPECTRAL_REG_2024`, `SVB_ORTHDNN`, `STABLE_RANK_NORM`, `SAM`, and `ASAM`.
- Added deferred candidates `SVD_SPARSE`, `FAD`, and `DISAM` with explicit reference-audit notes rather than surrogate implementations.
- Retired `WEIGHT_NUCLEAR` and `FEATURE_NUCLEAR` from the default primary panel while preserving them as legacy-only/default-disabled paths.
- Extended the shared algorithm interface with replayable state/admission metadata and compute-equivalent accounting.
- Added spectral/flatness diagnostics for weight spectra, representation spectra, classifier-gradient spectra, Hessian/trace/sharpness, and compute budget.

Run summary:

- Runnable methods: `ERM`, `IRMv1`, `VREX`, `CORAL`, `FISHR`, `MLDG`, `SPECTRAL_NORM_REG`, `SPECTRAL_REG_2024`, `SVB_ORTHDNN`, `STABLE_RANK_NORM`, `SAM`, `ASAM`.
- Candidate methods: the runnable set plus `SVD_SPARSE`, `FAD`, and `DISAM`.
- Seeds: `10..14`.
- All 12 runnable methods completed all five seeds with no runner errors.
- Geometry gates passed for all 60 method/seed rows: `rank(A)=5`, `rank(O)=3`, `O e3 = O e5 = 0`.
- `Pi_full` replay completed for all admitted methods: `1980` response rows.
- Blind grouping status: `STAGE4-PASS`, silhouette `0.4288668582`, bootstrap mean ARI `1.0`.
- Output row counts: `method_fidelity=60`, `method_performance=60`, `geometry_A=660`, `geometry_O=660`, `pi_full=1980`, `normalized_response=1980`, `weight_spectrum_long=1080`, `representation_spectrum=360`, `gradient_spectrum=60`, `flatness_diagnostics=120`, `compute_budget=60`, `spectral_flatness_admission=15`.

Mean target accuracy across seeds `10..14`:

- `IRMv1`: `66.91%`
- `FISHR`: `55.88%`
- `VREX`: `55.31%`
- `MLDG`: `11.22%`
- `ERM`: `10.98%`
- `CORAL`: `10.93%`
- `ASAM`: `10.62%`
- `SPECTRAL_NORM_REG`: `10.56%`
- `SAM`: `10.45%`
- `SPECTRAL_REG_2024`: `10.31%`
- `STABLE_RANK_NORM`: `10.20%`
- `SVB_ORTHDNN`: `10.17%`

Diagnostic highlights:

- Largest encoder spectral-norm reduction: `STABLE_RANK_NORM`.
- Highest encoder effective rank / stable rank: `SVB_ORTHDNN`.
- Highest representation effective rank: `FISHR`.
- Highest classifier-gradient effective rank: `SVB_ORTHDNN`.
- Lowest Hessian top eigenvalue: `FISHR`.
- Lowest Hessian trace estimate: `VREX`.
- Lowest SAM-style sharpness at `rho=0.05`: `SAM`.
- Counterexamples remain explicit: low-rank and flatness improvements are not sufficient for OOD success under this harness.

Artifacts written under `round3_redesign/task3_aopi_multimethod_mechanism_survey/` include `spectral_reference_audit.md`, `flatness_reference_audit.md`, `fad_reference_audit.md`, `disam_reference_audit.md`, `report.md`, `final_adversarial_audit.md`, `provenance.json`, and the required `results/*.csv/json` tables.

Canonical state remains unchanged: `CURRENT_STATE.md`, `docs/state/**`, and old repair result directories were not modified.
