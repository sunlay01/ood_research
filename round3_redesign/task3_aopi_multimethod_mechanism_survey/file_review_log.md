# Per-file adversarial review log

This log records the required review after each file write. A file is not
accepted because tests are green alone; its source/target/method boundaries,
coordinate semantics, rank bounds, and failure modes are reviewed explicitly.

## Configuration

- file: `configs/task3_aopi_multimethod_mechanism_survey.json`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Are all primary values fixed? Is the world a five-dimensional displacement? Are target selection and hidden grids prohibited? Are the blind-group thresholds explicit?
- findings: The base probabilities, methods, seeds, architecture, V-REx/CORAL coefficients, response horizons, bank sizes, grouping rule, and no-target-selection flags are explicit. No adaptive or target-based key is present. Derived directions are explicitly excluded from the basis.
- fixes made: none.
- final verdict: `PASS`

## Final result-layer review

- files: `signatures.py`, `blind_grouping.py`, `analysis.py`, `run_task3_aopi_multimethod_mechanism_survey.py`
- current commit/worktree: `d13d1c9a6e285d592c6f611e772544d7dffdcafc` plus the uncommitted survey track
- review: the final implementation aggregates signatures over all five seeds, resamples seed-level signatures for stability, keeps A/O on five primary columns, preserves source-only method-independent O construction, and records distinct reference-manifest versus generated-checkpoint hashes.
- adversarial counterexample: a last-seed-only signature or a target-performance column could produce a plausible cluster; neither enters the final grouping path. A rank-inflated O or nonzero target-only column fails the world gate rather than being clipped.
- final verdict: `PASS`
- unresolved limitation: CORAL remains conditional on the runtime fidelity gate as required.

## Configuration schema

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/config_schema.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Can malformed base/world/target/grid settings enter silently? Are explicit false target guards accepted? Are all fixed objective and grouping values checked?
- findings: Initial review caught an overbroad forbidden-key set that rejected the required explicit false target guards. It was narrowed to forbidden target-selection/grid keys; the legal guards remain required false. Manual malformed configs for dimension 11, probability 1.1, V-REx list lambda, and target grid are rejected.
- fixes made: narrowed `FORBIDDEN_SELECTION_KEYS` and reran compile plus adversarial malformed-config checks.
- final verdict: `PASS`
- unresolved limitation: Validation is schema-level; runtime stage gates must still verify checkpoint and row completeness.

## Method objectives

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/method_objectives.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Does ERM/IRM dispatch remain canonical? Is V-REx exactly population risk variance? Does CORAL use encoder features, centered covariance, `n-1`, and fixed dimension normalization? Can target tensors enter?
- findings: ERM/IRM call the existing CPU-minimal objective. V-REx uses `mean((R-mean(R))**2)` with the fixed pre/post weights. CORAL uses the two encoder feature matrices, centered covariance with `n-1`, and the declared `d` normalizations. No target/evaluation argument exists.
- fixes made: none.
- final verdict: `PASS`
- unresolved limitation: V-REx's fixed post-anneal weight is represented by the preregistered post weight; the scalar lambda is validated separately.

## Method trainer

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/method_trainer.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Are parameters and source schedules shared? Does V-REx reset Adam exactly once at step 100 without resetting model parameters? Is the final optimizer state paired with the final model? Can post-hoc data enter?
- findings: The trainer accepts only source environments and a source batch schedule. It creates Adam once, recreates it only at the fixed V-REx transition, and returns model/optimizer hashes plus reset count. A static audit initially matched the word `evaluation` in a docstring; that ambiguity was removed and the guard rerun.
- fixes made: clarified source-only docstring; no algorithmic change.
- final verdict: `PASS`
- unresolved limitation: Runtime must still prove that every required seed/method row is present.

## Smooth R5 world

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/smooth_world5.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Does zero displacement recover `(0.2,0.1,0.9,0.25,0.25)`? Are source/evaluation color and noise independent? Are there exactly five basis columns and six derived validation directions? Are primary derivatives smooth expectations?
- findings: Manual checks for zero, `0.01e1`, `0.01e3`, and `0.01e5` recover the declared probabilities. Four outcome weights are valid and sum to one. Primary/derived vectors are stored separately, and balanced subsampling uses clean digit labels only.
- fixes made: none.
- final verdict: `PASS`
- unresolved limitation: Semantic names coexist in this world module for the post-hoc mapping; blindness is enforced by the signatures/grouping APIs and tests, which receive only opaque IDs.

## Task response

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/task_response.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Does the object implement `A=H_S^(-1/2)(J_T-J_S)` in 65-dimensional head coordinates? Are source/evaluation derivatives separated? Is the source Hessian used and is rank bounded by R5?
- findings: Source and evaluation weighted expectations are independently constructed, subtraction precedes whitening, and the primary matrix has five columns. The Hessian is the source-risk Hessian in the augmented final-head coordinate. No finite-difference thresholded data path is used.
- fixes made: none.
- final verdict: `PASS`
- unresolved limitation: Evaluation pools are intentionally used here for A only; the runner must keep their metrics out of training and grouping.

## Source observation

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/source_observation.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Is O built only from source risk gradients, with no method argument or penalty? Are the evaluation-only columns null? Is rank checked rather than clipped?
- findings: The function signature has no method input, uses only source pools, and creates five columns through JVPs. Evaluation-only columns are structurally zero because source parameters do not read them. Rank values above three raise immediately; no clipping is applied.
- fixes made: none.
- final verdict: `PASS`
- unresolved limitation: The dataclass field `target_only_column_norms` is descriptive naming only; it is not used in any training or grouping path.

## Functional banks

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/functional_banks.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Are source and post-hoc counterfactual banks separate? Does each red/green pair use the same raw digit image? Is clean-label balance explicit? Can bank measurements mutate training state?
- findings: The source bank includes both source pools. The counterfactual bank constructs red and green from the same evaluation raw image while ignoring sampled color, and the clean-task bank averages them. Static and toy checks confirm equal grayscale content and clean-label balance. Only frozen inference is used for bank logits.
- fixes made: none. The initial manual check used the wrong original index order; it was corrected without changing implementation.
- final verdict: `PASS`
- unresolved limitation: Evaluation images are intentionally present for post-hoc measurement; runner boundaries must keep them out of optimization.

## Package entry point

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/__init__.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Are task identity, method panel, seed panel, and verdict enum centralized without adding algorithm semantics?
- findings: The entry point exposes only fixed identifiers and the four allowed verdict strings. It does not import training, target data, or historical artifacts.
- fixes made: none.
- final verdict: `PASS`
- unresolved limitation: Runtime inclusion of CORAL remains governed by F0-F9.

## Full response

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/full_response.py`
- starting commit: `d13d1c9a6e285d592c6f611e772544d7dffdcafc`
- review questions: Are plus/minus/control clones initialized from the real final Adam state? Are continuation objectives method-faithful and source-only? Are target-only directions prevented from learner updates? Is replay exact?
- findings: The initial review found unnecessary repeated continuations for each horizon and insufficient support for derived directions. The file was repaired to run each path once to K=20, take snapshots at K=1/5/20, derive six directions from the three source-exposed basis responses, and emit zero response for evaluation-only directions. CORAL uses the declared unbiased covariance correction in its smooth weighted batch path. Replay hashes and functional vectors are compared.
- fixes made: consolidated horizon replay and added linear-combination derived directions; no target data is used by continuation objectives.
- final verdict: `PASS`
- unresolved limitation: Full response is finite-time and descriptive; it is not an equilibrium derivative or an OOD utility score.

## Signatures

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/signatures.py`
- review: fixed opaque direction IDs and method-internal source-exposed normalization; no target metric or semantic label enters signature construction. A plausible failure is a missing method row silently becoming zero, so the runner enforces complete 20-model geometry and response row counts before grouping.
- boundary check: source/evaluation values enter only through already-computed A/O/functional rows; method is used only as an internal code. No thresholded finite difference or method-specific O is introduced.
- final verdict: `PASS`

## Blind grouping

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/blind_grouping.py`
- review: grouping consumes opaque IDs and numeric signatures, applies fixed column z-score, Euclidean distance, average linkage, candidate k 2/3/4, silhouette and 200 perturbation repetitions. Semantic names are assigned only after labels are frozen in the runner.
- boundary check: target accuracy and method success are not accepted by the grouping API. A plausible counterexample is a constant signature column; standardization fixes its scale to one without changing other columns, and the stability gate prevents overclaiming.
- final verdict: `PASS`

## Analysis

- file: `src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/analysis.py`
- review: A/O require exactly five primary columns; derived directions are evaluated as linear combinations and rank limits are explicit. `O e3` and `O e5` are hard zero checks. Verdict is descriptive-only and never returns PASS.
- boundary check: source and evaluation roles are already separated in task geometry; no target value enters grouping or selection. A plausible failure is rank inflation or nonlinear derived response, which invalidates Stage 1 instead of being clipped.
- final verdict: `PASS`

## Runner

- file: `src/ood_repr_reg/run_task3_aopi_multimethod_mechanism_survey.py`
- review: writes preregistration before metrics, trains all fixed methods/seeds from shared source schedules, keeps target evaluation post-hoc, writes isolated results, emits heartbeat logs, and stops on fidelity/world failures. ERM/IRM reference hashes are checked; no method is removed for poor target accuracy.
- boundary check: method objectives receive source batches only; target is passed only to evaluation. Full responses use source smooth continuations, fixed horizons, and replay hashes. A plausible failure is an incomplete timeout run; it is recorded as invalid rather than silently interpreted.
- final verdict: `PASS`

## Algorithm panel expansion

- task: `TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK`
- files: `algorithms/base.py`, `fishr.py`, `mldg.py`, `weight_nuclear.py`, `feature_nuclear.py`, `stable_rank.py`, `registry.py`, `method_trainer.py`, `full_response.py`, `signatures.py`, `config_schema.py`, and `run_task3_aopi_multimethod_mechanism_survey.py`
- worktree basis: expansion implemented after `bb31c19fe709df44e84415a779fb66762045212e`; canonical state files and old repair artifacts were not edited.
- mathematical objects and dimensions: default algorithm step interface owns `AlgorithmState`, `StepResult`, and `SmoothStepResult`; Fishr uses 65D classifier-parameter per-example gradients; MLDG uses first-order source env meta-train/meta-test roles; weight nuclear penalizes only the two encoder linear matrices; feature nuclear applies positive source-feature nuclear norm; stable rank remains diagnostic-only.
- boundary checks: training paths consume source batches only; `full_response.py` clones final model, Adam state, and algorithm state, then calls `algorithm.smooth_train_step()`; no new method-specific math lives in runner/full-response; `signatures.py` generates dynamic opaque method codes and does not use target accuracy.
- displacement/base semantics: `smooth_world5.py`, `task_response.py`, `source_observation.py`, and functional banks were preserved; R5 base identity and `O e3/e5 = 0` remain enforced by tests and runner gates.
- rank and derived-direction checks: primary A/O rank uses only e1..e5; derived directions remain linear-combination diagnostics and do not add rank columns.
- counterexample considered: a plausible but wrong expansion would duplicate Fishr/MLDG math inside `full_response.py` or hard-code eight method strings in signature construction. Regression tests now check method-owned interfaces, dynamic codes, and deletion of `method_objectives.py`.
- tests: focused expansion tests plus prior multimethod, project-state, CPU-minimal, and counterfactual regression tests passed after the expanded run.
- final verdict: `PASS`
