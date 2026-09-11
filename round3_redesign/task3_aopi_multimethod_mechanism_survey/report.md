# Algorithm-panel mechanism survey

Verdict: `ALGORITHM-PANEL-EXPANSION-PARTIAL`

This isolated survey is descriptive only. It does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, a new algorithm, theory validation, or a universal DG taxonomy. Target/evaluation data is restricted to A, evaluation functional response, and post-hoc performance.

Methods: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'WEIGHT_NUCLEAR', 'FEATURE_NUCLEAR']`. Methods admitted to Pi_full: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'WEIGHT_NUCLEAR', 'FEATURE_NUCLEAR']`.

## Fidelity gates

F0 checkpoint and shared initialization/schedule reconstruction: PASS for ERM/IRM reference hashes; all configured methods finite unless listed in errors.
F1 V-REx objective and anneal/reset: PASS with squared source-risk gap, lambda=10000, anneal=100, Adam reset, and post-anneal whole-loss rescale.
F2 CORAL representation penalty and `n-1` covariance: PASS.
F3 Fishr classifier-gradient variance, first-order MLDG, weight nuclear, and feature nuclear modules: PASS when admitted rows are present.
F4 method completeness: PASS, 8 methods x 5 seeds.
F5 base R5 world identity and valid mixture weights: PASS.
F6 primary basis and displacement semantics: PASS.
F7 source-only O, `O e3 = O e5 = 0`, rank limit: PASS for all model rows.
F8 target/evaluation leakage: PASS by construction and provenance flags.
F9 finite continuation replay: PASS for 1320 response rows across 8 admitted methods.

## Interpretation

The run produced descriptive response profiles for the expanded algorithm panel. This task deliberately caps scientific interpretation at `ALGORITHM-PANEL-EXPANSION-PARTIAL`; response differences are not promoted to a PASS or algorithm claim. Errors: `[]`.
