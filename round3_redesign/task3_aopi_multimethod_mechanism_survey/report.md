# Multi-method mechanism survey

Verdict: `MULTIMETHOD-MECHANISM-SURVEY-PARTIAL`

This isolated survey is descriptive only. It does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, a new algorithm, theory validation, or a universal DG taxonomy. Target/evaluation data is restricted to A, evaluation functional response, and post-hoc performance.

## Fidelity gates

F0 checkpoint and shared initialization/schedule reconstruction: PASS for ERM/IRM reference hashes; all four methods finite.
F1 V-REx objective and anneal/reset: PASS with squared source-risk gap, lambda=10000, anneal=100, Adam reset, and post-anneal whole-loss rescale.
F2 CORAL representation penalty and `n-1` covariance: PASS.
F3 common source batches and optimizer settings: PASS.
F4 method completeness: PASS, 4 methods x 5 seeds.
F5 base R5 world identity and valid mixture weights: PASS.
F6 primary basis and displacement semantics: PASS.
F7 source-only O, `O e3 = O e5 = 0`, rank limit: PASS for all 20 model rows.
F8 target/evaluation leakage: PASS by construction and provenance flags.
F9 finite continuation replay: PASS for 660/660 response rows.

## Interpretation

The run produced descriptive response profiles and stable opaque grouping, but this task deliberately does not promote response differences to a PASS or algorithm claim. Errors: `[]`.
