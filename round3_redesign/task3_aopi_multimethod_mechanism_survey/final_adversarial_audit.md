# Final adversarial audit

Q1. Existing-method regression: ERM/IRMv1/VREX/CORAL/FISHR/MLDG remain on the modular algorithm-owned path; no method-specific math was added to runner/full_response.
Q2. Legacy rank probes: WEIGHT_NUCLEAR and FEATURE_NUCLEAR are removed from the primary panel only and preserved as legacy/default-disabled code paths.
Q3. New algorithm identity: SNR, SR2024, SVB, SRN, SAM, and ASAM are implemented as common-harness variants; SVD-SPARSE, FAD, and DISAM are deferred instead of replaced by fake surrogates.
Q4. Common budget: all runnable methods share model, data, seed, source schedule, batch size, and 501 outer steps; extra intrinsic compute is in `results/compute_budget.csv`.
Q5. Target exclusion: target/evaluation is excluded from training, tuning, method inclusion, normalization, and grouping; provenance flags are all false for those uses.
Q6. Spectral changes: largest top singular value reduction is `STABLE_RANK_NORM` (-0.876); tail/effective rank is highest under `SVB_ORTHDNN` (63.978); stable rank is highest under `SVB_ORTHDNN` (53.623).
Q7. Flatness changes: lowest Hessian top eigenvalue is `FISHR` (0.267), lowest trace is `VREX` (1.418), and lowest SAM sharpness@0.05 is `SAM` (0.012).
Q8. OOD target accuracy: best target mean is `IRMv1` (0.669); worst target mean is `SVB_ORTHDNN` (0.102).
Q9. Spectral without flatness: SVB_ORTHDNN strongly changes singular spectra but is not the flattest by Hessian/sharpness diagnostics.
Q10. Flatness without spectral: SAM/ASAM reduce sharpness proxies relative to ERM without producing a corresponding OOD improvement.
Q11. Lower rank sufficiency: rejected by STABLE_RANK_NORM, which is low-rank/compressed but OOD-poor.
Q12. Lower sharpness sufficiency: rejected by SAM/SR2024-style counterexamples under this harness.
Q13. Gradient effective rank: useful descriptive axis, highest under `SVB_ORTHDNN` (25.165), but not a sufficient separator.
Q14. Pi_full validity: 12 methods are admitted to Pi_full; SVD-SPARSE/FAD/DISAM remain deferred.
Q15. Pi fingerprint: no single common fingerprint is established; response profiles remain heterogeneous across method families.
Q16. Strong spectral counterexample: STABLE_RANK_NORM compresses spectrum heavily but does not improve target accuracy.
Q17. Strong flatness counterexample: SAM/ASAM reduce local sharpness diagnostics but remain ERM-like on target.
Q18. Remaining gaps: no causal claim, semantic recovery claim, source-identifiability claim, paper-benchmark reproduction claim, theory validation, or new algorithm claim is established.

Final verdict: `SPECTRAL-FLATNESS-PANEL-PARTIAL`. Strongest defensible statement: Correct CMNIST can show reproducible descriptive differences in task/source-conditioned response treatment across these DG learners under a fixed common harness.
