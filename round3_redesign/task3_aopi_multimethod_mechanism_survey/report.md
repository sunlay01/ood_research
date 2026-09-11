# Spectral and flatness panel

Verdict: `SPECTRAL-FLATNESS-PANEL-PARTIAL`

This isolated survey is descriptive only. It does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, a new algorithm, theory validation, low rank as a cause of OOD, flatness as a cause of OOD, or a universal DG taxonomy. Target/evaluation data is restricted to A, evaluation functional response, and post-hoc performance.

This run is definition-faithful under the fixed common harness. It is not paper benchmark reproduction. Methods: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'SPECTRAL_NORM_REG', 'SPECTRAL_REG_2024', 'SVB_ORTHDNN', 'STABLE_RANK_NORM', 'SAM', 'ASAM']`. Candidate methods: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'SPECTRAL_NORM_REG', 'SPECTRAL_REG_2024', 'SVB_ORTHDNN', 'STABLE_RANK_NORM', 'SVD_SPARSE', 'SAM', 'ASAM', 'FAD', 'DISAM']`. Methods admitted to Pi_full: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'SPECTRAL_NORM_REG', 'SPECTRAL_REG_2024', 'SVB_ORTHDNN', 'STABLE_RANK_NORM', 'SAM', 'ASAM']`.

## Common-budget protocol

All runnable methods use the same CMNIST data/model semantics, seeds, outer horizon, source batch schedule, and target-blind policy. SAM/ASAM and projection methods record additional intrinsic compute in `results/compute_budget.csv`; their outer step count is not reduced.

## Source-only calibration

The spectral/flatness families were calibrated with 30 actual seed-10 training runs, not a one-row placeholder. Canonical variants were frozen using source accuracy >=55% and the preregistered method-specific source geometry score, with source loss as tie-breaker. Only after `selected_source_only_variants.json` and the pre-target sweep table were written were target metrics evaluated for every retained variant.

- `SPECTRAL_NORM_REG`: 5 variants; post-hoc target range 10.2% to 10.6%; source-only canonical `SPECTRAL_NORM_REG[lambda=1]`
- `SPECTRAL_REG_2024`: 5 variants; post-hoc target range 10.2% to 10.6%; source-only canonical `SPECTRAL_REG_2024[lambda=1]`
- `SVB_ORTHDNN`: 5 variants; post-hoc target range 10.2% to 11.0%; source-only canonical `SVB_ORTHDNN[factor=0.05,frequency=1]`
- `STABLE_RANK_NORM`: 5 variants; post-hoc target range 10.2% to 10.2%; source-only canonical `STABLE_RANK_NORM[target_rank=8]`
- `SAM`: 5 variants; post-hoc target range 10.2% to 10.6%; source-only canonical `SAM[rho=0.2]`
- `ASAM`: 5 variants; post-hoc target range 10.3% to 10.7%; source-only canonical `ASAM[rho=2]`

Across this declared grid, every spectral/flatness variant remained a color-shortcut solution: target accuracy stayed near chance while target prediction/color agreement stayed near 100%. This supports a negative result for these tested variants under this harness. It does not justify the broader claim that the complete SNR, SR2024, SVB, SRN, SAM, or ASAM method families cannot work under other source-only configurations.

## Fidelity gates

F0 checkpoint and shared initialization/schedule reconstruction: PASS for ERM/IRM reference hashes; all configured methods finite unless listed in errors.
F1 V-REx objective and anneal/reset: PASS with squared source-risk gap, lambda=10000, anneal=100, Adam reset, and post-anneal whole-loss rescale.
F2 CORAL representation penalty and `n-1` covariance: PASS.
F3 Fishr classifier-gradient variance and first-order MLDG remain frozen from the prior panel.
F4 Spectral methods: SNR, SR2024, SVB, and SRN are implemented as common-harness variants; SVD-SPARSE is deferred rather than replaced with a nuclear-norm substitute.
F5 Flatness methods: SAM and ASAM are implemented as two-step source-only methods; FAD and DISAM are deferred rather than approximated.
F4 method completeness: PASS, 12 methods x 5 seeds.
F5 base R5 world identity and valid mixture weights: PASS.
F6 primary basis and displacement semantics: PASS.
F7 source-only O, `O e3 = O e5 = 0`, rank limit: PASS for all model rows.
F8 target/evaluation leakage: PASS by construction and provenance flags.
F9 finite continuation replay: PASS for 1980 response rows across 12 admitted methods.

## Diagnostics

- Weight spectra: `results/weight_spectrum_long.csv` records singular values, spectral norm, Frobenius norm, nuclear norm, stable rank, effective rank, spectral mass, numerical ranks, and condition diagnostics.
- Representation spectra: `results/representation_spectrum.csv` records fixed-source-bank encoder spectra.
- Gradient spectra: `results/gradient_spectrum.csv` records classifier-gradient spectra on the fixed source bank.
- Flatness: `results/flatness_diagnostics.csv` records source loss, gradient norm, HVP power-iteration top eigenvalue, Hutchinson trace, SAM-style sharpness deltas, and random-direction sharpness.

## Performance panel

Five-seed means:

- `ERM`: source 85.0%, target 11.0%, color agreement 99.1%
- `IRMv1`: source 60.0%, target 66.9%, color agreement 34.9%
- `VREX`: source 63.1%, target 55.3%, color agreement 47.5%
- `CORAL`: source 85.0%, target 10.9%, color agreement 99.2%
- `FISHR`: source 69.0%, target 55.9%, color agreement 48.4%
- `MLDG`: source 85.0%, target 11.2%, color agreement 98.9%
- `SPECTRAL_NORM_REG`: source 84.6%, target 11.0%, color agreement 99.0%
- `SPECTRAL_REG_2024`: source 85.0%, target 10.2%, color agreement 100.0%
- `SVB_ORTHDNN`: source 85.0%, target 10.2%, color agreement 100.0%
- `STABLE_RANK_NORM`: source 85.0%, target 10.2%, color agreement 100.0%
- `SAM`: source 85.0%, target 10.2%, color agreement 100.0%
- `ASAM`: source 85.0%, target 10.3%, color agreement 99.9%

Best target mean: `IRMv1` (0.669). Worst target mean: `SPECTRAL_REG_2024` (0.102). These target metrics are post-hoc only and were not used for method admission, variant selection, normalization, or grouping.

## Spectral geometry

Five-seed final encoder-weight means:

- `ERM`: final encoder spectral norm 1.701, delta -0.175, stable rank 5.309, encoder effective rank 41.987
- `IRMv1`: final encoder spectral norm 1.553, delta -0.323, stable rank 13.847, encoder effective rank 54.831
- `VREX`: final encoder spectral norm 1.418, delta -0.458, stable rank 17.135, encoder effective rank 55.207
- `CORAL`: final encoder spectral norm 1.700, delta -0.176, stable rank 5.272, encoder effective rank 41.949
- `FISHR`: final encoder spectral norm 1.383, delta -0.494, stable rank 16.465, encoder effective rank 54.830
- `MLDG`: final encoder spectral norm 1.951, delta 0.074, stable rank 7.648, encoder effective rank 47.833
- `SPECTRAL_NORM_REG`: final encoder spectral norm 0.509, delta -1.367, stable rank 51.083, encoder effective rank 61.097
- `SPECTRAL_REG_2024`: final encoder spectral norm 1.019, delta -0.858, stable rank 48.040, encoder effective rank 59.404
- `SVB_ORTHDNN`: final encoder spectral norm 1.050, delta -0.826, stable rank 53.623, encoder effective rank 63.978
- `STABLE_RANK_NORM`: final encoder spectral norm 1.000, delta -0.876, stable rank 1.296, encoder effective rank 10.628
- `SAM`: final encoder spectral norm 1.544, delta -0.332, stable rank 5.163, encoder effective rank 44.486
- `ASAM`: final encoder spectral norm 1.611, delta -0.265, stable rank 5.710, encoder effective rank 47.104

Largest top-singular-value reduction: `SPECTRAL_NORM_REG` (-1.367). Highest final encoder tail/effective rank: `SVB_ORTHDNN` (63.978). Lowest final encoder tail/effective rank: `STABLE_RANK_NORM` (10.628). Highest final encoder stable rank: `SVB_ORTHDNN` (53.623). Highest final representation effective rank: `FISHR` (39.476). Highest final classifier-gradient effective rank: `SVB_ORTHDNN` (25.165).

## Flatness geometry

Five-seed source-bank means:

- `ERM`: source loss 0.368, Hessian top eig 3.756, trace 17.250, sharpness@0.05 0.019
- `IRMv1`: source loss 0.656, Hessian top eig 4.223, trace 14.918, sharpness@0.05 0.051
- `VREX`: source loss 0.681, Hessian top eig 1.351, trace 1.418, sharpness@0.05 0.025
- `CORAL`: source loss 0.369, Hessian top eig 3.620, trace 16.719, sharpness@0.05 0.018
- `FISHR`: source loss 0.675, Hessian top eig 0.267, trace 3.228, sharpness@0.05 0.014
- `MLDG`: source loss 0.365, Hessian top eig 3.709, trace 18.892, sharpness@0.05 0.019
- `SPECTRAL_NORM_REG`: source loss 0.670, Hessian top eig 0.651, trace 0.030, sharpness@0.05 0.010
- `SPECTRAL_REG_2024`: source loss 0.424, Hessian top eig 1.262, trace 3.580, sharpness@0.05 0.010
- `SVB_ORTHDNN`: source loss 0.395, Hessian top eig 3.497, trace 8.026, sharpness@0.05 0.015
- `STABLE_RANK_NORM`: source loss 0.384, Hessian top eig 3.684, trace 14.080, sharpness@0.05 0.016
- `SAM`: source loss 0.383, Hessian top eig 0.916, trace 3.430, sharpness@0.05 0.008
- `ASAM`: source loss 0.378, Hessian top eig 1.231, trace 6.081, sharpness@0.05 0.011

Lowest Hessian top eigenvalue: `FISHR` (0.267). Lowest Hessian trace estimate: `SPECTRAL_NORM_REG` (0.030). Lowest SAM-style sharpness at rho=0.05: `SAM` (0.008). Low source loss is retained mainly by ERM-like methods, while the lowest flatness metrics occur in methods that do not necessarily have the best target accuracy.

## A/O/Pi response

Mean K=20 source-exposed source-bank response norms:

- `ERM`: mean K=20 source-exposed source-bank response 94.791
- `IRMv1`: mean K=20 source-exposed source-bank response 15.028
- `VREX`: mean K=20 source-exposed source-bank response 3.801
- `CORAL`: mean K=20 source-exposed source-bank response 94.027
- `FISHR`: mean K=20 source-exposed source-bank response 8.731
- `MLDG`: mean K=20 source-exposed source-bank response 94.855
- `SPECTRAL_NORM_REG`: mean K=20 source-exposed source-bank response 0.308
- `SPECTRAL_REG_2024`: mean K=20 source-exposed source-bank response 13.040
- `SVB_ORTHDNN`: mean K=20 source-exposed source-bank response 52.670
- `STABLE_RANK_NORM`: mean K=20 source-exposed source-bank response 66.997
- `SAM`: mean K=20 source-exposed source-bank response 81.294
- `ASAM`: mean K=20 source-exposed source-bank response 102.653

Grouping status: `STAGE4-PASS`, silhouette `0.424`, bootstrap ARI `1.000`. This is a blind numeric grouping over opaque direction IDs, not a semantic-discovery claim.

## Required answers

1. Existing ERM/IRMv1/VREX/CORAL/FISHR/MLDG implementations were not edited in their algorithm files for this task; the common runner admits them through the modular registry and reference hashes cover ERM/IRMv1.
2. WEIGHT_NUCLEAR and FEATURE_NUCLEAR are legacy-only/default-disabled; their historical code and artifacts are preserved.
3. New runnable methods are definition-faithful common-harness variants for SNR, SR2024, SVB, SRN, SAM, and ASAM; SVD-SPARSE, FAD, and DISAM are deferred rather than approximated.
4. All runnable methods use 501 outer steps, identical CMNIST model/data/seed/schedule semantics, and source-only training. Extra SAM/ASAM/projection compute is recorded instead of hidden.
5. Target information is excluded from tuning and canonical selection; target is post-hoc performance and evaluation functional measurement only.
6. Spectral norm changes most under `SPECTRAL_NORM_REG` (-1.367); tail/effective rank is highest under `SVB_ORTHDNN` (63.978) and lowest under `STABLE_RANK_NORM` (10.628).
7. Flatness changes most by Hessian eigenvalue under `FISHR` (0.267), trace under `SPECTRAL_NORM_REG` (0.030), and sharpness proxy under `SAM` (0.008).
8. Target accuracy is highest for `IRMv1` (0.669); most spectral/flatness additions remain ERM-like on target under this harness.
9. SVB_ORTHDNN produces a large spectral-geometry change without being the flattest method.
10. SAM/ASAM reduce sharpness metrics relative to ERM while leaving spectra and target behavior close to ERM-like failures.
11. Lower rank is not sufficient: STABLE_RANK_NORM has the lowest encoder effective-rank profile but remains target-poor.
12. Lower sharpness is not sufficient: SAM and SR2024 reduce sharpness/eigenvalue metrics but remain target-poor.
13. Gradient effective rank separates some successful methods from ERM-like failures, but it is not a sufficient scalar because FISHR/VREX/IRM differ in target and response structure.
14. Pi_full continuation is valid for the 12 admitted runnable methods and deferred for SVD-SPARSE/FAD/DISAM.
15. Successful algorithms do not collapse to one universal Pi fingerprint; response norms and blind clusters remain heterogeneous.
16. Strongest counterexample to a simple spectral explanation: STABLE_RANK_NORM compresses spectrum heavily but stays OOD-poor.
17. Strongest counterexample to a simple flat-minima explanation: SAM/ASAM improve sharpness diagnostics but stay OOD-poor.
18. What remains unestablished: causality, semantic recovery, source identifiability, theory validation, paper-benchmark reproduction, and whether any scalar spectral/flatness diagnostic is necessary or sufficient.

## Counterexamples

- Low source loss but OOD-poor: ERM/CORAL/MLDG retain about 85% source accuracy and about 11% target accuracy.
- Flat but OOD-poor: SAM/SR2024 lower source sharpness metrics but remain near ERM target behavior.
- Low-rank but OOD-poor: STABLE_RANK_NORM yields the lowest encoder effective rank and about 10% target accuracy.
- High-rank but OOD-good: IRMv1 and FISHR retain higher representation/gradient effective-rank profiles while reaching substantially higher target accuracy than ERM.
- Similar target with different response: several ERM-like spectral/flatness methods cluster around 10%-11% target accuracy but have different K=20 Pi response norms.
- Similar response with different geometry: ERM and MLDG have close K=20 source-bank response norms while their encoder spectral norms differ.

## Interpretation

The run produces descriptive response, spectral, and flatness profiles. This task deliberately caps scientific interpretation at `SPECTRAL-FLATNESS-PANEL-PARTIAL`; response, spectrum, and flatness differences are not promoted to a PASS, causal claim, or algorithm claim. Raw parameter-space sharpness is not invariant under arbitrary reparameterization, but it is a controlled diagnostic here because architecture, parameterization, initialization, optimizer family, and source schedule are fixed. Errors: `[]`.
