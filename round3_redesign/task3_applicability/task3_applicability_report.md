# Task 3 Applicability Report

## A. Question

Do the frozen Round-3 mechanism, information, and spectral quantities add non-tautological grouped out-of-sample discrimination of finite held-out DG behavior beyond method/lambda and conventional source-side baselines?

## B. Preregistration

The preregistration was written to `task3_preregistered_design.json` before feature materialization and before held-out target outcome generation. It freezes methods, lambdas, families, radius grids, feature sets B0/B1/B2/B3, outcomes, CV grouping, null controls, and verdict thresholds.

## C. Leakage / Tautology Audit

- Logic audit pass: `True`.
- Feature rows: `238`.
- Held-out target outcome rows: `5208`.
- Valid joined rows: `4260`.
- Grouped CV keeps training runs together: `True`.
- Target outcomes used for feature construction or selection: `False`.

## D. Gaussian Results

Gaussian rows use legal population environment-family target perturbations. Family/source-induced rows are evaluated as finite held-out target risks, not universal robust risks.

## E. CMNIST Results

CMNIST rows use frozen ERM representations and squared-loss linear heads. Accuracy is an offline post-hoc target outcome and is not used for training, lambda choice, feature construction, or threshold selection.

## F. B0/B1/B2/B3 Comparison

| benchmark | test | metric | B1 | B2 | B3 | B3-B1 |
|---|---|---|---:|---:|---:|---:|
| cmnist | finite_risk_prediction | cv_r2 | 0.9107520194934643 | 0.6338505449039193 | 0.7630624715578259 | -0.14768954793563838 |
| cmnist | ranking | kendall_tau | 0.5377777777777777 | 0.4577777777777778 | 0.4997979797979798 | -0.03797979797979789 |
| cmnist | win_loss_discrimination | auc | 1.0 | 1.0 | 1.0 | 0.0 |
| gaussian | finite_risk_prediction | cv_r2 | 0.07530757260744159 | -0.8029057584806882 | -3808584160690041.0 | -3808584160690041.0 |
| gaussian | ranking | kendall_tau | -0.03888888888888889 | 0.008333333333333312 | 0.000925925925925918 | 0.03981481481481481 |
| gaussian | win_loss_discrimination | auc | 0.5915441176470588 | 0.5915441176470588 | 0.5228860294117647 | -0.06865808823529418 |

## G. Null Controls

| benchmark | test | metric | observed B3 | permuted B3 | random B3 |
|---|---|---|---:|---:|---:|
| cmnist | ranking | kendall_tau | 0.4997979797979798 | 0.41898989898989897 | 0.40686868686868694 |
| gaussian | ranking | kendall_tau | 0.000925925925925918 | 0.000925925925925918 | 0.12037037037037035 |
| cmnist | finite_risk_prediction | cv_r2 | 0.7630624715578259 | 0.6265369202565758 | 0.6516259733197882 |
| cmnist | win_loss_discrimination | auc | 1.0 | 1.0 | 1.0 |
| gaussian | finite_risk_prediction | cv_r2 | -3808584160690041.0 | -3808584160690041.0 | -0.35993732915868604 |
| gaussian | win_loss_discrimination | auc | 0.5228860294117647 | 0.5228860294117647 | 0.5454963235294118 |

## H. Counterexamples

Counterexample rows saved: `3`. The table is deliberately retained even when it weakens the applicability verdict.

## I. Per-Method Interpretation

L2, IRMv1, and V-REx are compared through actual-solution quantities only. Common-base attribution remains secondary and is not counted as the actual method mechanism.

| benchmark | method | range ||z0|| | range ||E||op | finite range rho_slack | mean delta risk | win rate |
|---|---|---:|---:|---:|---:|---:|
| cmnist | L2 | 0..1.33301 | 9.01244..18.6286 | NA | 0.888673 | 0 |
| cmnist | IRMV1 | 0.000262585..0.0373908 | 9.01072..17.3876 | NA | -0.0222195 | 0.917 |
| cmnist | VREX | 1.95571e-05..0.00708148 | 8.99571..17.3885 | NA | -0.00417074 | 0.917 |
| gaussian | L2 | 1.96262e-17..0.0281651 | 1.3973e-12..0.105744 | 0..0.00608824 | 7.43267e-05 | 0.163 |
| gaussian | IRMV1 | 1.96262e-17..0.000297055 | 1.3973e-12..0.00131241 | 1.23169e-10..8.35429e-07 | -3.95951e-07 | 0.262 |
| gaussian | VREX | 1.96262e-17..5.88999e-05 | 1.3973e-12..0.000141511 | 9.2746e-13..9.2709e-09 | -1.87124e-08 | 0.179 |

Mechanism and family-aware quantities vary with lambda, but the grouped B3-vs-B1 rows above show that this variation does not become stable SUPPORT-level finite-target discrimination. Seed holdout is enforced for CMNIST; Gaussian uses family/config holdout and is reported with the sample-size limitation. Method/lambda conditioning is represented by B0/B1 and the null controls; apparent common-base explanations are not promoted to actual mechanisms.

## J. Radius Dependence

`radius_sweep.csv` reports the same B3-B1 comparisons separately by preregistered radius regime. Radius was not tuned after observing held-out performance.

## K. Limitations

Task 3 tests finite held-out behavior in audited Gaussian and CMNIST frozen-head settings. It does not establish a source-only deployable selector, target-risk lower bound, finite-sample guarantee, causal identification, semantic recovery, global nonlinear robustness, or universal DG theorem.

## L. Verdict

`TASK3-APPLICABILITY-PARTIAL`

Historical reopen: none
