# Task 3 CMNIST Local Response Report

## A. Question

Does curvature-aware local response geometry add algorithmic value on end-to-end CMNIST?

## B. Cleanup

The discarded Gaussian Task3R implementation/results were removed from live source and result paths. `round3_redesign/task3_applicability/` was retained as historical diagnostic only.

## C. Prior Art

The exact object audit is in `prior_art_exact_object.md`. The run uses neutral internal names and makes no algorithmic novelty claim.

## D. Benchmark

The run uses the existing CMNIST generator, binary digit label, source correlations, target correlation, `SmallCMNISTCNN`, Adam optimizer family, and counterfactual color probe. Profile: `main`.

## E. Methods

- `ERM`: mean target acc `0.1136`, mean worst-source acc `0.8070`, mean color response `0.515479`
- `IRMv1`: mean target acc `0.1136`, mean worst-source acc `0.8070`, mean color response `0.515499`
- `LOCAL_RESPONSE`: mean target acc `0.1233`, mean worst-source acc `0.8078`, mean color response `0.372375`
- `RANDOM_METRIC`: mean target acc `0.1196`, mean worst-source acc `0.8065`, mean color response `0.206458`
- `SHUFFLED_LOCAL_RESPONSE`: mean target acc `0.1251`, mean worst-source acc `0.8045`, mean color response `0.440863`
- `UNPRECONDITIONED_GRAD_ALIGN`: mean target acc `0.1136`, mean worst-source acc `0.8070`, mean color response `0.515399`
- `V-REx`: mean target acc `0.1136`, mean worst-source acc `0.8070`, mean color response `0.515470`

## F. Source-Only Selection

Beta and checkpoint are selected by worst-source validation accuracy with mean-source validation as tie-breaker. `target_leakage_detected=false`.

## G. Main OOD Results

- `LOCAL_RESPONSE` vs `ERM`: mean diff `0.0097`, wins `2/10`, 95% CI `[0.0000, 0.0283]`
- `LOCAL_RESPONSE` vs `UNPRECONDITIONED_GRAD_ALIGN`: mean diff `0.0097`, wins `2/10`, 95% CI `[0.0000, 0.0283]`
- `LOCAL_RESPONSE` vs `RANDOM_METRIC`: mean diff `0.0037`, wins `2/10`, 95% CI `[-0.0146, 0.0267]`
- `LOCAL_RESPONSE` vs `IRMv1`: mean diff `0.0097`, wins `2/10`, 95% CI `[0.0000, 0.0283]`
- `LOCAL_RESPONSE` vs `V-REx`: mean diff `0.0097`, wins `2/10`, 95% CI `[0.0000, 0.0283]`
- `UNPRECONDITIONED_GRAD_ALIGN` vs `ERM`: mean diff `0.0000`, wins `0/10`, 95% CI `[0.0000, 0.0000]`

## H. Curvature Increment

The primary increment is `LOCAL_RESPONSE - UNPRECONDITIONED_GRAD_ALIGN`; see `paired_comparisons.csv` and `run_table.csv` for paired seed rows.

## I. Mechanism

Mechanism diagnostics include raw gradient disagreement, local-response disagreement, response-vector norm, damped curvature spectrum, update norms, and counterfactual color response.

## J. Controls

Random metric control rows are in `random_metric_control.csv`; shuffled environment-control rows are in `environment_shuffle_control.csv`.

## K. Counterexamples

Strongest detected counterexamples are recorded in `counterexamples.csv`; count `9`.

## L. Relation to Frozen Theory

This experiment tests the lower-level source-risk metric insight. It does not estimate `A_rec`, optimize `E`, validate spectral slack for neural networks, or claim target-risk lower bounds.

## M. Verdict

`TASK3-CMNIST-FAIL`

Criteria:

- `no_target_leakage`: `true`
- `end_to_end_representation_training`: `true`
- `all_10_primary_seeds_completed`: `true`
- `mean_ood_vs_erm_ge_2pp`: `false`
- `mean_ood_vs_grad_ge_1pp`: `false`
- `seed_wins_vs_erm_ge_7`: `false`
- `seed_wins_vs_grad_ge_7`: `false`
- `worst_source_degradation_within_1pp`: `true`
- `real_metric_beats_random_metric`: `false`
- `color_or_mechanism_predicted_direction`: `true`
- `not_explained_by_update_norm`: `true`

Historical reopen: none.
