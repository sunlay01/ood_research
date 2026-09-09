# Task 3 CMNIST Local Response Report

## A. Protocol Repair

The interpreted primary comparison is now on the official IRMv1 Colored MNIST reversed-color protocol: 25% label noise, train color-flip probabilities 0.2/0.1, target color-flip probability 0.9, 2-channel 14x14 MLP, BCE-with-logits, L2 weight penalty 0.001, penalty annealing, and whole-loss rescaling after anneal.

## B. Baseline Recovery Gate

`passed=true`; ERM target accuracy `0.1710`; IRMv1 target accuracy `0.6686`; IRMv1 advantage `0.4976`.

## C. Official-Protocol Pilot

- `ERM`: selected beta `0`, target acc `0.1710`, train acc `0.8754`, pred/color agreement `0.9299`
- `IRMV1`: selected beta `1e+04`, target acc `0.6686`, train acc `0.6818`, pred/color agreement `0.3653`
- `HEAD_GRADIENT_VARIANCE_SURROGATE`: selected beta `0.01`, target acc `0.1787`, train acc `0.8767`, pred/color agreement `0.9216`

## D. Paired Comparisons

- `IRMV1` vs `ERM`: mean target-acc diff `0.4976`, wins `1/1`

## E. Verdict

`TASK3-CMNIST-PARTIAL`

Criteria:

- `baseline_recovery_passed`: `true`
- `official_reversed_color_protocol`: `true`
- `all_10_primary_seeds_completed`: `false`
- `mean_ood_vs_erm_ge_2pp`: `false`
- `mean_ood_vs_grad_ge_1pp`: `false`
- `seed_wins_vs_erm_ge_7`: `false`
- `seed_wins_vs_grad_ge_7`: `false`
- `real_metric_beats_random_metric`: `false`

This is an official-protocol pilot, not a paper-level success claim. Historical reopen: none.
