# Task 3R source-only algorithmization

Verdict: `TASK3R-ALGORITHM-PARTIAL`

## A. Prior-work design bridge

The reusable pattern is to create virtual train/test domain splits from source domains, derive a local object, and then optimize a computable surrogate. MLDG supplies the pseudo-domain protocol; Fish/Fishr, Hessian Alignment, and CMA show how gradient/Hessian objects can be approximated or reduced; Transferability keeps the distinction between a local quantity and held-out behavior. The selected quadratic reduction is labeled `SPECIAL-CASE` and carries no novelty claim.

## B. Controlled theorem object

The frozen chain is `A_rec, O_S, Pi -> E -> spectral slack / affine regret`. This probe replaces unavailable target `A_rec` during training with a finite source-domain LOO response target. `Pi` remains `-(H_R+lambda K)^-1(B_R+lambda C)` and is never optimized as a free matrix.

## C. Source estimability

Source moments, head optima, Hessians, task-state contrasts, pseudo-target risk, and finite LOO response are source-only. Oracle `A_rec`, true held-out risk, and slack diagnostics are post-hoc only. The machine rows record both target-use flags as false.

## D. Selected objective

The prototype learns a low-rank forcing block `C=UV^T` in a centered quadratic regularizer. It minimizes source pseudo-target risk plus a directional response mismatch over LOO source folds. Centering makes `z0=0` at every meta-source reference. `K=I`; exact FOC differentiation yields the actual response.

## E. Relation to prior methods

The finite source split is MLDG-like, but the optimized object is a constrained exact head response rather than a one-step model gradient. Under quadratic assumptions it overlaps derivative/moment alignment and is therefore a special case, not established as a distinct general algorithm. It does not inherit target-risk bounds from Transferability or Hessian Alignment.

## F. Exact Gaussian result

All optimizer rows were stable: `True`. Exact IFT versus retraining passed: `True`, maximum relative error `2.482e-11`. However, the preregistered Gaussian gate passed: `False`.

- `gaussian_primary_1`: response reduction `-1`, held-out risk improvement `1.456e-07`, H1/H2/H3 = `False/False/True`.
- `gaussian_primary_2`: response reduction `-1`, held-out risk improvement `1.829e-07`, H1/H2/H3 = `False/False/True`.
- `gaussian_reduced_exposure`: response reduction `-1`, held-out risk improvement `-1.618e-06`, H1/H2/H3 = `False/False/False`.

Aggregate method diagnostics:

- `ERM`: target risk `0.2908719`, estimated residual `0`, oracle residual `0.0001245835`.
- `MLDG_LOCAL`: target risk `0.29087249`, estimated residual `2.2528898e-05`, oracle residual `0.00011447813`.
- `NO_RESPONSE`: target risk `0.29087233`, estimated residual `0.00062609691`, oracle residual `0.00067072114`.
- `PSEUDO_RESPONSE`: target risk `0.29087233`, estimated residual `0.0006639062`, oracle residual `0.00070695734`.
- `RANDOM_RESPONSE`: target risk `0.2910727`, estimated residual `0.015026653`, oracle residual `0.01507856`.

## G. Mechanism intervention

The matched-norm random-response control was trained with the same source-only protocol. It did not validate the proposed causal mechanism: H3 may pass locally because the random control is harmful, but H1 and H2 did not pass in the required independent configurations. Reduced source exposure also failed the preregistered H4 gate: `True`.

## H. Approximation fidelity

`not_applicable_exact_quadratic`. This Phase-I implementation uses exact population moments and exact quadratic derivatives; no cheap approximation was introduced.

## I. Failure decomposition

- Theory/objective: the frozen theorem remains an affine geometry result; this source pseudo-target surrogate is not implied to improve arbitrary finite target risk.
- A-estimation: the finite LOO response did not reliably reduce the post-hoc residual.
- Learner realizability: the low-rank `C=UV^T` class was optimized stably but did not reach the desired improvement.
- Optimization: no numerical failure was observed.
- Approximation: not applicable in the exact probe.

The bottleneck is: Exact source-only mechanism is valid, but the preregistered held-out Gaussian gate did not pass. CMNIST was not run because its preregistered gate did not pass.

## J. Scope

This is deterministic population Gaussian evidence. It is not a target-risk theorem, causal result, finite-sample guarantee, semantic recovery claim, or universal DG result.

Historical reopen: none.
