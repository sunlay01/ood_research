# Esfahani2018 / Data-Driven Distributionally Robust Optimization Using the Wasserstein Metric

## Problem
Optimize against distributions near the empirical source law under Wasserstein distance.

## Target
Worst-case population risk `sup_{Q:W(Q,P)≤rho} E_Q[ell_f]` and its data-driven estimator.

## Representation
An uncertainty set (Wasserstein ball) around the empirical/source distribution.

## Proof bottleneck
The supremum over distributions is infinite-dimensional and cannot be optimized directly.

## Proof-enabling property
Optimal-transport duality converts the robust supremum into a finite-dimensional regularized expression; concentration controls empirical-to-population error.

## Main theorem
Theorem 4.2 characterizes the worst-case expectation/robust optimization problem through a dual formulation and finite-sample guarantees under stated transport assumptions.

## Price of tractability
The metric, radius, support and loss regularity define the shift model. Validity outside the Wasserstein ball is not implied.

## Algorithm relationship
EXACT for the robust population objective and its dual under assumptions; a neural implementation may still be an approximate solver.

## Scope
Naturally covered: Wasserstein DRO and adversarial-training variants. Excluded: methods with no defensible uncertainty set or conditional mechanism.

## Source-only status
YES conditional on the selected ball and source law; radius selection is assumption-controlled.

## Evidence
Theorem 4.2; Sec. 3 transport duality and robustness results.
