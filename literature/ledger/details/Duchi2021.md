# Duchi2021 / Learning Models with Uniform Performance via Distributionally Robust Optimization

## Problem
Learn a model with uniformly good performance across a specified family of reweightings/distributions.

## Target
Worst-case or uniformly controlled population risk.

## Representation
An f-divergence/entropy uncertainty set and its simplex reweighting dual.

## Proof bottleneck
Uniform performance is a supremum over distributions or groups rather than an ordinary expected risk.

## Proof-enabling property
Convex duality reduces the supremum to a regularized weighted-risk objective; concentration controls the empirical robust objective.

## Main theorem
Theorems 1-2 characterize the robust population objective and finite-sample uniform guarantees for the selected divergence family.

## Price of tractability
The divergence, radius and support define the target family; misspecification is not repaired by optimization.

## Algorithm relationship / source-only status
EXACT robust objective; `YES` conditional on the uncertainty set and source sampling.

## Evidence
Theorems 1-2; Annals of Statistics 49(3), 2021.
