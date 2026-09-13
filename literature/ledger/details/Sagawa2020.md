# Sagawa2020 / Distributionally Robust Neural Networks for Group Shifts

## Problem
Improve worst-group performance when groups are observed in training and may be imbalanced.

## Target
Maximum risk over a finite set of observed groups.

## Representation
Risk vector `(R_1(f),...,R_E(f))` and simplex weights; GroupDRO optimizes the maximum/group-weighted risk.

## Proof bottleneck
Average source risk can hide a high-risk minority group.

## Proof-enabling property
Finite-group uniform convergence and convex reweighting give a direct empirical-to-population robust-risk guarantee.

## Main theorem
Section 3-4 bounds population group risk from empirical group risks with complexity and group-coverage terms.

## Price of tractability
Finite, observed groups and enough samples per group; no arbitrary unseen-domain extrapolation.

## Algorithm relationship / source-only status
EXACT for finite-group robust learning; `YES` for observed groups only.

## Evidence
Sections 3-4; ICLR/OpenReview PDF.
