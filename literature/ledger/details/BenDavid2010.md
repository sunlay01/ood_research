# BenDavid2010 / A theory of learning from different domains

## Problem
Transfer a classifier from a labeled source distribution to an unlabeled target distribution.

## Target
Target risk `R_T(h)` under 0-1 loss.

## Representation
The source risk plus the `HΔH` discrepancy between source and target, with a joint-error term.

## Proof bottleneck
Source labels do not identify target behavior. Add and subtract the risk of a comparison hypothesis, then control disagreement between domains.

## Proof-enabling property
Triangle inequality for risks and uniform convergence of the symmetric-difference hypothesis class.

## Main theorem
Theorem 2 gives the standard source-risk + one-half `d_HΔH` + `lambda*` upper bound (with empirical complexity terms in the finite-sample form).

## Price of tractability
The discrepancy needs target samples and `lambda*` is an oracle joint-error term; this is DA, not source-only DG.

## Algorithm relationship
EXACT for the discrepancy theorem; it does not assert that a particular representation-learning optimizer attains the bound.

## Scope
Naturally covered: hypothesis-class alignment and domain classifiers. Excluded: arbitrary unseen-domain extrapolation without a target/family model.

## Source-only status
NO: target discrepancy and joint error are target-dependent/oracle.

## Evidence
Theorem 2; definitions of `d_HΔH` and `lambda*`.
