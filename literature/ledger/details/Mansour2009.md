# Mansour2009 / Domain Adaptation: Learning Bounds and Algorithms

## Problem
Bound target error and design discrepancy-based learners when source and target distributions differ.

## Target
Target risk for a loss class/hypothesis class.

## Representation
A loss-class discrepancy (including `HΔH`-style special cases) plus an ideal joint hypothesis error.

## Proof bottleneck
Risk differences depend on all hypotheses in the class, not one empirical classifier.

## Proof-enabling property
Discrepancy triangle inequalities and Rademacher/uniform convergence control the supremum over the loss class.

## Main theorem
Theorems 1-2 give source-risk plus discrepancy plus joint-error style bounds and learning algorithms for estimating the discrepancy.

## Price of tractability
Target samples, bounded loss/class complexity and an oracle joint-error term.

## Algorithm relationship
EXACT for the discrepancy objective; not a source-only DG theorem.

## Scope / source-only status
DA discrepancy methods; `NO` for source-only target guarantees.

## Evidence
Theorems 1-2; arXiv:0902.3430.
