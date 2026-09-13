# Arjovsky2019 / Invariant Risk Minimization

## Problem
Find a representation for which the same classifier is optimal across training environments and therefore transfers to a suitable unseen environment.

## Target
The ideal population invariant-risk objective and its induced predictor risk, not a generic finite-sample deep-training guarantee.

## Representation
Shared optimal classifier constraint `w ∈ argmin_{w'} R_e(w'∘Φ)` for every environment; IRMv1 uses a classifier-variable gradient penalty as a surrogate.

## Proof bottleneck
Environment-wise empirical objectives do not identify which features remain predictive under shifts. The shared-optimum constraint encodes a candidate stable conditional mechanism.

## Proof-enabling property
Equality of population optima across environments gives an algebraic invariance condition; restricted structural models can then identify invariant features.

## Main theorem
The paper's formal principle is Eq. (2), with the practical penalty in Eq. (3). Broad guarantees require additional assumptions and are supplied by later restricted analyses.

## Price of tractability
Realizability, sufficiently heterogeneous environments and population optimization; IRMv1's gradient is with respect to `w`, not all network parameters.

## Algorithm relationship
POPULATION-ABSTRACTION / SURROGATE for IRMv1.

## Scope
Naturally covered: ideal IRM and restricted linear/structural models. Excluded: exact arbitrary deep optimizer dynamics.

## Source-only status
YES only under the stated invariant-family assumptions; no assumption-free unseen-domain guarantee.

## Evidence
Eq. (2)-(3); theoretical sections and appendices.
