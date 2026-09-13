# Blanchard2021 / Domain Generalization by Marginal Transfer Learning

## Problem
Control performance on a new domain when only labeled source domains are available.

## Target
Risk on a domain drawn from, or lying in, an admissible meta-distribution/family.

## Representation
Marginal transfer: a domain-level statistic/function is estimated from source environments and transferred through a meta-distribution.

## Proof bottleneck
There is no observed target distribution. The theorem must replace target discrepancy by concentration of a domain-generating law and an explicit admissible family.

## Proof-enabling property
Domain-level concentration and a structured marginal-transfer functional make an unseen-domain quantity estimable from multiple source domains.

## Main theorem
The main JMLR result bounds new/admissible-domain performance using source empirical risk, environment complexity, and a domain-family deviation term (see Sec. 3 and theorem statements).

## Price of tractability
Coverage/representativeness of source domains for the meta-distribution is essential; arbitrary target shifts are not covered.

## Algorithm relationship
POPULATION-ABSTRACTION: the theorem is about induced predictors and domain-family statistics, not exact optimizer trajectories.

## Scope
Naturally covered: domain-of-domains and marginal-transfer DG. Awkward: gradient covariance, causal mechanisms, or target families not expressible by the marginal law.

## Source-only status
YES conditional on the meta-distribution/family assumption; the family term is assumption-controlled rather than directly observed.

## Evidence
JMLR v22 article; Sec. 3 and main generalization theorem.
