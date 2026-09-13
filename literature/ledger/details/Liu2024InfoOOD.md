# Liu2024InfoOOD / An Information-Theoretic Framework for OOD Generalization

## Problem
Relate source-trained stochastic learner information to a risk gap under distribution shift.

## Target
Source-to-target OOD risk gap / target risk bound.

## Representation
Mutual information or information-density quantities connecting learner output and shifted data.

## Proof bottleneck
The target distribution is unavailable; a change-of-measure decomposition must separate shift from learner complexity.

## Proof-enabling property
Information-theoretic variational inequalities and concentration convert dependence between data and learner output into a generalization term.

## Main theorem
The main bound decomposes OOD error into source/generalization, shift-dependent and information-radius terms under the paper's stated stochastic assumptions.

## Price of tractability
Information quantities and shift terms may be oracle or hard to estimate; the theorem is not an algorithm taxonomy.

## Algorithm relationship
POPULATION-ABSTRACTION for stochastic learners.

## Scope
Naturally covered: randomized/stochastic OOD learners and the SGLD extension. Excluded: exact deterministic IRMv1, Fishr or latent direct-sum semantics.

## Source-only status
Conditional: source-only if the information/shift model is assumed, but key radius terms may not be operational.

## Evidence
ArXiv:2403.19895v1; main theorem and definitions.
