# Kamath2021 / Does Invariant Risk Minimization Capture Invariance?

## Problem
Test whether IRM's invariant-optimality criterion actually captures the intended invariance and when it identifies a robust predictor.

## Target
Identifiability/consistency of invariant predictors in a restricted structural setting.

## Representation
Environment-wise risk gradients/optimality conditions and the corresponding invariant predictor structure.

## Proof bottleneck
The IRM constraint can be satisfied by non-causal or degenerate predictors unless environment variation and model structure rule them out.

## Proof-enabling property
Linear structural algebra and rank/heterogeneity conditions expose exactly which directions are identifiable from the collection of environments.

## Main theorem
Theorem 1-3 characterize when invariant risk minimization captures invariance and when spurious solutions remain; the conclusions are model- and environment-dependent.

## Price of tractability
Restricted linear/structural models, realizability and sufficiently informative environments. Results do not lift automatically to deep IRMv1.

## Algorithm relationship
CONDITIONAL: theorem-level analysis of IRM-like population conditions, not an exact general deep optimizer theorem.

## Scope
Naturally covered: identifiability diagnostics for IRM. Excluded: arbitrary representation entanglement and optimizer-statistic methods.

## Source-only status
YES under structural assumptions.

## Evidence
Theorems 1-3; AISTATS/PMLR 130:4069-4077.
