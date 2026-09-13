# Historical status of the mechanistic line

This document records what the previous unified project established and what it did not. It is a status ledger, not a claim that one representation explains all OOD behavior.

## Established infrastructure

- Deterministic CMNIST and smooth-world source schedules.
- Matched continuation forks with complete model, optimizer, algorithm-state, and logical-step bundles.
- Model/optimizer/algorithm-state hash gates and initial functional-bank equality checks.
- Semantic perturbation probes, functional banks, finite-difference response vectors, and independent continuation schedules.
- A/O/Pi and C/K diagnostic implementations with explicit validity and provenance fields.
- Synthetic and CMNIST trajectory, representation/head, response-geometry, memory, and factorial audit harnesses.

The main implementations remain in `src/ood_repr_reg/` and the corresponding artifacts remain in `round3_redesign/`; this branch adds a scientific routing layer without rewriting their imports.

## Positive but limited findings

- Some synthetic failure-memory probes show controllable effects in deliberately constructed worlds.
- Method-specific response structure is reproducible in the small CMNIST panel.
- Full `A` geometry can strongly predict selected finite semantic behaviors in some cross-method audits.
- Complete checkpoint bundles and matched forks make local intervention claims substantially more reliable than earlier trajectory correlations.

These findings concern local or finite behavior under declared probes. They do not establish a universal mechanism or target-risk theorem.

## Negative findings

- Generic memory is insufficient as an explanation.
- Standalone CMNIST memory methods are not competitive under the common harness.
- Simple path-value scoring failed to transfer reliably.
- Active probing produced weak or confounded signals.
- Stable low-rank response geometry does not imply semantic recovery.
- C/K geometry did not yield a universal mechanism.
- After full A/O controls and independent schedules, no stable incremental `Pi O_S` mechanism remained.
- Method identity can explain substantial finite behavior, so geometry must beat that baseline before it is called method-agnostic.

## Interpretation

The negative evidence motivates the branch split. It does not prove that microscopic mechanism research is useless. It shows that local algorithmic evidence and macroscopic statistical OOD theory answer different questions and require different standards of representation and validation.
