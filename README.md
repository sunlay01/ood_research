# OOD Representation Regularization Research

This repository contains the historical and current research artifacts for the
OOD representation regularization project.

## Current Authoritative State

The single source of truth for the current research state is:

- [CURRENT_STATE.md](CURRENT_STATE.md)

New agents should start with:

1. [AGENTS.md](AGENTS.md)
2. [CURRENT_STATE.md](CURRENT_STATE.md)
3. active task files only if `active/TASK.md` and `active/CONTEXT.md` exist

The current mainline is the Round-3 operator/source-information/spectral-regret
framework after Task 1, Task 2, and the repair evidence gate. Task 3
applicability is next, but it is not executed by this state-management layer.

## Research Plan (Archived)

The project studies **OOD generalization error bounds**. Round 1 through Round
3 are one continuous line of work:

1. Transform OOD generalization into a local-geometric problem around the
   training solution.
2. Measure the microscopic local behavior of different OOD algorithms.
3. Let recurring geometric structure reveal latent success mechanisms, rather
   than imposing semantic mechanism labels in advance.
4. Decompose the resulting mechanisms into a method-independent estimate of
   OOD generalization error, explaining both help and hurt cases.
5. Test whether the decomposition can predict help, neutral, or hurt behavior
   before evaluating the target domain, and establish local-to-robust validity
   conditions.

The current mechanism hypothesis uses the analysis coordinates

\[
(C_j, K_j, g_j) \longrightarrow \Pi_j \longrightarrow E_j
\longrightarrow \mathfrak R_j,
\]

where \(C_j\) describes environment sensing/forcing, \(K_j\) curvature
filtering, \(g_j\) (or \(z_j^0\)) static steering, \(\Pi_j\) the resulting
source-adaptive response, and \(E_j\) the recoverable response residual. These
are hypotheses and analysis coordinates to be validated empirically, not
already-established mechanism labels.

### Project stages

- **Task 0 — Geometry / CMNIST bridge:** establish the experimental bridge
  between local geometry and observed DG behavior.
- **Tasks 1–2 — Theory and repair:** establish source exposure,
  identifiability, information floor, spectral slack, affine regret, and verify
  the implementation and formalization.
- **Task 3 — Empirical mechanism decomposition (current):** collect local
  geometric data across methods, seeds, environment families, regularization
  strengths, and training stages; discover stable clusters or structures; then
  interpret them post hoc and test whether they explain both successful and
  failed cases.
- **Task 4 — Ex-ante prediction:** test whether the discovered decomposition
  predicts help/neutral/hurt behavior from source-side geometry.
- **Task 5 — Local-to-robust theory:** establish validity conditions and control
  approximation remainders.

The CMNIST experiments are evidence for Task 3, not a universal DG theorem.
Likewise, the proposed decomposition is not considered complete until the
data-driven discovery and cross-method validation succeed.

## Registries

- [Theorem Registry](docs/state/THEOREM_REGISTRY.md)
- [Result Registry](docs/state/RESULT_REGISTRY.json)
- [Decision Log](docs/state/DECISION_LOG.md)
- [File Status Registry](docs/state/FILE_STATUS.json)
- [Open Questions](docs/research/open_questions.md)
- [State Migration Report](docs/state/STATE_MIGRATION_REPORT.md)

## Historical Material

Earlier semantic-latent, Round-1, Round-2, and previous Round-3 artifacts remain
available as historical evidence. They are not default authority for the current
research state unless a registry entry explicitly points to them.
