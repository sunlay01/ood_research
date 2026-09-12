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

## Research Objective (Current)

The primary objective is to explain and estimate OOD generalization error from
the local geometry of the learner, rather than from method names or a single
post-hoc accuracy statistic. The project follows one continuous question:

> Given a source environment family and a trained learner, which stable
> microscopic geometric structures determine whether an OOD method helps,
> remains neutral, or hurts, and how can those structures yield a
> method-independent, source-side estimate of the resulting OOD error?

The intended contribution has three linked parts:

1. Discover mechanisms from local response data across methods, seeds,
   regularization strengths, environment families, and training stages. Numeric
   structure must be found before any semantic mechanism name is assigned.
2. Formalize the discovered structure as a local OOD error decomposition that
   separates source-information limits from algorithm-dependent residuals and
   includes estimation and local-approximation errors.
3. Validate the decomposition without target-domain selection: it must explain
   paired successful and failed cases, survive controls for method identity,
   \(\lambda\), and shift family, and state where the local approximation
   breaks down.

A lightweight algorithm is a downstream objective, not the starting premise.
It will be designed only after a mechanism survives common-base
counterfactuals and cross-family validation. Such an algorithm must target a
measured residual, use matched training budgets, and report the settings in
which it does not help. The project does not claim a universal DG theorem or a
complete mechanism decomposition until these empirical and theoretical gates
are passed.

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

## Task 3 Data-Analysis Protocol (Archived)

Task 3 reads every result through six layers. The purpose is to discover
microscopic mechanisms from local geometry, while explaining both successful
and failed cases. No single scalar, method name, or semantic label is treated
as a mechanism by itself.

1. **Source exposure.** Measure the source observation operator \(O_S\), its
   exposed rank and kernel, and the information floor
   \(\|A P_{\ker O_S}\|_{\mathrm{op}}\). First ask whether the target-relevant
   variation was observable from source data at all. More nuisance diversity
   does not imply more task-relevant exposure.
2. **Adaptive response.** Measure whether the learner response
   \(\Pi O_S\) compensates the recoverable target response
   \(A_{\mathrm{rec}}\). The primary residual is
   \(E=A_{\mathrm{rec}}+\Pi O_S\). Response magnitude alone is not evidence of
   a correct response; direction and source-visible compensation are required.
3. **Static steering.** Measure \(z^0\), the predictor displacement present
   without an environment perturbation. Compare its size and direction with
   finite target behavior, while actively checking cases where a smaller
   \(\|z^0\|\) does not produce better target performance.
4. **Spectral placement.** Compare \(EE^\ast\) with the information-floor slack
   (S_{\mathrm{slack}}=\alpha^2I-A_{\mathrm{irr}}A_{\mathrm{irr}}^\ast), where
   \(\alpha=\|A_{\mathrm{irr}}\|_{\mathrm{op}}\). A nonzero residual can still be
   information-optimal when it lies inside the available slack. Always test
   equal-norm, different-direction counterexamples.
5. **Comparative mechanism.** Compare help, neutral, and hurt cases in pairs:
   same method across \(\lambda\), same method across seeds, and different
   methods under matched source exposure. The unit of evidence is a
   `(method, lambda, seed, environment family)` case, not a method-level
   average. A proposed mechanism must separate paired successes and failures,
   not merely describe a bad case after the fact.
6. **External validity.** Test whether local quantities predict finite-domain
   behavior as the shift radius grows. Report local ranking, degradation, and
   ranking reversal separately. A breakdown at larger radius is a validity
   boundary of the local theory, not automatically an implementation failure.

Three confounders are controlled in every comparison: **method identity**,
regularization strength \(\lambda\), and **environment family / shift radius**.
Mechanism quantities must provide information beyond method identity, source
risk, simple source statistics, and \(\lambda\). Target accuracy is post-hoc
evaluation only; it cannot select methods, hyperparameters, checkpoints,
clusters, or response directions.

### Training-Trajectory Microscope

Final-state (A,O_S,\Pi,E) describe what happened. Task 3 also records the
training path and, where available, decomposes each update
\(\Delta w_t=w_{t+1}-w_t\) against fixed opaque gold/reference directions:

\[
\operatorname{proj}_{d_k(t)}\Delta w_t,\qquad
\cos(\Delta w_t,d_k(t)).
\]

The trajectory analysis asks when a method first enters a useful response
subspace, when curvature filtering changes its direction, and whether a
successful and failed run share an early path before diverging. Gold directions
are analysis references, not semantic labels; clusters are discovered from
numeric geometry first and named only post hoc. The same plots and tables must
include successful and failed cases.

### Evidence Status of Current Artifacts

The official BIRM/LoRA-BIRM rerun stores checkpoints at steps 0, 100, 200, 300,
400, 500, and 501 for five seeds. Static representation diagnostics are in
`round3_redesign/birm_cmnist_checkpoint_rerun/checkpoint_geometry.csv`.
The BIRM-specific (A/O_S/\Pi) extraction is in
`round3_redesign/birm_cmnist_checkpoint_rerun/birm_lora_aopi_final.csv`.
The current BIRM/LoRA-BIRM (\Pi) table is explicitly a fixed-encoder,
head-only, source-expected-risk continuation diagnostic. It is not yet the
full-network optimizer-state (\Pi) used by the earlier common-harness
experiments. Full comparability requires replaying each upstream algorithm's
source-only continuation with its own optimizer and algorithm state.

The first data-first mechanism discovery pass is archived in
`round3_redesign/algorithm_mechanism_discovery/`. It uses only source-side
features for clustering and keeps target outcomes in a separate external
audit. The full-network (k=2) solution is stable but separates ERM from
IRMv1/V-REx/Fishr, so it is currently a method/protocol family rather than a
mechanism family. The representation/head solution is less stable and is kept
separate because BIRM/LoRA use a head-only continuation. These are explicit
negative findings: no mechanism label is promoted until it survives controls
for method identity, \(\lambda\), and environment family and passes common-base
counterfactuals.

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
