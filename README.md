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

The first path-functional probe is archived in
`round3_redesign/vrex_trajectory_microscope/`. V-REx seeds 11, 12, and 13 were
reused to compute early/middle/late cumulative functional updates, signed
leave-one-seed-out projections, cancellation efficiency, and source-side
recovery rates. A single first-divergence event was already rejected as a
sufficient explanation. The new probe finds a candidate pattern in late
functional cancellation (seed 12 has higher late total and clean-bank
efficiency), while recovery rate and reference alignment fail to separate the
same success case. These are path-statistic candidates only: the probe does
not yet estimate time-varying \(A_t\), \(O_t\), or a target-relevant gold
direction.

The signed path-budget follow-up is archived in
`round3_redesign/vrex_signed_path_budget_v2/`. It constructs a source-visible
task-response direction through a dynamic (A_t/O_{S,t}) projector and a
head-to-bank pullback, then decomposes each functional update into signed and
orthogonal budget. This probe falsifies the stronger version of the
late-coherence hypothesis: seed 12 has the best target accuracy but the lowest
late useful-budget fraction under this current proxy (0.243 versus 0.276 and
0.251). Thus low cancellation and source-visible projection are not yet a
sufficient mechanism; the earlier coherence result remains an optimizer/path
statistic candidate pending a representation-aware residual and held-out
validation.

The representation/readout split is archived in
`round3_redesign/vrex_representation_head_split_v4/`. The exact identity
(Delta f=H_{t-1}Delta W+Delta H W_{t-1}+Delta HDelta W) shows that late
functional motion is representation-dominated for all three V-REx seeds. Seed
12 has the highest cancellation efficiency of the representation-induced
functional component (0.198 versus 0.135 and 0.147), while latent activation
cancellation itself is not uniquely high. This narrows the candidate to
coherent *functional use of representation changes*, rather than simply stable
features or stable readout. It remains a three-seed correlation and requires
matched reruns, task-relevant residuals, and intervention.

### Method-agnostic mechanism and local-regret audit

The independent audit in `round3_redesign/method_agnostic_mechanism/` connects
the source-side trajectory analysis to a controlled Gaussian truth model and a
conditional local response-regret bound. The Gaussian track checks

\[
\Pi=-(H_R+\lambda K)^{-1}(B_R+\lambda C)
\]

and the common-base cells `Pi00`, `PiC0`, `Pi0K`, `PiCK`, while keeping pure C/K
increments, interaction, and Shapley shares distinct. It reports finite-sample
operator errors, resolvent certificates, spectral-gap rejection, and independent
trust-region checks. Median \(\Pi\) error decreases from 0.129 at \(n=128\) to
0.0347 at \(n=2048\) in the designed fixture; the confidence certificate
abstains when the inverse margin is not positive.

The V-REx track freezes source-only features before reading target outcomes and
tests three pre-registered hypotheses: observable transfer, late settling, and
state transition. Observable transfer and late settling rank seed 12 first in
the primary late window, but the evidence is only three seeds and does not
surpass method/path-length controls. State transition does not separate the
successful seed. The result is therefore `INSUFFICIENT_VALIDATION`: these are
numeric path candidates, not identified forcing/filtering mechanisms.

The finite-horizon audit verifies the full optimizer-state chain rule, shows
that non-commuting update order matters even when Jacobian spectra match, and
rejects literal noisy-kernel rank as a stable projector. The accompanying
report is [mechanism_theorem_report.md](round3_redesign/method_agnostic_mechanism/mechanism_theorem_report.md),
with hashes and commands in `provenance.json`. No CMNIST classification DG
theorem or causal mechanism claim is made.

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
### Dynamic geometry follow-up (V-REx seeds 11/12/13)

The next mechanism probe evaluates local task and source-observation geometry at checkpoints 0, 25, ..., 500 while keeping target outcomes out of geometry construction and checkpoint choice. Results are stored in `round3_redesign/vrex_dynamic_geometry_v2/`. The three runs keep rank(A)=5 and rank(O)=3; seed 12 combines the best late functional cancellation efficiency with a larger late mean O norm, while A scale is nearly unchanged. This is only a candidate joint path/exposure pattern. A and O have different codomains, so the current pass reports their spectra separately and does not form an invalid direct projection. A valid common-space pullback and held-out seeds are required before calling this a mechanism.

### Cross-method CMNIST trajectory audit (IRMv1 / V-REx / Fishr / BIRM)

A unified source-only pass now compares the 501-step ColoredMNIST trajectories for
full-network IRMv1, V-REx and Fishr, with BIRM and LoRA-BIRM retained as a separate
sparse-checkpoint representation/head-level track. Results are in
`round3_redesign/method_agnostic_mechanism/cmnist_cross_method_report.md` and the
feature tables `cmnist_method_path_features.csv` and
`cmnist_rephead_method_features.csv`.

The late path statistics do not support a method-independent cancellation mechanism:
IRMv1 has the highest target mean (~0.675) but the lowest late cancellation efficiency
(~0.095), while Fishr (~0.552 target) and V-REx (~0.528 target) have higher values
(~0.238 and ~0.158). Thus cancellation/coherence is method-conditioned and confounded
with path length and optimizer dynamics. BIRM/LoRA-BIRM reach approximately 0.705/0.735
best target accuracy in the head-only protocol, but their geometry is not pooled with
full-network Pi. The current conclusion remains `numeric_geometry_family`, with no
forcing/filtering claim.

The candidate-mechanism audit is implemented in `src/ood_repr_reg/audit_candidate_mechanisms.py`. It explicitly tests the earlier V-REx hypotheses on IRMv1 and Fishr CMNIST trajectories using within-method seed rankings and pooled cross-method checks. The audit finds no method-independent mechanism: cancellation agrees with target only in selected windows/methods and reverses in pooled data; adjacent cosine and clean/source transfer produce multiple within-method counterexamples. BIRM/LoRA-BIRM remain sparse head-only data and are therefore not used to test full-network path hypotheses.

### Mechanism identification revision

The trajectory audits are observational, even when expanded to five seeds. They cannot identify forcing/filtering causality because method, optimizer state, representation state, and measured geometry co-vary. A mechanism claim now requires matched fork interventions from the same complete checkpoint state: change only forcing (`C`), only filtering (`K`), or representation/readout coupling while replaying identical data and randomness. Until those forks are run, `(C,K)->Pi` remains a theoretical coordinate system and CMNIST path statistics remain descriptive.

### Data-first mechanism-atom pilot

`src/ood_repr_reg/mechanism_atom_discovery.py` implements the SINDy-style first stage proposed in the methodology revision. A fixed neutral source-side atom library predicts the 2048-dimensional next functional update, with leave-one-seed-out evaluation on five-seed CMNIST trajectories. Held-out vector R² is 0.296±0.089 (IRMv1), 0.212±0.080 (V-REx), and 0.216±0.130 (Fishr). This establishes limited generative predictability of observed updates, not causal mechanism identification. Selected atoms and the audit are in `round3_redesign/method_agnostic_mechanism/mechanism_atom_*`; held-out intervention and matched fork validation remain required.

### Matched fork intervention pilot

The first controlled perturbation-response dataset is in `round3_redesign/method_agnostic_mechanism/regularizer_forks/`. For each IRMv1, V-REx, and Fishr CMNIST run (five seeds), the complete step-300 model, Adam state, algorithm state, and source batch schedule were cloned. Branches changed only the post-anneal regularizer scale (0, 0.5, 1, 2) and were rolled out for horizons 1, 5, and 20. Duplicate control replays matched exactly in parameters, optimizer state, algorithm state, and functional response.

The intervention produces large, reproducible short-horizon effects when the regularizer is removed (mean horizon-5 effect norms: IRMv1 69.95, V-REx 36.15, Fishr 34.04) and much smaller effects for half/double scaling. This validates the matched-fork apparatus and establishes intervention sensitivity. It does **not** isolate forcing `C` from filtering `K`, nor does it establish an OOD mechanism; those require separate common-base component interventions.

The initial C/K factorial pilot is recorded in `round3_redesign/method_agnostic_mechanism/ck_factorial/`. It is currently a numerical diagnostic, not a mechanism result: absolute epsilon was not trust-region calibrated, producing method-dependent blow-up in C-driven steps (especially IRMv1/V-REx). The scientific gate now requires common metric step normalization, finite-difference linearity, and projected-Hessian conditioning checks before interpreting forcing/filtering or interaction effects.
