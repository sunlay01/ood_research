# Decision Log

Negative decisions are first-class state. Historical files remain available as
evidence but are not default current-state authority.

## D-SEMANTIC-LATENT-SUPERSEDED

- date/stage: 2026-09-09 state migration
- status: `SUPERSEDED`
- decision: Semantic latent decomposition / LATENT-001 is historical, not the
  current mainline.
- reason: The accepted current state is Round-3 operator/source-information /
  spectral-regret after Task 1, Task 2, and repair evidence gate.
- consequence: Future agents must not start from LATENT-001 as the active
  paradigm unless a task explicitly asks for historical comparison.
- canonical evidence: `CURRENT_STATE.md`, `docs/state/RESULT_REGISTRY.json`.
- reopen condition: An explicit task asks to compare against or revive a
  semantic-latent track.

## D-ROUND2-QUOTIENT-SUPERSEDED

- date/stage: 2026-09-09 state migration
- status: `SUPERSEDED`
- decision: Round-2 quotient-style formulations are not the canonical current
  ambiguity interface.
- reason: Current source ambiguity is represented through `O_S`, `ker O_S`,
  `A_irr`, `A_rec`, and corrected `E=A_rec+Pi O_S`.
- consequence: Do not reconstruct current theory from Round-2 unless explicitly
  required.
- canonical evidence: `docs/state/THEOREM_REGISTRY.md`.
- reopen condition: A task explicitly modifies historical quotient comparisons.

## D-REG-EXPOSURE-CONTAINMENT-REJECTED

- date/stage: post Task 2 / repair evidence gate
- status: `REJECTED`
- decision: The containment `Im(K_j|_R) subset R_exp` is not necessary.
- reason: The sharp condition is spectral slack placement of `E E^*`, not a
  regularizer/exposure containment assumption.
- consequence: Do not use containment as an assumption, theorem target, design
  principle, or explanation of Task 1 results.
- canonical evidence: `docs/state/THEOREM_REGISTRY.md`, `tests/test_sharp_optimality.py`.
- reopen condition: Only if a future task studies containment as a separate
  sufficient condition.

## D-ACTUAL-VS-COMMON-BASE-SEPARATION

- date/stage: Task 1 algorithm mechanism
- status: `FROZEN`
- decision: Actual-solution mechanism `(g_j,K_j,C_j)` and common-base
  counterfactual attribution are separate objects.
- reason: Exact IFT rows evaluate the method's actual source solution;
  common-base rows decompose forcing/filtering at a shared ERM base.
- consequence: Do not report common-base attribution as the actual method
  mechanism.
- canonical evidence: `round3_redesign/repair_evidence_gate/results/task1_summary.json`.
- reopen condition: A future Task 1 extension changes the mechanism interface.

## D-SHARP-SLACK-REPLACES-E0

- date/stage: Task 2 sharp optimality
- status: `FROZEN`
- decision: Spectral slack is the sharp adaptive optimality criterion; `E=0`
  is sufficient but not necessary in partial-information settings.
- reason: `R_adap=R_info iff E E^* <= S_slack` after support compatibility.
- consequence: Do not restore `E=0` as a necessary condition.
- canonical evidence: `round3_redesign/repair_evidence_gate/results/task2_spectral_rows.csv`.
- reopen condition: Only if the spectral theorem itself is modified.

## D-TASK3-NEXT

- date/stage: 2026-09-09 state migration
- status: `ACTIVE`
- decision: Task 3 applicability is the next research task.
- reason: Task 1/Task 2 and the repair gate are complete; the unresolved
  question is whether the quantities discriminate actual method behavior.
- consequence: Record Task 3 as next, but do not execute it in this migration.
- canonical evidence: `CURRENT_STATE.md`, `docs/research/open_questions.md`.
- reopen condition: A new active task changes the queue.

## D-MECH001-C011-DO-NOT-RESTORE

- date/stage: retained repository policy
- status: `DO_NOT_RESTORE`
- decision: `MECH-001/C011` must not be restored, reconstructed, summarized, or
  cited for substance.
- reason: Existing repository policy says the erroneous material was deleted at
  user request.
- consequence: State layer records only `do_not_restore=true` and
  `do_not_reference=true`.
- canonical evidence: `docs/state/FILE_STATUS.json`.
- reopen condition: none, unless the user explicitly revokes this policy.
