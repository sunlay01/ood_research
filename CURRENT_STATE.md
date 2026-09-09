# Current Project State

## 1. Snapshot

- Current stage: Round 3 redesign after Task 1, Task 2, and repair evidence gate.
- Latest validated artifact commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.
- Repair code-base commit recorded by evidence: `ef134888809d4b093758887dedd6b8d5fbaeb05a`.
- Repair gate: `REPAIR-EVIDENCE-PASS`; readiness: `READY-FOR-TASK-3`.
- LATENT-001 / semantic latent decomposition is not the current mainline.

## 2. Current Research Question

Task 3 is next:

```text
Do the current theoretical quantities produce nontrivial, interpretable,
discriminative predictions on actual DG methods and benchmarks?
```

Task 3 is not executed by this state-management task.

## 3. Frozen Mainline

- 3A response operator: `A: U -> Z`, with `z=H_S^{1/2}(w-w*)` and
  `A u = H_S^{-1/2} g_u`. Registry: `T-3A-A`.
- 3C exact source-adaptive IFT response:
  `Pi_j = -(H_R,j + lambda K_j)^-1 (B_R,j + lambda C_j)`. Registry: `T-3C-PI`.
- 3C common-base attribution is a separate counterfactual track, not the actual
  solution mechanism. Registry: `T-3C-COMMON-BASE`.
- 3D source observation operator: `O_S: U -> Y`; zero information loss iff
  `ker O_S subset ker A`. Registry: `T-3D-IDENTIFIABILITY`.
- 3E corrected decomposition after metric whitening:
  `P=P_ker(O_S)`, `Q=I-P`, `A_irr=A P`, `A_rec=A Q`, and
  `E_j=A_rec+Pi_j O_S`. Registry: `T-3E-INFO-FLOOR` and
  `T-3E-SPECTRAL-SLACK`.
- Full affine regret:
  `R_j = 1/2 sup_{||u||<=1} ||z_j^0 + (A + Pi_j O_S)u||^2`. Registry:
  `T-3E-AFFINE-REGRET`.

## 4. Frozen Numerical Evidence

- Repair gate: R1/R2/R3 all `REPAIR-PASS`. Result: `R-REPAIR-GATE`.
- Full pytest: `259 passed`, `0 failed`. Result: `R-PYTEST-FULL`.
- Lean: build completed successfully. Result: `R-LEAN-BUILD`.
- Task 1: `220` rows, `ALGORITHM-MECHANISM-PASS`, max Pi reconstruction
  relative error `3.396285345773631e-10`. Result: `R-TASK1-MECHANISM`.
- Task 2: `220` rows, `SHARP-OPTIMALITY-PASS`, corrected snapshot,
  coordinate invariance, cross-operator audit, and counterexamples all pass.
  Result: `R-TASK2-SHARP`.
- Natural nonzero-`E` inside-slack count: `9`. Result: `R-TASK2-SHARP`.
- Legacy hidden-U: dimension `8`, rank `O_S=7`, dim `ker O_S=1`, rank
  `A_irr=1`, information floor `0.05118145108608892`. Result:
  `R-LEGACY-HIDDEN-U`.
- Source-induced comparison: dimension `4`, rank `O_S=4`, dim `ker O_S=0`,
  rank `A_irr=0`, information floor `0.0`. Result: `R-SOURCE-INDUCED`.

These are registry facts, not universal scientific claims.

## 5. Open Issues

- Whether Task 1/Task 2 quantities discriminate actual method behavior remains
  the next research question.
- CMNIST evidence is a frozen-head empirical bridge, not a finite-sample or
  universal DG theorem.
- Source-response recovery evidence is not a target-risk lower bound.

## 6. Next Authorized Research Tasks

- Task 3: applicability of `(g,K,C)`, `Pi`, `E`, `rho_slack`, `R_info`, and
  `z0` to actual DG method behavior.
- Task 4: ex-ante helps/hurts prediction from local source/family geometry.
- Task 5: local-to-robust validity conditions and remainder control.

## 7. Explicitly Superseded / Rejected Paths

- Semantic latent decomposition / LATENT-001 is historical and superseded as the
  active mainline. Decision: `D-SEMANTIC-LATENT-SUPERSEDED`.
- Round-2 quotient-style formulations are historical, not the canonical current
  ambiguity interface. Decision: `D-ROUND2-QUOTIENT-SUPERSEDED`.
- Regularizer/exposure containment is rejected as a necessary condition.
  Decision: `D-REG-EXPOSURE-CONTAINMENT-REJECTED`.
- The old `E=0` sharp-optimality necessity is superseded by spectral slack.
  Decision: `D-SHARP-SLACK-REPLACES-E0`.

## 8. Do-Not-Reopen / Do-Not-Restore

- `MECH-001/C011`: `do_not_restore=true`, `do_not_reference=true`. Decision:
  `D-MECH001-C011-DO-NOT-RESTORE`.
- Historical files may be reopened only under the `AGENTS.md` reopen protocol.

## 9. Canonical Files

- Boot protocol: `AGENTS.md`.
- Theorem registry: `docs/state/THEOREM_REGISTRY.md`.
- Result registry: `docs/state/RESULT_REGISTRY.json`.
- Decision log: `docs/state/DECISION_LOG.md`.
- File-status registry: `docs/state/FILE_STATUS.json`.
- Current open questions: `docs/research/open_questions.md`.
- Repair evidence: `round3_redesign/repair_evidence_gate/results/repair_status.json`.

## 10. Last Validation

- State migration baseline: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.
- Repair evidence code base: `ef134888809d4b093758887dedd6b8d5fbaeb05a`.
- Validation artifacts: `round3_redesign/repair_evidence_gate/results/regression_status.json`.
