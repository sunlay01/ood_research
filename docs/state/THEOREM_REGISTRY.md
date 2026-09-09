# Theorem Registry

Status vocabulary: `FROZEN`, `ACTIVE`, `PARTIAL`, `REJECTED`, `SUPERSEDED`.

## T-3A-A -- Source-Whitened Response Operator

- status: `FROZEN`
- statement: At source optimum `w*`, with `H_S=nabla^2 R_S(w*)` and
  `z=H_S^{1/2}(w-w*)`, the world tangent response is `A u=H_S^{-1/2}g_u`.
- assumptions: finite-dimensional local tangent, declared source metric, source
  optimum and Hessian nonsingular in the audited setting.
- scope: response geometry only; not a target-risk theorem.
- canonical implementation / test: `src/ood_repr_reg/round3r_3e_world_tangent.py`,
  `src/ood_repr_reg/environment_family/geometry.py`, `tests/test_environment_family_refactor.py`.
- canonical proof/evidence pointer: `round3_redesign/repair_evidence_gate/results/family_provenance.json`.
- supersedes / superseded_by: supersedes older semantic-latent current-state interface.
- do_not_overinterpret: Does not identify causal mechanisms or universal OOD risk.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3C-PI -- Exact IFT Source-Adaptive Response

- status: `FROZEN`
- statement: At method `j`'s actual source solution,
  `Pi_j=-(H_R,j+lambda K_j)^-1(B_R,j+lambda C_j)`.
- assumptions: local differentiability and invertible positive local metric for
  the valid method/lambda row.
- scope: local source-side optimizer response for L2, IRMv1, and V-REx valid rows.
- canonical implementation / test: `src/ood_repr_reg/algorithm_mechanism/core.py`,
  `src/ood_repr_reg/run_algorithm_mechanism.py`, `tests/test_algorithm_mechanism.py`.
- canonical proof/evidence pointer: `round3_redesign/repair_evidence_gate/results/task1_summary.json`.
- supersedes / superseded_by: separates actual solution mechanism from common-base diagnostics.
- do_not_overinterpret: Tiny reconstruction error validates the implementation
  identity, not universality of a mechanism interpretation.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3C-COMMON-BASE -- Symmetric Common-Base Attribution

- status: `FROZEN`
- statement: At a shared ERM base,
  `Pi_CK-Pi_00=Delta_C^sym+Delta_K^sym`, where forcing and filtering are
  symmetric counterfactual components.
- assumptions: common source base and the same source-state Jacobian blocks.
- scope: counterfactual attribution only; not the actual method solution.
- canonical implementation / test: `src/ood_repr_reg/algorithm_mechanism/core.py`,
  `tests/test_algorithm_mechanism.py`.
- canonical proof/evidence pointer: `round3_redesign/repair_evidence_gate/results/task1_common_base_attribution.csv`.
- supersedes / superseded_by: none.
- do_not_overinterpret: Does not replace actual-solution mechanism quantities.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3D-IDENTIFIABILITY -- Source Identifiability Criterion

- status: `FROZEN`
- statement: For source observation `O_S: U -> Y`, zero structural
  information loss holds iff `ker O_S subset ker A`.
- assumptions: fixed family tangent and response operator.
- scope: family-relative source identifiability.
- canonical implementation / test: `src/ood_repr_reg/round3r_3d_identifiability.py`,
  `src/ood_repr_reg/environment_family/geometry.py`, `tests/test_environment_family_refactor.py`.
- canonical proof/evidence pointer: `round3_redesign/repair_evidence_gate/results/family_provenance.json`.
- supersedes / superseded_by: supersedes older Round-2 quotient current interface.
- do_not_overinterpret: Distinguish structural source-null ambiguity from finite
  design exposure.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3E-INFO-FLOOR -- Information Floor

- status: `FROZEN`
- statement: With `P=P_ker(O_S)`, `A_irr=A P`, and
  `alpha=||A_irr||_op`, the source-information floor is `R_info=1/2 alpha^2`.
- assumptions: declared world metric has been whitened.
- scope: finite-dimensional response-recovery regret floor.
- canonical implementation / test: `src/ood_repr_reg/sharp_optimality/geometry.py`,
  `tests/test_sharp_optimality.py`, `tests/test_repair_evidence_gate.py`.
- canonical proof/evidence pointer: `formalization/OODRelevance/OptimalRecovery.lean`,
  `round3_redesign/repair_evidence_gate/results/task2_summary.json`.
- supersedes / superseded_by: none.
- do_not_overinterpret: Not a DG target-risk lower bound.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3E-SPECTRAL-SLACK -- Sharp Adaptive Spectral Slack

- status: `FROZEN`
- statement: With `E_j=A_rec+Pi_j O_S` and
  `S_slack=alpha^2 I-A_irr A_irr^*`, adaptive optimality satisfies
  `R_j^adap=R_info iff E_j E_j^* <= S_slack`. If `E_j E_j^*` spills outside
  `Im(S_slack)`, `rho_slack=+inf`.
- assumptions: finite-dimensional whitened family tangent and valid affine policy row.
- scope: adaptive-only sharp optimality.
- canonical implementation / test: `src/ood_repr_reg/round3r_3e_c_spectral.py`,
  `src/ood_repr_reg/sharp_optimality/geometry.py`, `tests/test_sharp_optimality.py`.
- canonical proof/evidence pointer: `formalization/OODRelevance/SharpGeometry.lean`,
  `round3_redesign/repair_evidence_gate/results/task2_spectral_rows.csv`.
- supersedes / superseded_by: supersedes old `E=0` necessity criterion.
- do_not_overinterpret: `E=0` is sufficient but not necessary in partial information.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-3E-AFFINE-REGRET -- Full Affine Regret and Static Tax

- status: `FROZEN`
- statement: `R_j=1/2 sup_{||u||<=1} ||z_j^0+(A+Pi_j O_S)u||^2`, with
  static steering lower bound `R_j >= R_info + 1/2 ||z_j^0||^2`.
- assumptions: finite-dimensional unit-ball tangent and valid shifted-ball solve.
- scope: family-conditioned affine decision geometry.
- canonical implementation / test: `src/ood_repr_reg/round3r_3e_joint_regret.py`,
  `src/ood_repr_reg/run_sharp_optimality.py`, `tests/test_round3_joint_affine_regret.py`.
- canonical proof/evidence pointer: `formalization/OODRelevance/JointAffine.lean`,
  `round3_redesign/repair_evidence_gate/results/task2_theorem_audit.csv`.
- supersedes / superseded_by: none.
- do_not_overinterpret: Not a finite-sample or universal DG guarantee.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-OLD-E0-SHARP -- Old E=0 Sharp-Optimality Necessity

- status: `SUPERSEDED`
- statement: `E=0` was previously treated as the sharp criterion for full
  minimax optimality.
- assumptions: none retained as current theorem assumptions.
- scope: historical criterion only.
- canonical implementation / test: `tests/test_sharp_optimality.py`.
- canonical proof/evidence pointer: `docs/state/DECISION_LOG.md`.
- supersedes / superseded_by: superseded_by `T-3E-SPECTRAL-SLACK`.
- do_not_overinterpret: Do not restore as a necessary condition.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.

## T-REG-EXPOSURE-CONTAINMENT -- Regularizer/Exposure Containment Necessity

- status: `REJECTED`
- statement: The containment `Im(K_j|_R) subset R_exp` is not necessary for the
  current sharp optimality theory.
- assumptions: rejected as a required assumption.
- scope: negative decision for current mainline.
- canonical implementation / test: `tests/test_sharp_optimality.py`.
- canonical proof/evidence pointer: `docs/state/DECISION_LOG.md`.
- supersedes / superseded_by: none.
- do_not_overinterpret: Do not use as theorem target, design principle, or Task 1 explanation.
- last_validated_commit: `9715f535dc7f5ab06a2b7ec01f3026d9705719c8`.
