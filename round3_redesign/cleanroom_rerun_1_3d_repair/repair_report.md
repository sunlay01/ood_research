# Clean-room Repair Round Report

Task: `TASK-RERUN-1-3D-CLEANROOM` focused repair round

Final repaired verdict: `RERUN-INCONCLUSIVE`

## What Was Repaired

- CMNIST `O` and `Pi O` are now built from source images and source labels only.
- Target samples/labels are used only for post-hoc `A`, target risk/accuracy, and bridge comparison.
- The old target-built 8D PCA bridge is replaced by a per-seed source-only projection common to ERM and IRMv1, retaining dimensions `56` to `64`.
- Old-vs-new diff now computes actual `ECK - E00` direction changes.

## Repaired CMNIST Bridge

- ERM mean target accuracy: `0.10687872363214797`.
- IRMv1 mean target accuracy: `0.6780484521511904`.
- ERM mean `||E||`: `3.724776969281058`.
- IRMv1 mean `||E||`: `3.515780491452099`.
- Rows: `191`; errors: `9`.

This is still not a neural scientific PASS. The family is declared and source-target-coupled, and the projection is non-invertible for some seeds even though it is source-only and common across ERM/IRMv1.

## Repaired Task2 CMNIST

```json
{
  "cmnist_erm": {
    "audit_mode": "source_only_fast_world_space_no_coordinate_transport",
    "audit_row_count": 94,
    "coordinate_invariance_pass": "NOT_RUN_REPAIR_HIGH_DIMENSION_COST",
    "corrected_snapshot_pass": true,
    "cross_operator_audit_pass": true,
    "error_count": 0,
    "errors": [],
    "expected_audit_row_count": 94,
    "full_coverage_pass": true,
    "operator_snapshot_source": "/Users/sunlay/Desktop/ood-representation-regularization/round3_redesign/cleanroom_rerun_1_3d_repair/results/source_only_bridge/erm_operator_snapshots.npz",
    "partial_reason": "Audits pass/fail on repaired source-only bridge, but CMNIST remains empirical and family-coupled; no full scientific PASS is claimed.",
    "verdict": "SHARP-OPTIMALITY-CMNIST-SOURCE-ONLY-PARTIAL"
  },
  "cmnist_irmv1": {
    "audit_mode": "source_only_fast_world_space_no_coordinate_transport",
    "audit_row_count": 97,
    "coordinate_invariance_pass": "NOT_RUN_REPAIR_HIGH_DIMENSION_COST",
    "corrected_snapshot_pass": true,
    "cross_operator_audit_pass": true,
    "error_count": 0,
    "errors": [],
    "expected_audit_row_count": 97,
    "full_coverage_pass": true,
    "operator_snapshot_source": "/Users/sunlay/Desktop/ood-representation-regularization/round3_redesign/cleanroom_rerun_1_3d_repair/results/source_only_bridge/irmv1_operator_snapshots.npz",
    "partial_reason": "Audits pass/fail on repaired source-only bridge, but CMNIST remains empirical and family-coupled; no full scientific PASS is claimed.",
    "verdict": "SHARP-OPTIMALITY-CMNIST-SOURCE-ONLY-PARTIAL"
  }
}
```

## Helps Diff

- Old stable help claims: `16`.
- Old help claims not reproduced/lost: `12`.
- Old help directions persisting only below 10% of old magnitude: `4`.
- Old help claims persisted under repaired comparison: `0`.

The old positive regularizer-response story is materially weakened, not cleanly confirmed: missing family counterparts and attenuated magnitudes mean the repaired evidence is insufficient to say whether `LOCAL_RESPONSE` failed independently or followed a contaminated 3C-to-Task3 path.

## Final Verdict

`RERUN-INCONCLUSIVE`
