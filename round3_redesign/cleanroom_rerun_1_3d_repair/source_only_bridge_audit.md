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

