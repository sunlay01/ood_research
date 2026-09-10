# OOD Capability Decomposition Report

## Question

Can the corrected CPU-minimal ERM-vs-IRMv1 OOD gap be separated into feature coverage, feature separability, and head-selection bottlenecks?

## Result

- verdict: `FIRST-ROUND-CAPABILITY-PARTIAL`
- dominant_bottleneck: `mixed`
- mean IRMv1-ERM target gap in loaded checkpoints: `0.559340`
- coverage oracle gap on ERM encoders: `0.804470`
- best separability gain on ERM encoders: `0.533780`
- selection oracle gap on ERM encoders: `0.297960`

## Row Counts

- feature_coverage rows: `10`
- separability_curve rows: `80`
- head_selection rows: `30`
- paired_capability_summary rows: `5`

## Isolation Votes

- coverage: `5/5`
- separability: `5/5`
- selection: `3/5`

## Interpretation

This report is diagnostic only. It makes no new regularizer claim, no algorithm claim, no source-identifiability claim, no causal/additive decomposition claim, no theory-validation claim, and no D/E execution claim.

## Paired Seed Snapshot

| seed | IRMv1-ERM target | coverage vote | separability vote | selection vote |
|---:|---:|:---:|:---:|:---:|
| 10 | 0.547600 | True | True | False |
| 11 | 0.554100 | True | True | True |
| 12 | 0.568200 | True | True | True |
| 13 | 0.573200 | True | True | True |
| 14 | 0.553600 | True | True | False |
