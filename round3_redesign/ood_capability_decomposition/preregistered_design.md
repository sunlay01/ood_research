# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND Preregistered Design

task_id: `TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND`
git_head_before_run: `e230af7dc9e19086a9b129113d5a3686a0148b07`
config_sha256: `a1083f49401660ecbb1f1b19e75848ad24612750b34e5255131945ebbbbbe60d`
checkpoint_manifest: `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv`
seeds: `10,11,12,13,14`
methods: `ERM`, `IRMv1`
ridge: `0.001`
color_subspace_ranks: `0,1,2,4,8,16,32,64`

## Question

Run a first-round diagnostic decomposition of the corrected CPU-minimal ERM-vs-IRMv1 OOD gap into tested capabilities `C_coverage`, `C_separate`, and `C_select`.

## Inputs

- Corrected CPU-minimal config: `configs/task3_cmnist_cpu_minimal.json`.
- Corrected checkpoint manifest: `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv`.
- Corrected reconstructed checkpoints for ERM/IRMv1 seeds `10..14`.
- Corrected target counterfactual construction from `task3_cmnist_counterfactual_audit.probe`.

## A: Feature Coverage

Freeze each encoder. Fit a deterministic closed-form ridge probe on source frozen features with noisy source labels, and separately fit two-fold oracle clean ridge heads on color-balanced target counterfactual features using clean label `digit < 5`. The oracle rows are diagnostic and are not source-only evidence.

## B: Feature Separability

Estimate the color-response subspace from the uncentered second moment of `phi(x_G)-phi(x_R)` on the full target counterfactual set. For ranks `0,1,2,4,8,16,32,64`, project out top color directions and report shortcut suppression, task loss, clean oracle coverage, and red/green color-probe accuracy.

## C: Feature Selection

Discard the original head and train head-only `HEAD_ERM`, head-only `HEAD_IRMv1`, and diagnostic `ORACLE_CLEAN` heads from identical initialization for each frozen encoder. Source-trained heads use only CPU-minimal source features, source noisy labels, and the fixed batch schedule.

## Forbidden Scope

No new regularizer, no algorithm claim, no source-identifiability claim, no causal/additive decomposition claim, no theory validation, and no D/E execution.

## Decision Logic

Allowed verdicts are `FIRST-ROUND-CAPABILITY-ISOLATED`, `FIRST-ROUND-CAPABILITY-PARTIAL`, `FIRST-ROUND-CAPABILITY-INCONCLUSIVE`, and `FIRST-ROUND-AUDIT-INVALID`. A bottleneck can be isolated only if an oracle/intervention gap is at least `20pp` in at least `4/5` paired seeds and competing tested capabilities do not explain the same gap.
