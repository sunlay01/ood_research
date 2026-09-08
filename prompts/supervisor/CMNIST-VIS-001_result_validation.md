# CMNIST-VIS-001 Independent Result Validation

Act as a read-only empirical-result validator. Do not edit files, rerun
training, select a lambda using target performance, or assess paper novelty.

Read:

- `docs/experiments/CMNIST-VIS-001_preregistered.md`
- `artifacts/CMNIST-VIS-001-REV1/config.json`
- `artifacts/CMNIST-VIS-001-REV1/environment.json`
- `artifacts/CMNIST-VIS-001-REV1/metrics.csv`
- `artifacts/CMNIST-VIS-001-REV1/paired_metrics.csv`
- `artifacts/CMNIST-VIS-001-REV1/aggregate_summary.csv`
- the implementation and tests for `cmnist_feature_probe`

Facts from execution that must be checked rather than assumed:

- 21 trained models: 3 seeds, paired ERM, and six regularized settings;
- the first run and REV1 have exact equality on every shared numeric metric;
- targeted CMNIST tests pass; full repository tests have one unrelated existing
  numerical-root failure in `scalar_irmv1_zero_candidates` and 59 passes.

Validate:

1. paired ratios and accuracy deltas use the same-seed ERM;
2. the three-seed direction consistency and effect sizes;
3. whether latent retention, head use, and task collapse are distinguishable;
4. whether any claimed mechanism is unsupported by the metrics;
5. whether the observed result is a valid exploratory signal, partial signal,
   no signal, invalid result, or cannot be validated.

Return one of:

```text
VALID_SIGNAL
PARTIAL_SIGNAL
NO_SIGNAL
INVALID_RESULT
CANNOT_VALIDATE
```

Then give concise evidence, limitations, and the strongest conclusion the data
supports. Do not turn three seeds or a single CMNIST shift into a general OOD
claim.
