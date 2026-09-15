# Synthetic failure-regime audit

Run a smoke test:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_failure_regime_synthetic --smoke
```

Run the preregistered factorial:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_failure_regime_synthetic
```

Outputs are written to `results/`. The primary result is a continuous diagnostic
vector, not a categorical regime assignment. The current verdict is
`TWO-AXIS-STRUCTURE-SUPPORTED; CONTAMINATION-UNRESOLVED`.

The runner writes raw rows (`factorial_rows.csv`), final-checkpoint cell
aggregates (`factorial_summary.csv`), checkpoint trajectories
(`checkpoint_summary.csv`), a machine-readable `summary.json`, and
`final_verdict.md`. The smoke equivalents use the `smoke_` prefix.

The current analysis tests seed-level factorial interaction contrasts for
`G_use` and `G_repr`, plus an independent head-repair check. `C_prob` is the
scale-invariant probability-space counterfactual sensitivity; raw-logit
`C_pred` is retained only for comparison. Categorical labels and contamination
causality are deliberately not used as acceptance criteria.
