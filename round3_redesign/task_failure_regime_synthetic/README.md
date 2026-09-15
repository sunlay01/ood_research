# Synthetic failure-regime audit

Run a smoke test:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_failure_regime_synthetic --smoke
```

Run the preregistered factorial:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_failure_regime_synthetic
```

Outputs are written to `results/`. The final verdict is deliberately limited
to `REGIMES-SUPPORTED`, `REGIMES-NOT-DISCRETE`, or `REGIMES-NONOPERATIONAL`.

The runner writes raw rows (`factorial_rows.csv`), final-checkpoint cell
aggregates (`factorial_summary.csv`), checkpoint trajectories
(`checkpoint_summary.csv`), a machine-readable `summary.json`, and
`final_verdict.md`. The smoke equivalents use the `smoke_` prefix.

The final label is an audit gate, not a statistical theorem. A supported result
requires at least two threshold-separated diagnostic labels, a matching repair
intervention, and no optimizer-dominance warning. Otherwise the result is
reported as non-discrete; an absent or non-improving repair signal is reported
as non-operational. Thresholds are the preregistered descriptive values and
must not be treated as universal effect-size claims.
