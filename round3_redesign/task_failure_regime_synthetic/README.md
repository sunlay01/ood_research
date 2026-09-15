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

The follow-up optimizer/capacity audit is run with:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_optimizer_capacity_audit
```

It matches SGD and Adam states by source BCE before comparing `G_repr`, then
sweeps continuous task difficulty `alpha` and capacities
`d_z in {1,2,3,4,6,8,16}`. Its current result is in
`results/optimizer_capacity_report.md`: the matched optimizer gap is small,
while the dense sweep shows a capacity/difficulty trend without supporting a
single phase-boundary theorem.

The exact binary preference calculation and acquisition-collapse audit are run
with:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_preference_acquisition_formalization
```

The binary model proves the declared-model switch `rho > 1 - sigma_c` and
computes exact target BCE and balanced-head repair gain. The acquisition audit
reuses the convergence trajectories and finds a mean absolute matched
`G_repr` gap of `0.0227`; this supports a training-progress explanation but is
not a neural-dynamics theorem.
