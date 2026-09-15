# Optimizer/capacity audit

- Verdict: **SMOKE-ONLY**
- Convergence rows: 6; dense rows: 8
- Matched source-risk pairs: 1
- Mean matched `G_repr` difference (SGD - Adam): `0.0`
- Mean absolute matched `G_repr` difference: `0.0`
- Dense first-capacity-below-0.10 map: `{'sgd|alpha=0.0': {'first_dz_below_0.10': None}, 'sgd|alpha=0.2': {'first_dz_below_0.10': None}, 'sgd|alpha=0.4': {'first_dz_below_0.10': None}, 'sgd|alpha=0.6': {'first_dz_below_0.10': None}, 'sgd|alpha=0.8': {'first_dz_below_0.10': None}, 'sgd|alpha=1.0': {'first_dz_below_0.10': None}, 'adam|alpha=0.0': {'first_dz_below_0.10': 1}, 'adam|alpha=0.2': {'first_dz_below_0.10': 1}, 'adam|alpha=0.4': {'first_dz_below_0.10': None}, 'adam|alpha=0.6': {'first_dz_below_0.10': None}, 'adam|alpha=0.8': {'first_dz_below_0.10': None}, 'adam|alpha=1.0': {'first_dz_below_0.10': None}}`

## Acquisition/preference diagnostic

- `q_hat` rows: `6` across `2` trajectories
- Below-to-above `q_hat` crossings of pooled `rho_bar=0.90`: `1` (`0.5000` of trajectories)
- Correlation(`rho_bar-q_hat`, `shortcut_reliance`): `-0.42374752967242807`
- Correlation(`rho_bar-q_hat`, `G_use`): `-0.268906211799246`

The q_hat and counterfactual shortcut metrics are descriptive diagnostics, not a theorem about neural representation dynamics. This report does not validate a phase-boundary theorem or a categorical failure taxonomy.
