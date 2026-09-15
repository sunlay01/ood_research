# Optimizer/capacity audit

- Verdict: **CONTINUOUS-AXIS-AUDIT**
- Convergence rows: 420; dense rows: 252
- Matched source-risk pairs: 150
- Mean matched `G_repr` difference (SGD - Adam): `0.007916238640171696`
- Mean absolute matched `G_repr` difference: `0.017271679624129146`
- Dense first-capacity-below-0.10 map: `{'sgd|alpha=0.0': {'first_dz_below_0.10': 6}, 'sgd|alpha=0.2': {'first_dz_below_0.10': 6}, 'sgd|alpha=0.4': {'first_dz_below_0.10': 16}, 'sgd|alpha=0.6': {'first_dz_below_0.10': None}, 'sgd|alpha=0.8': {'first_dz_below_0.10': 16}, 'sgd|alpha=1.0': {'first_dz_below_0.10': None}, 'adam|alpha=0.0': {'first_dz_below_0.10': 1}, 'adam|alpha=0.2': {'first_dz_below_0.10': 1}, 'adam|alpha=0.4': {'first_dz_below_0.10': 1}, 'adam|alpha=0.6': {'first_dz_below_0.10': 1}, 'adam|alpha=0.8': {'first_dz_below_0.10': 1}, 'adam|alpha=1.0': {'first_dz_below_0.10': 1}}`

## Acquisition/preference diagnostic

- `q_hat` rows: `420` across `60` trajectories
- Below-to-above `q_hat` crossings of pooled `rho_bar=0.90`: `25` (`0.4167` of trajectories)
- Correlation(`rho_bar-q_hat`, `shortcut_reliance`): `0.2396371730897138`
- Correlation(`rho_bar-q_hat`, `G_use`): `0.3101143339859971`

The q_hat and counterfactual shortcut metrics are descriptive diagnostics, not a theorem about neural representation dynamics. This report does not validate a phase-boundary theorem or a categorical failure taxonomy.
