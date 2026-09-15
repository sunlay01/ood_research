# Optimizer/capacity audit

- Verdict: **CONTINUOUS-AXIS-AUDIT**
- Convergence rows: 420; dense rows: 252
- Matched source-risk pairs: 150
- Mean matched `G_repr` difference (SGD - Adam): `0.007916238640171696`
- Mean absolute matched `G_repr` difference: `0.017271679624129146`
- Dense sweep: SGD `G_repr` falls below 0.10 at `d_z=6` for alpha 0.0/0.2 and only at `d_z=16` for alpha 0.4/0.8; it remains above 0.10 through `d_z=16` for alpha 0.6/1.0. Adam is below 0.10 from `d_z=1` for every alpha.

This report tests optimization-speed and capacity confounds. The matched result
does not support an optimizer-implicit-bias claim under this protocol; the dense
result shows a capacity/difficulty trend but not a single universal phase
boundary. It does not validate a phase-boundary theorem or a categorical failure
taxonomy.
