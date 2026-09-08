# Benchmark Design

The population state is generated from
\[
X=(1,C,S_1,S_2,N_1,\ldots,N_K,U),\qquad Y\sim N(0,1).
\]

`C` is a noisy predictive coordinate, `S_1` and `S_2` are separated shortcut
coordinates, `N_i` are irrelevant nuisance coordinates, and `U` is an
emergent target-relevant coordinate. The Gaussian loading construction keeps
all covariance matrices positive semidefinite and makes risk moments exact.

Each shortcut has relation, mean and variance interventions at magnitudes
`0.25`, `0.5`, and `1.0`, with signed relation/mean pairs. Main discovery
uses only pure probes. Held-out mixed probes are formed by adding moment
deltas, so exact first-order response additivity can be tested independently
of nonlinear environment parameterization.

The sweep uses nuisance dimensions `0,4,16,64,256`, one and two shortcuts,
and redundant-copy groups `1,4,16,64`. Coordinate robustness uses 20
invertible predictor reparameterizations.
