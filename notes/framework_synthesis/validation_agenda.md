# Validation agenda

## Priority 1: conditional witness theorem

Prove a bounded-loss population and finite-sample theorem of the form

`R_T(f) <= R_S(f) + D_G(P_T^Phi,P_S^Phi) + Delta_cond(f) + eps_est`

for a declared source-defined target family. Then prove an empirical MMD (and a
moment-restricted CORAL) penalty-to-radius lemma. Keep `Delta_cond` explicit and
test tightness with the binary channel-swap construction.

## Priority 2: restricted IRMv1 translation

Choose a low-dimensional smooth classifier family. Derive the empirical derivative
to population risk-surface/TV bound, including stationarity and optimization
errors. Check whether the resulting functional controls a mechanism or only a
local objective geometry. A failure that leaves the conditional residual maximal
is a useful negative result.

## Priority 3: Fishr falsification probe

In a finite linear model, construct two source worlds with matched per-example
gradient covariance but different target conditional risks. If possible, this
disproves covariance as a sufficient certificate for that model. If impossible
under an explicit restricted model, state the resulting covariance-to-risk lemma
and its assumptions; do not extrapolate to deep networks.

## Priority 4: target-family calibration

For finite-group, f-divergence, Wasserstein, anchor and meta-domain queries,
quantify the cost of target-family misspecification. Report when the radius makes
the certificate vacuous. This is deployment semantics, not a hyperparameter
ablation.

## Priority 5: estimation/computation separation

For each surviving local theorem, report three independent quantities:

- population approximation or mechanism mismatch;
- finite-sample confidence/complexity error;
- optimization, solver or relaxation error.

Do not use a single residual symbol for all three.

## Minimal decision gates

- **Advance:** a local `Omega -> B` theorem plus a non-vacuous target-family query.
- **Probe:** a formally distinct bridge with one cheap discriminating experiment.
- **Kill:** two admissible source-indistinguishable worlds with constant-separated
  target query, or a bridge that is always maximal/vacuous.
- **Do not claim novelty:** absence from the repository or a failed web query is
  not evidence that no later work exists.
