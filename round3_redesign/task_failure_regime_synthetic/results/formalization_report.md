# Preference/acquisition formalization audit

- Verdict: **ACQUISITION-SPEED-CONSISTENT; PREFERENCE-MODEL-EXACT**
- Exact acquisition-coupled switch: `rho > q_t`, where `q_t = 1 - sigma_c - (1 - 2 sigma_c) delta`
- Binary rows: 33 (including `8` pooled-mixture rows)
- Matched source-loss pairs: `156`
- Mean absolute matched `G_repr` gap: `0.022691251560282233`

## Exact population model

Let `C=Y eta_c`, `S=Y eta_s`, with `P(eta_c=1)=1-sigma_c`, `P(eta_s=1)=rho`, and conditional independence. If acquisition flips the core with probability `delta`, its effective reliability is `q_t = 1 - sigma_c - (1 - 2 sigma_c) delta`. The population log-odds are `q_t`-reliability core weight `log(q_t/(1-q_t))` plus shortcut weight `log(rho/(1-rho))`; therefore shortcut preference is exactly `rho > q_t`. Target BCE and the frozen balanced-head repair gain are evaluated by enumerating the four effective-noise states.

## Pooled source environments

For equal-weight source environments, the marginal shortcut reliability is `rho_bar=(rho_1+rho_2)/2`. With `(rho_1,rho_2)=(.95,.85)`, `rho_bar=.90` and `sigma_c=.20`, shortcut preference holds even at perfect acquisition (`q_t=.80`); with `(.65,.55)`, `rho_bar=.60`, preference switches as acquisition improves, at `delta_c=1/3` in this model. The high-family `delta=0` exact target repair gain is `1.0077` against target `rho_T=.10`.

## Acquisition audit

The trajectory audit treats source BCE as a progress coordinate. After matching source risk, the optimizer gap is small in the existing neural runs, which is consistent with a speed/progress explanation. The new `q_hat`/shortcut diagnostic is descriptive evidence about the proposed coupling, not a theorem about neural dynamics. The dense alpha/capacity sweep shows a capacity/difficulty trend, but not a single universal phase boundary.
