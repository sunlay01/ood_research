# Preference/acquisition formalization audit

- Verdict: **ACQUISITION-SPEED-CONSISTENT; PREFERENCE-MODEL-EXACT**
- Exact binary switch: `rho > 1 - sigma_c`
- Binary rows: 25
- Matched source-loss pairs: `156`
- Mean absolute matched `G_repr` gap: `0.022691251560282233`
- Lean slice: `PreferenceAcquisition/Basic.lean` passes the odds/preference
  equivalence check with local Lean 4.33.1 + cached mathlib artifacts.

## Exact population model

Let `C=Y eta_c`, `S=Y eta_s`, with `P(eta_c=1)=1-sigma_c`, `P(eta_s=1)=rho`, and conditional independence. The population log-odds are `C log((1-sigma_c)/sigma_c) + S log(rho/(1-rho))`; therefore the shortcut coefficient exceeds the core coefficient exactly when `rho > 1-sigma_c`. Target BCE and the frozen balanced-head repair gain are evaluated by enumerating the four `(eta_c, eta_s)` states.

## Acquisition audit

The trajectory audit treats source BCE as a progress coordinate. After matching source risk, the optimizer gap is small in the existing neural runs, which is consistent with a speed/progress explanation. The dense alpha/capacity sweep shows a capacity/difficulty trend, but not a single universal phase boundary. Neither result is a theorem about neural dynamics.
