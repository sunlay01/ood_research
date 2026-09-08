# Benchmark Definition

Let `Y ~ N(0,1)`. The observed vector is

```text
X = (1, C, S_1, ..., S_q, N_1, ..., N_K, U, Z)
```

with `C = Y + sigma_C eps_C`, `S_j = rho_j Y + sigma_j eps_j`, independent
noise coordinates `N_i`, an emergent coordinate `U = gamma_U Y + sigma_U eps_U`,
and a stable predictive coordinate `Z = gamma_Z Y + sigma_Z eps_Z`.

The source optimum is computed from the mixture population moments. A source
relation-exposed design varies `rho_j` around the registered base value. The
reference environment matches the resulting source second moments before
isolated target probes are formed.
