# Concrete problem instances

## P1 — V-REx to expected future-domain risk

- **Algorithmic quantity:** `Rhat_bar(f) + lambda Var_e[Rhat_e(f)]`.
- **Target:** `Q_Pi(f)=E_{P~Pi} R_P(f)` for a fresh exchangeable domain.
- **Source information:** labeled samples from independent source domains.
- **Admissible semantics:** source domains and target are draws from the same meta-law `Pi`.
- **Known bridge:** domain-risk function `g_f(P)=R_P(f)` plus two-level concentration.
- **Missing step:** a non-vacuous theorem showing whether low source risk variance improves a domain-tail or quantile term under a named `Pi`; variance alone does not imply conditional invariance.

## P2 — MMD / representation matching to transfer risk

- **Algorithmic quantity:** source representation MMD (or CORAL moment discrepancy).
- **Target:** target risk or transfer gap for a declared target family.
- **Source information:** labeled source samples; no target samples.
- **Admissible semantics:** target marginal lies in a source-defined witness-radius family, with an explicit conditional-label budget.
- **Known bridge:** RKHS/IPM duality and conditional change-of-measure decomposition.
- **Missing step:** prove empirical penalty to population radius, and separately control the conditional remainder; Zhao's counterexample blocks a marginal-only claim.

## P3 — IRMv1 to intervention-family risk

- **Algorithmic quantity:** empirical classifier-gradient penalty at the reference classifier.
- **Target:** risk over a specified SCM/intervention family, or an expected target under `Pi`.
- **Source information:** labeled source environments and the empirical derivative statistic.
- **Admissible semantics:** shared mechanism plus rank/heterogeneity, positivity and intervention coverage.
- **Known bridge:** ideal IRM/rank arguments; Lai's functional translation in restricted smooth settings.
- **Missing step:** empirical derivative to population invariant property, including stationarity and optimization error; no general deep bridge is established.

## P4 — Fishr to target risk

- **Algorithmic quantity:** covariance matching of per-example classifier gradients.
- **Target:** only a declared target family; arbitrary target is not admissible.
- **Source information:** source gradient samples and labels.
- **Admissible semantics:** would require a restricted model in which derivative covariance controls a conditional or risk-shift object.
- **Known bridge:** motivational optimizer-statistic interpretation only.
- **Missing step:** covariance-to-property theorem. This is a bottleneck candidate, not an accepted premise.

## P5 — GroupDRO / DRO control comparison

- **Algorithmic quantity:** `max_e Rhat_e` or robust empirical objective over a declared divergence/transport set.
- **Target:** `sup_{Q in U} R_Q(f)`.
- **Source information:** source labeled samples and predeclared `U`.
- **Admissible semantics:** finite source mixtures, f-divergence balls or Wasserstein balls with justified support/radius.
- **Known bridge:** support-function or primal-dual robust-risk theorem plus uniform convergence.
- **Missing step:** justify that the chosen `U` contains the deployment target; this is a scientific coverage question, not an optimization detail.
