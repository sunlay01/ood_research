# Novelty audit: source-exposure geometry

## Executive verdict

**Status: PROBE, not a novelty claim.** The idea has a plausible framework-level
contribution, but the central operator `C_S = m^{-1} sum_e delta_e tensor delta_e`
is standard covariance geometry. A publishable distinction must come from the
semantic role and a theorem that links this operator to multiple regularizers
and a target-risk query under explicit source-only information constraints.

The current evidence does not establish an exact prior-art duplicate of the full
composition. It also does not justify claiming that no such work exists: recent
search services were rate-limited, and repository absence is not negative
evidence.

## Closest overlaps already present in the project

| Existing line | What overlaps | What is not yet supplied |
|---|---|---|
| Gretton et al. 2012 / MMD | kernel mean embeddings and Hilbert-space distances | risk-functional representation, exposure-blind subspace, source-only target support |
| Krueger et al. 2021 / V-REx | risk-vector variance across environments | domain-embedding operator and a target-family support theorem |
| Sagawa et al. 2020 / GroupDRO | support function of a risk vector | operator-induced target geometry rather than observed-group simplex |
| Blanchard et al. 2011/2021 | domain-of-domains and source exposure/coverage | this covariance-operator/task-functional formulation |
| Shui et al. 2022 | representation-level invariance and contraction | exposure spectrum/nullspace and a common risk-functional state |
| Wang et al. 2026 Tri-Space | invariant/variant decomposition and localized discrepancy | RKHS exposure operator and optimizer probes |
| Lai & Wang 2024 | functional translation of IRMv1 gradients | proof that derivative probes are the same state as V-REx/MMD geometry |

These overlaps make the object-level novelty claim weak. The strongest defensible
claim is a **new composition/interface plus a master theorem**, pending proof.

## Exact mathematical risks

### Risk 1: `ell_f` may live in the wrong space

The loss on examples, the risk as a function of a domain law, and an RKHS element
are different objects. The equation ` <ell_f,C_S ell_f> ` is undefined unless a
domain embedding and a bounded representer `g_f` are specified. The candidate
must include a representability residual or reject the mapping.

### Risk 2: MMD does not control the whole operator by default

Average pairwise squared MMD controls `tr(C_S)` (up to the usual finite-sample
factor). It does not identify eigenvectors or the complete spectrum. Claiming
“MMD controls `C_S`” without a spectral regularizer is an overstatement.

### Risk 3: exposure is not invariance

`ker(C_S)` means no observed source variation, not a causal or label-invariant
direction. A target may move arbitrarily in that subspace. The target support must
contain an explicit nullspace/conditional ambiguity term.

### Risk 4: DRO requires a separately justified target set

An operator can parameterize `U_EX`, but source data do not prove that `P_T` lies
in that set or determine its radius. Target inclusion is an external assumption.

### Risk 5: optimizer methods need lifted state

IRMv1's classifier derivative and Fishr's per-example gradient covariance are not
functions of `C_S` in general. A valid extension would need a differentiable task
map `f -> g_f` and, for Fishr, a second lifted covariance operator. Otherwise the
framework should classify these methods as outside or translation-pending.

## Candidate contribution specification

- **Failure of baseline A:** source-risk variance, marginal MMD and finite-group
  support each ignore different unseen target directions and cannot state one
  exposure-aware ambiguity price.
- **Mechanism from B:** Hilbert covariance geometry plus a nullspace-aware support
  function, coupled to a domain-risk representer `g_f`.
- **New coupling rule:** define `U_EX(rho,kappa)` from the source exposure operator
  and prove one risk bound with a range term and an exposure-blind term; map each
  regularizer only through an explicit fidelity lemma.
- **Divergent prediction:** when two source sets have the same mean embedding and
  trace but different spectra/nullspaces, the framework predicts different
  robustness for spectrum-aware functionals, while vanilla MMD predicts the same
  trace-level penalty.
- **Closest-equivalent risk:** a repackaging of MMD covariance, V-REx variance,
  domain-of-domains concentration, or a standard DRO ellipsoid.
- **Cheapest falsification test:** finite-dimensional linear domains with exact
  embeddings. Construct equal-trace/different-spectrum source sets and test (i)
  the proposed support bound, (ii) V-REx equality, and (iii) whether any claimed
  IRMv1/Fishr map depends only on `C_S`.

## Score (0–2 each)

| Criterion | Score | Reason |
|---|---:|---|
| unmet failure addressed | 2 | exposes seen versus unseen source directions and a nullspace price |
| distinct mechanism | 1 | covariance geometry is standard; coupling may be distinct |
| separation from plain composition | 1 | not yet shown beyond writing existing objects in one notation |
| cheap discriminating test | 2 | finite-dimensional operator counterexamples are inexpensive |
| resource feasibility | 2 | theorem/probe can start in finite-dimensional RKHS or linear model |
| importance beyond one benchmark | 2 | target-family calibration and identifiability affect many DG methods |
| **total** | **10/12** | advance only to a theorem/probe, not a novelty verdict |

## Decision ledger

- **Budget spent:** repository evidence plus targeted 2024–2026 metadata checks;
  exact-overlap web search remains incomplete because Scholar/OpenAlex/S2 were
  rate-limited or CAPTCHA-protected.
- **Evidence IDs:** Gretton2012, Krueger2021, Sagawa2020, Blanchard2011/2021,
  Shui2022, Lai2024, Wang2026TriSpace, Zhao2019, Wang2024Lost.
- **Selected status:** `PROBE`.
- **Next artifact:** prove or refute the finite-dimensional master support bound,
  then test a held-out optimizer method. Do not call the result a new universal
  framework until an optimizer mapping survives.
