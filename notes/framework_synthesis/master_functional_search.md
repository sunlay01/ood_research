# Master Functional Search: first candidate screen

## 1. Purpose

Stage 13R.1 rejected raw ESF as an independent master, but not the search for a
master object. This document starts a separate, bounded search at the level of
the entire environment-indexed risk functional

```text
R_P : F -> R,       f |-> R_P(f),
Delta_{P,Q} = R_Q - R_P.
```

The search is not a new regularizer design and does not enter Stage 14. Each
candidate is screened against the same requirements: scientific invariance,
algorithm independence, a target-risk theorem, exact/controlled translations,
and prior-art novelty.

## 2. Candidate matrix

| Candidate | Master object | First hard-gate result | Status |
|---|---|---|---|
| Transfer functional | `E_Q(f)-E_P(f)` | Moment Alignment already uses this excess-risk transfer target | `STOP: ALREADY-COVERED` |
| Risk-landscape quotient discrepancy | `||[R_P-R_Q]||_{X/<1>}` | Sup-norm quotient equals half the classical pairwise loss discrepancy; another norm needs an external measure on `F` | `REVISE` |
| Robust regret | `sup_{P in U}(R_P(f)-R_P*)` | Standard robust regret/minimax excess-risk object; target-family and Bayes-risk assumptions remain the same bottleneck | `CROWDED-INCREMENTAL` |
| Optimality-map instability | `diam{argmin_f R_P : P in U}` | Optimizer diameter alone does not control risk gaps or target risk | `STOP: NO-CERTIFICATE` |

The only candidate retained for a bounded follow-up is risk-landscape quotient
discrepancy, and only after its norm and target theorem are made non-arbitrary.
This is a search-stage `REVISE`, not an accepted master functional and not a
publication-level novelty judgment.

## 3. Risk-landscape quotient candidate

Let `X = R^F` be the space of bounded risk landscapes and quotient by the
environment-only constants

```text
R ~ R + c * 1_F.
```

This implements the Stage 13R.1 invariance principle: predictor-independent
environment difficulty is not a change in the relative learning problem.
However, a quotient vector space does not carry a canonical norm. The natural
choice `||.||_infty` gives the exact identity proved in
`master_functional_discrepancy_audit.md`:

```text
inf_c max_f |Delta(f)-c| = (max_f Delta(f)-min_f Delta(f))/2,
max_{f,h} |Delta(f)-Delta(h)| = max_f Delta(f)-min_f Delta(f).
```

Thus the quotient sup-norm is exactly one half of a loss-class pairwise
discrepancy (Mansour--Mohri--Rostamizadeh 2009; Ben-David et al. 2010). It is
not a new discrepancy theorem.

An `L2(nu)` quotient requires a declared probability measure `nu` over the
predictor class. That measure is an additional modeling object, may be
algorithm- or search-dependent, and changes the value of the discrepancy. A
future candidate may use such a measure only if it has an independent
scientific interpretation and yields a target-risk theorem not reducible to a
standard IPM/discrepancy bound.

## 4. Observation/projection map

If a risk-landscape object survives, the method map should be written as
observations of the same `Delta`, not as separate primitives:

```text
Delta_{P,Q}
  -> point evaluation          (V-REx risk vector)
  -> support over environments  (GroupDRO/MM-REx)
  -> argmin/optimality          (ideal IRM)
  -> parameter Taylor probes    (IRMv1/Moment Alignment)
  -> gradient-covariance probe  (Fishr)
  -> distribution witness bound (MMD/CORAL).
```

This diagram is a hypothesis only. A translation is accepted only with a fixed
upstream landscape, a fixed observation map, and a theorem controlling the
relevant target quantity.

## 5. Immediate stop reasons

- The quotient sup-norm branch is classical discrepancy after an exact algebraic
  reduction.
- Robust regret is a standard robust decision/DRO quantity and does not yet
  explain a new OOD regularizer mechanism.
- Optimizer-set diameter has no general implication for target risk: two risks
  can share the same optimizer while differing by an arbitrarily large
  predictor-independent or predictor-dependent scale away from the optimum.
- No candidate currently supplies a source-only, non-vacuous target-risk bound
  for heterogeneous regularizers without an explicit target-family assumption.

## 6. Next bounded gate

Do not design a new method. The only authorized continuation is a focused
discrepancy hard gate for the risk-landscape quotient:

1. declare a finite predictor class and a scientifically fixed landscape norm;
2. prove whether the norm is a classical discrepancy/IPM or genuinely different;
3. derive one target-risk theorem with all oracle/conditional terms exposed;
4. test whether V-REx, GroupDRO, MMD, and ideal IRM are observations of that
   same object without changing the norm or landscape state.

If this gate again reduces to classical discrepancy or requires method-specific
norms, terminate the candidate and preserve Stages 12/13 as the project's
validated local calculus.
