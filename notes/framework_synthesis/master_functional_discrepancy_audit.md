# Risk-landscape quotient: discrepancy hard gate

## Exact finite-class calculation

For a finite predictor class `F={f_1,...,f_n}`, write

```text
d_i = R_Q(f_i) - R_P(f_i).
```

The quotient by constants under the sup norm is

```text
||[d]||_{infty/1} = inf_{c in R} max_i |d_i-c|.
```

Let `d_max=max_i d_i` and `d_min=min_i d_i`. Every center `c` satisfies

```text
max_i |d_i-c| >= max(|d_max-c|, |d_min-c|)
                       >= (d_max-d_min)/2,
```

and equality is attained at `c=(d_max+d_min)/2`. Therefore

```text
||[d]||_{infty/1} = (d_max-d_min)/2.
```

The pairwise loss-class discrepancy is

```text
disc_pair(P,Q) = max_{f,h in F} |(R_Q(f)-R_P(f))-(R_Q(h)-R_P(h))|
                = d_max-d_min.
```

Consequently

```text
||[R_Q-R_P]||_{infty/1} = disc_pair(P,Q)/2.
```

This is an exact equivalence, not merely a loose neighboring bound. It places
the natural sup-quotient branch inside the classical loss-class discrepancy
framework of Mansour, Mohri & Rostamizadeh (2009) and Ben-David et al. (2010).
More precisely, their Definition 4 is
`disc_L(Q1,Q2)=sup_{h,h'}|L_{Q1}(h,h')-L_{Q2}(h,h')|`; when the landscape
coordinates are the losses of predictors against a fixed labeling/reference,
the pairwise landscape difference above is its direct fixed-reference
specialization. Thus the quotient does not evade the classical discrepancy
hard gate by changing notation.

## Why another norm is not free

An `L2` quotient requires a probability measure `nu` over predictors, for
example

```text
inf_c ( integral (Delta(f)-c)^2 dnu(f) )^(1/2).
```

Changing `nu` changes the value and can change which environments appear close.
Unless `nu` is fixed independently of the algorithm, representation, and target
data, it is a hidden method-specific choice. Even with fixed `nu`, the resulting
quantity is a standard function-class IPM/mean-square discrepancy unless a new
target theorem demonstrates otherwise.

## Target-risk implication

The quotient discrepancy controls differences of predictors across environments,
not the absolute target risk. A target bound still needs a source-risk term and
an oracle/conditional term, exactly as in classical DA:

```text
R_Q(f) <= R_P(f) + discrepancy(P,Q) + joint/conditional residual.
```

Quotienting removes additive environment difficulty, but it does not remove the
identifiability barrier for target conditional mechanisms. In particular, a
source-only DG claim still needs a declared admissible target family or a
source-identifiable mechanism assumption.

## Prior-art evidence

- Mansour, Mohri & Rostamizadeh (2009), arXiv:0902.3430, defines loss-class
  discrepancy and derives target-risk bounds/algorithms.
- Ben-David et al. (2010), *A theory of learning from different domains*,
  defines the `H Delta H` discrepancy and the source-risk + discrepancy + joint
  error bound.
- Gretton et al. (2012), JMLR 13, supplies the RKHS/IPM/MMD norm analogue.
- Lai & Wang (ICML 2024) give a functional/TV interpretation of IRMv1, showing
  that lifting a parameter penalty to a function-space object is already an
  established route for a specific method.
- Wang et al. (JMLR 2026) prove a direct-sum latent representation and localized
  discrepancy target-risk theorem, but do not use the exact quotient landscape
  object proposed here.

The repository's 2023--2026 ledger also records Lai & Wang's ICML 2024 TV
functional interpretation of IRMv1 and Wang et al.'s JMLR 2026 Tri-Space
localized discrepancy. Those are important near-neighbors: they show that
lifting an algorithm statistic to a functional or localized discrepancy can be
theorem-bearing, but neither establishes the proposed full observation map
from one quotient landscape to V-REx, GroupDRO, MMD, and IRM.

Classification for the exact sup-quotient is `EQUIVALENT-AFTER-
REPARAMETERIZATION`; the full end-to-end multi-regularizer organization is
`NO-EXACT-OVERLAP-FOUND` in the checked corpus, which is negative evidence only.

## Gate result

`REVISE-RISK-LANDSCAPE-CANDIDATE`.

The candidate is not killed at the search level, but its most natural norm is
already classical discrepancy. A continuation must justify a non-arbitrary
landscape geometry and prove a cross-method theorem before any novelty claim.
This is a bounded search-stage result, not an accepted project master.
