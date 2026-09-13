# Phase III — Reuse test

The test is applied in the order required by the problem-first protocol. A
route is not upgraded merely because a new tuple of symbols would be convenient.

| Problem instance | Route 1: existing framework unchanged | Route 2: local algorithm theorem | Route 3: representation change | Route selected |
|---|---|---|---|---|
| P1 V-REx -> fresh-domain risk | Yes: domain-risk function `g_f(P)` and meta-law concentration already state the target | Needed only to quantify whether risk variance improves a tail/quantile term under a named `Pi` | No evidence that a new representation preserves more target information | 1 + 2 |
| P2 MMD/CORAL -> target risk | Yes for an IPM/conditional change-of-measure bound with a joint-error or conditional remainder | Needed to turn the empirical source discrepancy into a population radius; CORAL also needs a moment-to-IPM restriction | No: marginal representation is the intended object, and the missing label information is an explicit remainder | 1 + 2 |
| P3 IRMv1 -> intervention-family risk | Yes for ideal IRM/ICP/SCM objects, not for the exact optimizer dynamics | Needed for empirical classifier-gradient to population stationarity/conditional property, with optimization error | Only if a chosen theorem query cannot be expressed by the existing conditional object; current evidence does not show this | 1 + 2 |
| P4 Fishr -> target risk | Yes only as a diagnostic/functional statistic, with no target theorem yet | A covariance-to-conditional or covariance-to-risk-shift lemma is required | A new statistic would be justified only by a proof obstruction; none is established | 1, then 2 as an open probe |
| P5 GroupDRO/DRO -> robust target set | Yes: support-function or DRO uncertainty-set theory already matches the target quantity | Needed for empirical uniform convergence and target-set misspecification accounting, not a new architecture | No | 1 + standard estimation lemma |

## Result

All five instances can be stated with an existing semantic object. The only
new work forced by the problem is local: empirical-to-population control,
algorithm-to-functional translation, target-family coverage, or an explicit
impossibility remainder. No instance currently passes the Route 4 or Route 5
test. The working decision therefore remains `YES-WITH-LOCAL-THEOREM`.
