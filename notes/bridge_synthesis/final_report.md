# Bridge extraction round 1 — final report

## 1. What existing theories actually do

Discrepancy theories use a witness seminorm and pay target discrepancy plus a joint-error term. Risk-level theories use a risk vector, variance, or simplex support and pay observed-group/coverage limitations. Causal theories use conditional mechanisms, SCM interventions, or rank/diversity to identify a stable predictor. DRO theories define a target uncertainty set and use support-function/transport duality. Variational theories translate a parameterized penalty into a population functional; this is not the same as proving target-risk control. Newer latent/information/calibration theories add decomposition, information-radius, or density-ratio objects for specific assumptions.

## 2. Recurring versus different objects

Recurring proof roles are: distinguishability, target-family uncertainty, mechanism identification, contraction/localization, algorithmic translation, estimation, and irreducible ambiguity. IPMs and support functions are superficially similar dual norms but encode different things: a witness class versus a set of admissible targets. Rank/identifiability defects are not discrepancies, and derivative statistics are not distributional operators without a translation theorem.

## 3. Proved versus assumed bridges

Proved or conditionally proved: HΔH/loss-class transfer, RKHS duality, finite-group support bounds, DRO duality, domain-level concentration, SCM/ICP identification under assumptions, Dobrushin contraction under invariance, and restricted rank arguments. Assumed or translated: V-REx extrapolation, ideal IRM to causal mechanism, CORAL/DANN to target risk, IRMv1-to-TV for restricted smooth settings, Tri-Space latent decomposition. Motivational only: generic Fishr covariance-to-invariance and unrestricted deep MLDG/optimizer bridges.

## 4. Reusable target-risk patterns

The most reusable outer patterns are (i) discrepancy transfer with an oracle residual, (ii) uncertainty support over a declared family, (iii) two-level meta-domain concentration, and (iv) conditional decomposition plus contraction. Algorithms can share these outer theorems only after supplying their own Omega-to-bridge lemma.

## 5. Generalization test

A new conditional mean-embedding regularizer can reuse the conditional decomposition if it proves empirical penalty to population conditional-radius control. This demonstrates useful modularity without implying universal unification.

## 6. Strongest justified abstraction

The evidence supports **SMALL-FAMILY-OF-BRIDGES**, not ONE-COMMON-BRIDGE. The next framework, if built, must be generated from these proof-role objects and must retain target-family coverage, conditional mismatch, oracle/joint error, approximation, complexity and optimization terms.

## 7. Next theorem and kill counterexample

Attempt a bounded-loss marginal-plus-conditional witness theorem with explicit radii and source complexity, then prove or refute one regularizer-to-radius lemma in a restricted model. Kill the abstraction if two admissible worlds have identical retained bridge objects and regularizer values but constant-separated target risks, or if every conditional radius is necessarily maximal.

Final status: **SMALL-BRIDGE-FAMILY-IDENTIFIED**.
