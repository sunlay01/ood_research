# IRM / IRMv1 theory audit

## Exact objects in the proposal

Arjovsky et al. (2019, Eq. 2) define ideal IRM as finding a representation `Phi` and classifier `w` such that `w` is simultaneously optimal in every environment. Their practical surrogate (Eq. 3) fixes a scale (`w=1` in the scalar version) and penalizes `sum_e ||grad_w R_e(w o Phi)|_{w=1}||^2`. The derivative is with respect to the classifier variable `w`, not all representation/network parameters. It is a first-order stationarity surrogate, not a Hessian penalty.

## What reviewed theory actually analyzes

* The original paper's formal principle is population-level and uses a constrained argmin; experiments optimize IRMv1. Its theorem-like claims are restricted toy/linear constructions, not a finite-sample target-risk theorem for arbitrary deep IRMv1 training.
* Rosenfeld et al. (2021, “The Risks of Invariant Risk Minimization”) construct linear counterexamples showing that IRM can prefer a noncausal or non-invariant solution under finite environments and finite penalty. This is a failure/lower-bound analysis of the idealized objective, not a generalization guarantee for SGD on IRMv1.
* Kamath et al. (2021, “Does Invariant Risk Minimization Capture Causal Variables?”) show that even population IRM can fail to identify causal variables without additional assumptions. The proof studies invariance constraints and linear predictors, not deep optimizer dynamics.
* Ahuja et al. (2021) and related invariant-learning theory use algebraic invariance/optimality conditions, restricted linear models, or identifiability assumptions. The gradient penalty is either absent from the theorem or represented through the corresponding population constraint.

## Three theory treatments of gradient-based IRM

The reviewed literature now separates three distinct moves:

1. **Direct gradient-penalty analysis:** work with `||∇_w R_e||²` itself, requiring a derivative class, smoothness, and a stationary/optimization statement. This is uncommon for deep finite-sample DG.
2. **Population-constraint abstraction:** replace the penalty by simultaneous optimality/invariance and analyze restricted linear or structural models (Arjovsky et al.; Rosenfeld et al.; Kamath et al.).
3. **Objective-to-functional translation:** Lai & Wang (2024, PMLR 235:25913--25935, Theorems 3.1--3.11) identify the IRMv1 gradient norm with a TV-`ell_2` variation of risk as a function of the classifier variable, then study TV-`ell_1` and minimax variants under functional regularity and environment-family conditions. This preserves a mathematical trace of the gradient object, but it is not an equivalence between the full deep training dynamics and a standard DG bound.

The third route is a representation-translation example, not a broad cross-method unification: it captures IRM/TV variants, while GroupDRO, Fishr, and generic Wasserstein objectives are outside the stated functional unless separately mapped.

## Why the gradient penalty rarely appears in a clean bound

1. The penalty is a derivative of an *empirical risk*, so concentration must control a derivative class (and often its Lipschitz/smoothness envelope) jointly with the predictor class.
2. A small gradient norm says stationarity for a chosen classifier parameter; it does not by itself imply equality of conditionals or low target risk.
3. Finite optimization leaves a stationarity/optimization-error term. Deep nonconvex parameterizations make this term and representation identifiability difficult to state distribution-free.
4. Consequently, tractable papers move up one level: analyze a population invariant predictor, a linear-Gaussian model, or an arbitrary `f in F` satisfying a constraint, then add approximation/optimization error if needed.

## Evidence-bounded conclusion

In the sources reviewed here, no broadly applicable finite-sample source-only target-risk theorem analyzes the exact deep IRMv1 gradient-penalty training dynamics without restrictive model, smoothness, optimization, and environment-identifiability assumptions. Lai--Wang provides an important fourth option beyond “analyze a constraint”: translate the gradient penalty into a variational functional and state OOD conditions. This is narrower and defensible than saying “IRM has no theory”: there is substantial theory for ideal IRM, restricted linear cases, consistency/identifiability, impossibility, and functional translations.

## Audit checklist for future papers

For every claimed IRM theorem ask: (i) is the objective ideal IRM or Eq. 3 IRMv1? (ii) derivative variable and order? (iii) population or empirical risk? (iv) model class? (v) exact optimizer or stationary point? (vi) target family and source-only status? (vii) where is optimization/approximation error? These questions prevent silently treating a gradient surrogate as a target-risk representation.
