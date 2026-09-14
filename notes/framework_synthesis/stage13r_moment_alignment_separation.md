# Stage 13R separation from Moment Alignment

## Primary source inspected

Chen, Si, Zhang & Zhao, *Moment Alignment: Unifying Gradient and Hessian
Matching for Domain Generalization*, arXiv:2506.07378 (2025; UAI 2025).
The PDF and appendices were inspected directly. Its Definition 3 extends the
transfer measure to multiple sources and uses a center-of-mass source domain.
Proposition 3 assumes the target is a convex combination of sources. Theorem 1
under IRM bounds transfer by differences of parameter derivatives of orders
`2,...,N`; Theorem 3 removes IRM using bounded gradients, strong convexity and
optimality-gap terms. Section 4 unifies IRM, gradient matching, Hessian matching
and feature-moment duality; Section 5 proposes CMA.

## Exact difference in primitive and target

Moment Alignment primitive:

```text
D_theta^k L_mu(theta),   k = 1,2,...
```

and target is a transfer measure involving excess risks relative to source and
target optima.

ESF primitive:

```text
D_xi R(f,xi)[delta],
S_D(f;xi)=sup_{delta in D(xi)}D_xiR(f,xi)[delta],
```

and target is integrated risk motion along an admissible environment path. The
two are related only through additional mixed-derivative and path assumptions.

## Rigorous all-order separation

For `q(theta)=theta^2`, define

```text
R(theta,xi)=q(theta)+xi,
Rtilde(theta,xi)=q(theta)+2xi.
```

All parameter derivatives of every order agree for every source environment:
`D_theta^k R=D_theta^k Rtilde`. Any statistic made solely from those derivatives
is identical. Yet `D_xiR=1`, `D_xiRtilde=2`, and at `xi_T=1` the target risks
differ. This proves that all-order parameter alignment does not identify
environmental sensitivity.

## Interpretation

This is a separation theorem about information, not a novelty claim for
support functions, path integrals, or derivatives individually. ESF remains
subject to component-level overlap with DRO, local sensitivity analysis,
optimal-recovery/identified-set arguments, and Moment Alignment's broader
derivative framework. Its possible contribution is the explicit environment-
side risk-motion calculus plus cross-method consequences and limitations.
