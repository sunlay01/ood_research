# Mechanism primitives from exact method mathematics

Status: `OUR-INTERPRETATION`, grounded in the method cards and the bridge ledger.
The primitives are interfaces with explicit semantics, not a relabelled family taxonomy.

## P1. Source-risk vector and support functional

For a predictor `f`, retain

`r_S(f) = (R_1(f), ..., R_m(f))`.

The legal operations are a support functional `sigma_U(r)=sup_{q in U} q^T r`,
a variance/dispersion functional, or a domain-meta-law functional
`g_f(P)=R_P(f)`. GroupDRO is native to the support function over the observed
simplex; V-REx is native to a quadratic dispersion of the same vector; ERM is
the uniform linear functional. The mapping from a finite empirical vector to
population `r_S` is `PROVED` by uniform convergence under bounded loss.

**Retains:** predictive performance by observed environment and finite-group
robustness. **Discards:** feature mechanisms, conditional label response, and
unseen-domain tail structure. **Target claim:** only a declared finite mixture,
uncertainty set, or meta-law query. The target-family membership is external.

## P2. Distributional witness and conditional residual

Let `Phi` be a representation and `P_e^Phi = Phi#P_e`. A witness discrepancy is

`D_G(P,Q) = sup_{g in G} |E_P g - E_Q g|`.

MMD and domain-discriminator objectives instantiate `D_G`; CORAL instantiates a
finite second-moment witness. A sound transfer decomposition must retain a
conditional/joint residual, for example

`R_Q(f)-R_P(f) = Delta_X + Delta_{Y|X}`,

or a standard joint-error term. The mapping from an empirical MMD/CORAL statistic
to a population witness radius is `PROVED` under bounded kernels/features and
complexity control. The mapping from marginal witness equality to conditional
label equality is `KNOWN FALSE` without extra assumptions (Zhao2019).

**Retains:** the selected observable distribution geometry. **Discards:** any
conditional information outside the witness class. **Target claim:** a transfer
bound when the target marginal is in the declared radius family and the residual
is bounded.

## P3. Stable conditional mechanism / intervention response

Represent a world by an SCM or conditional kernel family

`W_mech = (G, {K_e(y|x_S)}, {I_e})`,

with a stable kernel `K` over an admissible intervention family. An invariant
representation satisfies `K_e = K` on the covered environments, or equivalently
`Y _||_ E | Phi(X)` in the relevant model. ICP and ideal IRM map here natively
under their structural semantics; anchor regression maps to a linear shift
response. The mapping from finite source tests or gradients to `K` is only
`PROVED-UNDER-RESTRICTIONS`.

**Retains:** causal/conditional response under interventions. **Discards:**
mechanism distinctions outside the graph and intervention family. **Target
claim:** transfer over the covered family. **Failure:** non-identifiability,
faithfulness failure, weak heterogeneity, or target intervention outside family.

## P4. Objective-surface / derivative statistic

For a fixed `Phi`, define the risk surface

`F_e(w) = R_e(w o Phi)`.

IRMv1 observes `grad_w F_e(w_0)`; Fishr observes covariance of per-example
gradients; MLDG observes an update-map composition. Lai2024 gives a restricted
translation from the derivative penalty to a TV functional of `F_e`.

**Retains:** local geometry of the chosen objective and optimizer statistic.
**Discards:** any world distinction not identifiable from the surface/derivative
law. A derivative statistic is not a conditional mechanism by itself. Target
claims require a separate local translation theorem. Fishr currently has no
general one in the evidence base.

## P5. Complexity / information control

Retain a class, posterior, stability coefficient, or information radius, e.g.
`KL(Q||P)`, mutual information, spectral norm, or replace-one stability.

This primitive controls estimation or learner sensitivity. It is orthogonal to
the shift semantics: it can tighten a source or bridge theorem, but cannot define
which target laws are admissible. PAC-Bayes and norm/stability methods map here
natively; OOD use requires P1-P3 or an explicit target-family theorem.

## Shared structure and irreducibility

The strongest justified compression is a *certificate interface*:

`Omega_j  ->  B_j  ->  (target family, query)  ->  R_T`.

`B_j` must have one of the semantic types above. P1 and P2 share risk/distribution
transfer algebra but differ in what information is retained. P3 is not a witness
discrepancy. P4 is a translation layer, not a target certificate. P5 is a
complexity layer, not shift identification. These distinctions are irreducible
under the current evidence.
