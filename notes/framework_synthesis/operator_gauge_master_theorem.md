# Operator/gauge master theorem

Status: `CANDIDATE THEOREM / PROBE`. The result below is elementary Hilbert-space
duality, but it gives the common mathematical interface that a regularizer must
instantiate. It does not by itself establish novelty or identify a target
radius from source data.

## Theorem (PSD operator with an exposure-blind residual)

Let `H` be a finite-dimensional Hilbert space and let `A >= 0` be self-adjoint.
For any `g, delta in H`, write

```
delta_parallel = Pi_range(A) delta,
delta_perp     = Pi_ker(A) delta.
```

Then

```
<g, delta>
 <= ||A^{1/2} g|| ||A^{dagger/2} delta_parallel||
    + ||Pi_ker(A) g|| ||delta_perp||.                 (1)
```

The first term is the polar pairing on the exposed subspace. If a target class
satisfies

```
||A^{dagger/2} delta_parallel|| <= rho,
||delta_perp|| <= kappa,
```

then

```
sup_{delta in U_A(rho,kappa)} <g,delta>
 <= rho ||A^{1/2} g|| + kappa ||Pi_ker(A) g||.          (2)
```

For an affine risk representation
`R_P(f)=b_f+<g_f,Psi(P)>+eta_f(P)` with
`|eta_f(P)| <= epsilon_repr`, (2) implies

```
R_T(f) <= R_bar_S(f)
          + rho ||A^{1/2} g_f||
          + kappa ||Pi_ker(A) g_f||
          + 2 epsilon_repr.                           (3)
```

## Proof

Because `A` is self-adjoint positive semidefinite, its range and kernel are
orthogonal. Thus

```
<g,delta> = <Pi_range(A)g, delta_parallel>
            + <Pi_ker(A)g, delta_perp>.
```

On `range(A)`, diagonalize `A` with positive eigenvalues. Coordinatewise,
`<g,delta_parallel> = <A^{1/2}g, A^{dagger/2}delta_parallel>`, so ordinary
Cauchy-Schwarz bounds the first term by the product in (1). Cauchy-Schwarz in
the kernel gives the second term. Taking the supremum under the two radius
constraints gives (2). Substituting
`R_T-R_bar_S = <g_f,delta_T> + eta_f(P_T)-m^{-1}sum_e eta_f(P_e)` and applying
the residual bound gives (3). Equality in (1) is attained on each component
when the shift is aligned with the corresponding representer, so the two-term
decomposition is tight for the declared class.

## Convex-gauge form

For an arbitrary convex gauge `J`, define its polar

```
J^circ(delta) = sup { <g,delta> : J(g) <= 1 }.
```

Then generalized Holder gives `<g,delta> <= J(g) J^circ(delta)` whenever the
polar is finite. A degenerate gauge has an unbounded polar outside its effective
dual domain; the PSD formula (1) makes that missing information explicit as a
kernel residual instead of silently assigning it a finite price.

## Interpretation for the research program

The theorem has one common bound, while methods may differ only in the
sensitivity gauge `J` and in how a target-family assumption calibrates its polar
radius. The current state is therefore:

```
regularizer -> sensitivity gauge J
target family -> coverage radius J^circ(delta_T)
source blindness -> explicit nullspace residual
```

V-REx corresponds to `J(g)=||C_S^{1/2}g||` under the affine risk assumption.
An exposure-normalized penalty corresponds to
`J(g)=||Pi_range(C_S)g||`, but then its polar radius is not the same as the
V-REx radius and must be calibrated separately. MMD and DRO are not mapped by
notation alone: each still needs a theorem connecting its observed objective to
`J`, the target radius, and the conditional/representation residual.
