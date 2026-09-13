# A minimal methodology for regularization-to-OOD bounds

Status: `PROBE / ADVANCE-TO-MAPPING`. This note gives a finite-dimensional
population theorem and two falsification tests. It is not a universal theorem
for all regularizers or all domain shifts.

## 1. Minimal population model

Let the declared domain representation be a Hilbert space `H` (in the probe,
`H = R^d`) and let `mu_e = Psi(P_e)`. For `m` source domains define

```
mu_bar = m^{-1} sum_e mu_e,
delta_e = mu_e - mu_bar,
C_S = m^{-1} sum_e delta_e \otimes delta_e.
```

For a fixed predictor `f`, assume an affine risk representation

```
R_P(f) = b_f + <g_f, Psi(P)> + eta_f(P),
```

where `g_f in H` and `|eta_f(P)| <= epsilon_repr` on the source and declared
target family. This assumption is essential: a covariance of domain embeddings
does not control risk unless the risk functional is represented in the same
space (or an explicit approximation theorem supplies the residual).

Ignoring the residual for the moment and writing `ell = g_f`,

```
R_e - R_bar = <ell, delta_e>,
Var_e R_e = <ell, C_S ell>.
```

In finite dimensions, let `D = [delta_1 ... delta_m]`, `r = D^T ell`, and
`K = D^T D`. Then

```
r^T K^dagger r = ||Pi_range(D) ell||^2
                 = ||Pi_range(C_S) ell||^2.
```

The identity follows from `D K^dagger D^T = Pi_range(D)` and
`range(C_S) = range(D)`. It is a projection identity, not a new covariance
estimator.

## 2. Target shift and master support bound

Write the target displacement as

```
delta_T = delta_parallel + delta_perp,
delta_parallel in range(C_S),
delta_perp in ker(C_S).
```

For declared budgets `rho, kappa >= 0`, define the source-exposure target set

```
U_EX(rho,kappa) = {
  delta_T : ||C_S^{dagger/2} delta_parallel|| <= rho,
              ||delta_perp|| <= kappa
}.
```

For every `ell`, Cauchy-Schwarz in the `C_S` geometry gives

```
sup_{delta_T in U_EX} <ell, delta_T>
 <= rho ||C_S^{1/2} ell||
    + kappa ||Pi_ker(C_S) ell||.
```

Consequently, under the affine representation and target inclusion
`delta_T in U_EX(rho,kappa)`,

```
R_T(f) <= R_bar_S(f)
          + rho ||C_S^{1/2} g_f||
          + kappa ||Pi_ker(C_S) g_f||
          + 2 epsilon_repr.
```

The last term is a representation/conditional-mismatch allowance. It cannot be
removed using source exposure alone. The parameters `rho`, `kappa`, and target
inclusion are target-family assumptions, not source-derived information.

## 3. What the theorem does and does not unify

* **V-REx:** under the affine representation, its population penalty is exactly
  `<g_f,C_S g_f>`; empirical-to-population and non-affine translation errors need
  separate lemmas.
* **MMD:** pairwise squared MMD controls `tr(C_S)` (with the usual finite-sample
  factor). Vanilla MMD does not identify the spectrum, range, or nullspace, so a
  claim of full operator control requires an additional spectral assumption or
  penalty.
* **DRO:** a DRO objective is a support function over a declared uncertainty set.
  It is an instance of the same final support calculation only after the target
  set is identified with, or bounded by, `U_EX`; the radius is not implied by
  `C_S`.

Thus the current result is a common *bound template*, not yet a theorem that
three algorithms are exact corollaries. The decisive next step is to prove
method-specific `Omega -> (g_f, rho, kappa, epsilon_repr)` translation lemmas.

## 4. Falsification tests

The accompanying script `experiments/exposure_geometry_tests.py` checks:

1. **Exposure amplitude.** Replacing every `delta_e` by `epsilon delta_e`
   changes `C_S` and risk variance by `epsilon^2`, while
   `r^T K^dagger r` is invariant for nonzero `epsilon`. Low source-risk
   variance therefore does not imply low intrinsic sensitivity.
2. **Exposure rank.** Two centered source systems have the same V-REx value for
   a chosen `ell`, but different `rank(C_S)`. With the same `(rho,kappa)`, the
   nullspace-aware target support is different. Any proposed certificate that
   sees only the scalar V-REx value fails to distinguish them.

If either identity fails, the representation or pseudoinverse convention is
wrong. If the rank test is collapsed by setting `kappa=0` or by replacing the
operator with its trace, the claimed exposure-blindness distinction has been
discarded by assumption.

## Decision

The finite-dimensional affine mother bound survives these checks and is worth
mapping to concrete regularizers. The novelty claim remains conditional: the
operator and support algebra are standard; a contribution would require a
non-vacuous, source-estimable translation theorem for multiple regularizers.
