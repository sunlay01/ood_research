# Stage 9: physical target coverage calibration

Status: `REVISE / CONTROL-REQUIRED`. Finiteness and support algebra are proved,
but Stage 9 is not complete until the coverage radii admit explicit
non-vacuous upper bounds for at least one nontrivial physical family.

## 1. Status and scientific question

The question is whether an externally specified target family interacts with a
source-exposure operator in a finite, interpretable, and falsifiable way. The
order is deliberately

```text
physical target family -> induced coverage -> target-risk support bound
```

The initial setting is `H = R^d`. No RKHS or finite-sample concentration claim
is made here.

## 2. Definitions and provenance

Let `A >= 0` be a self-adjoint source-exposure operator.

* `SOURCE-DERIVED`: `A`, when it is constructed from source laws by one declared
  rule such as `A=C_S`.
* `TARGET-SEMANTIC`: a fixed physical family `U_phys`, independent of `A` and of
  the regularizer.
* `MATHEMATICAL-REGULARITY`: finite dimension, compactness, orthogonal
  projections, and the Moore-Penrose pseudoinverse.
* `STABILIZATION-CHOICE`: ridge `A_lambda=A+lambda I` or spectral cutoff
  `A_tau`; these are numerical/estimable surrogates, not new scientific facts.

For a compact `U_phys`, define

```text
rho_A(U_phys)   = sup_{delta in U_phys} ||A^{dagger/2} Pi_range(A) delta||,
kappa_A(U_phys) = sup_{delta in U_phys} ||Pi_ker(A) delta||.
```

These are coverage quantities measured in the geometry induced by `A`. They are
not source-identifiable target statistics.

## 3. Compact-family finiteness theorem

**Theorem 9.1.** If `H=R^d`, `A>=0` is fixed, and `U_phys` is compact, then
`rho_A(U_phys)` and `kappa_A(U_phys)` are finite and their suprema are attained.

**Proof.** In finite dimension, `A^{dagger/2}`, `Pi_range(A)`, and `Pi_ker(A)`
are bounded linear maps. Their compositions with the Euclidean norm are
continuous. A continuous real-valued function on a compact set has a finite
maximum (Weierstrass theorem). `A=0` is included: then `A^{dagger/2}=0`, so
`rho_A=0` and `kappa_A=sup ||delta||`.

Compactness is doing real work. For `U_phys=R^d` and `A=I`, `rho_A=+infinity`.
For `A=diag(1,0)` and `U_phys=R^2`, `kappa_A=+infinity`. Thus boundedness or a
compact equivalent is required for a non-vacuous radius.

## 4. Zero-nullspace coverage theorem

**Theorem 9.2.** In finite dimension,

```text
kappa_A(U_phys)=0  <=>  U_phys subseteq range(A).
```

**Proof.** Every summand in the supremum is nonnegative. The supremum is zero
iff every `delta in U_phys` satisfies `Pi_ker(A) delta=0`. Since `A` is
self-adjoint in finite dimension, `ker(A)^perp=range(A)`. No closure, symmetry,
centering, or containment of zero is needed.

Scientifically, `kappa_A=0` says only that the declared target shifts have no
component invisible to the source-exposure subspace. It does not make
`ker(A)` an invariant or causal subspace.

## 5. Exact support identity for the operator-defined product class

Define

```text
U_A(rho,kappa) = { delta=delta_R+delta_K:
  delta_R in range(A), delta_K in ker(A),
  ||A^{dagger/2} delta_R|| <= rho, ||delta_K|| <= kappa }.
```

**Theorem 9.3.** For every `g`,

```text
h_{U_A(rho,kappa)}(g)
 = rho ||A^{1/2}g|| + kappa ||Pi_ker(A)g||.
```

**Proof.** Write `g=g_R+g_K` by the same orthogonal decomposition. On the range,
set `u=A^{dagger/2}delta_R`. Then `delta_R=A^{1/2}u` and

```text
<g_R,delta_R> = <A^{1/2}g, u> <= rho ||A^{1/2}g||.
```

On the kernel, Cauchy-Schwarz gives
`<g_K,delta_K> <= kappa ||g_K||`. The two constraints are independent, so the
upper bounds add and are jointly attainable.

If `A^{1/2}g != 0` and `rho>0`, an exposed maximizer is

```text
delta_R^* = rho A g / ||A^{1/2}g||.
```

If `A^{1/2}g=0` or `rho=0`, choose `delta_R^*=0`; every feasible range point is
optimal when the corresponding objective component vanishes. If
`Pi_ker(A)g != 0` and `kappa>0`, a kernel maximizer is

```text
delta_K^* = kappa Pi_ker(A)g / ||Pi_ker(A)g||.
```

If the kernel component or `kappa` vanishes, choose `delta_K^*=0`; all feasible
kernel points are tied when the objective component is zero. These alignment
conditions are exactly the equality conditions for the two Cauchy-Schwarz
steps. The degenerate case `A=0` reduces to `kappa ||g||`.

## 6. External physical family gives only an outer bound

For `rho_A=rho_A(U_phys)` and `kappa_A=kappa_A(U_phys)`, every physical shift
satisfies both defining inequalities of `U_A(rho_A,kappa_A)`. Therefore

```text
U_phys subseteq U_A(rho_A,kappa_A),
h_{U_phys}(g) <= rho_A ||A^{1/2}g||
                    + kappa_A ||Pi_ker(A)g||.
```

The product class has an exact support identity. An external physical family
generally has only this outer upper bound because its range and kernel
components may be coupled.

Equality holds for a particular `g` when the physical family contains a shift
whose two components simultaneously attain the aligned range and kernel
maxima. It also holds for all `g` if the physical family equals the product class
(or has the same support function). It is strictly loose when the physical
family forbids those component combinations. For example, with
`A=diag(1,0)`, `U_phys={+(1,1),-(1,1)}`, and `g=(1,-1)`, the physical support is
zero, while `rho_A=kappa_A=1` and the outer bound is `2`. A perturbation
`g=(1,-1+epsilon)` gives physical support `|epsilon|` and an outer bound tending
to `2`, so the ratio can be arbitrarily large.

## 7. Scale-calibration theorem

For `A_epsilon=epsilon^2 A`, `epsilon != 0`, and fixed `U_phys`,

```text
range(A_epsilon)=range(A),
ker(A_epsilon)=ker(A),
rho_{A_epsilon}=rho_A/|epsilon|,
kappa_{A_epsilon}=kappa_A.
```

This follows from `(A_epsilon)^{dagger/2}=|epsilon|^{-1}A^{dagger/2}` and the
unchanged projectors. Since
`||A_epsilon^{1/2}g||=|epsilon| ||A^{1/2}g||`,

```text
rho_{A_epsilon} ||A_epsilon^{1/2}g||
 = rho_A ||A^{1/2}g||.
```

Holding a product-class parameter `rho` fixed while scaling `A` changes the
physical target family and can create a false improvement. Fixed-family
calibration removes that artifact.

## 8. Subspace-ball formulas

Let `V_T` be fixed independently of `A`, let `P_V` be its orthogonal projector,
and let

```text
U_ball = {delta in V_T: ||delta|| <= R}.
```

Then

```text
rho_A(U_ball) = R ||A^{dagger/2} Pi_range(A) P_V||_op,
kappa_A(U_ball) = R ||Pi_ker(A) P_V||_op.
```

The factor `R` appears exactly once by homogeneity. The second norm is the cosine
of the smallest principal angle between `V_T` and `ker(A)` (equivalently the
sine of the largest principal angle between `V_T` and `range(A)`). It is zero
iff `V_T subseteq range(A)`.

## 9. Ellipsoid formulas

For fixed `Q>=0`, independent of `A`, define the singular ellipsoid

```text
U_Q = {Q^{1/2} z: z in range(Q), ||z|| <= 1}.
```

Because `Q^{1/2}` vanishes on `ker(Q)`, the restriction to `range(Q)` is
equivalent to taking the operator norm over all `z`. Hence

```text
rho_A(U_Q)   = ||A^{dagger/2} Pi_range(A) Q^{1/2}||_op,
kappa_A(U_Q) = ||Pi_ker(A) Q^{1/2}||_op.
```

These formulas remain valid for singular `A` and singular `Q`. The first norm
depends on the positive eigenvalues of `A` and the orientation of its range
relative to the axes of `Q`; the second depends on overlap between
`range(Q)` and `ker(A)`. Equivalently, the squared norms are largest generalized
eigenvalues of the corresponding compressed operators.

## 10. Ideal versus stable geometry

The ideal scientific objects are `A`, `A^dagger`, `range(A)`, and `ker(A)`.
They expose non-identifiability. A ridge surrogate
`A_lambda=A+lambda I` has `ker(A_lambda)={0}` for `lambda>0`, but this does not
mean hidden directions became observed. For `A=diag(1,0)` and a target shift
along `e_2`, the ideal values are `rho_A=0`, `kappa_A=1`; ridge gives
`kappa_{A_lambda}=0` and `rho_{A_lambda}=lambda^{-1/2}`. Ridge has converted a
kernel ambiguity into a large sensitivity price.

A spectral cutoff
`A_tau=sum_{lambda_i>=tau}lambda_i u_i u_i^T` preserves an explicit kernel,
possibly enlarging it by treating small positive eigenvalues as hidden. It can
therefore introduce a deterministic bias, but it does not erase the scientific
meaning of non-identifiability. Stable quantities must always be reported beside
their ideal counterparts.

## 11. Falsification and counterexamples

* **F1, unbounded family:** `U=R^d`, `A=I` gives `rho=+infinity`.
* **F2, nullspace dominated:** with `A=diag(1,0)`, target ball radius one, and
  `g=(epsilon,1)`, the outer bound is `epsilon+1`; the kernel term dominates as
  `epsilon -> 0`. This is informative about unidentifiability, although it is
  vacuous for deployment if `kappa` is too large relative to the acceptable
  target-risk scale.
* **F3, loose outer approximation:** the two-point diagonal example in Section 6
  has physical support zero and outer bound two; the looseness ratio is unbounded.
* **F4, wrong scale handling:** holding `rho` fixed under `A -> epsilon^2 A`
  shrinks the operator-defined target family and makes the exposed term scale by
  `|epsilon|`; recalibrating `rho` by `1/|epsilon|` restores invariance.
* **F5, diversity without coverage:** `A=diag(100,100,0)` has high trace and rank
  two, while a physical family of shifts along `e_3` has `rho_A=0` and
  `kappa_A=1`. Large source diversity in the wrong directions gives no coverage.

The deterministic checks are in
`experiments/physical_target_coverage_tests.py` and include product support,
scale calibration, ball/ellipsoid formulas, orientation, loose outer support,
and ridge/cutoff sanity.

## 12. Coverage controllability and non-vacuity

Finiteness alone is insufficient. The quantity entering the risk bound is

```text
rho_A S_A(f) + kappa_A N_A(f),
S_A(f)=||A^(1/2) g_f||,
N_A(f)=||Pi_ker(A) g_f||.
```

The following sufficient conditions make the radii controllable.

### Ellipsoid domination

For the externally fixed ellipsoid `U_Q`, if

```text
Q <= c A
```

in the PSD order, then `range(Q) subseteq range(A)`, hence
`kappa_A(U_Q)=0` and

```text
rho_A(U_Q)^2 = lambda_max(A^dagger/2 Q A^dagger/2) <= c.
```

Thus `rho_A <= sqrt(c)`. This is a coverage-domination condition, not a
consequence of having many source domains. It says every target-relevant
direction is excited by the source operator at least up to a common factor.

### Subspace-ball orientation and excitation

For `U_ball={delta in V_T: ||delta||<=R}`,

```text
kappa_A(U_ball) = R ||Pi_ker(A) P_V||_op <= R eta
```

whenever the principal-angle mismatch is bounded by `eta`. If `V_T` is a
reducing subspace for `A` and `A|_{V_T} >= alpha I`, then

```text
rho_A(U_ball) <= R / sqrt(alpha),
kappa_A(U_ball)=0.
```

The reducing-subspace qualification is essential; a bare ambient inequality
without an invariance condition does not automatically control the pseudoinverse
on `V_T`.

### Shift-generator form

If the physical mechanism is `delta=B z`, `||z||<=1`, then

```text
rho_A = ||A^dagger/2 Pi_range(A) B||_op,
kappa_A = ||Pi_ker(A) B||_op.
```

If `B B^T <= c A`, then `range(B) subseteq range(A)`, `kappa_A=0`, and
`rho_A <= sqrt(c)`. This expresses coverage directly as a relation between the
source exposure operator and the physical shift generator.

### Non-vacuity gate

For a loss in `[0,1]`, a target-risk certificate is non-vacuous only if

```text
bar_R_S + rho_A S_A(f) + kappa_A N_A(f) + 2 epsilon_repr < 1.
```

A stronger pre-registered requirement is

```text
rho_A S_A(f) + kappa_A N_A(f) + 2 epsilon_repr <= epsilon_transfer,
```

where `epsilon_transfer` is fixed before comparing regularizers. If no explicit
coverage or learner-side condition achieves this, the theorem correctly reports
non-identifiability but does not provide a useful certificate.

These conditions separate control paths: target-family structure can bound
`rho_A,kappa_A`; learner design can reduce `S_A,N_A`; source or augmentation
design can change `A`. None of these should be conflated with a source-only
estimate of an unobserved target law.

## 13. Scientific interpretation

The quantities are finite and interpretable for compact finite-dimensional target
families, but they remain target-semantic calibrations rather than source-only
statistics. Coverage depends on relative orientation, not only rank, trace, or
domain count. The nullspace term is an explicit price for source information that
cannot identify target-relevant directions. Ridge can stabilize estimation only
by pricing those directions; it cannot make them identifiable.

The operator-defined product class yields an exact support identity. An external
physical family yields an outer bound, potentially arbitrarily loose. This
distinction prevents the support algebra from being mistaken for a target-family
theorem.

## 14. Decision gate

**REVISE.** Stage 9 has proved finite, explicit, and falsifiable coverage
geometry, exact product support, external-family outer bounds, scale calibration,
orientation sensitivity, and ideal/stable semantics. However, the new scientific
gate requires an explicit small-coverage sufficient condition and a non-vacuity
check. The domination, excitation, and generator conditions above provide the
candidate routes; they must be instantiated and audited before Stage 10.

Stage 10, algorithm mapping, regularizer invention, finite-sample concentration,
and universal-DG claims are not authorized yet.

## 15. Final scientific verdict

1. Proved: compact-family finiteness, zero-kernel equivalence, exact product support,
   outer physical-family bound, scale calibration, and ball/ellipsoid formulas.
2. Defined: `rho_A`, `kappa_A`, `U_A`, and the physical target families.
3. Source-derived: only `A` when built from source laws.
4. Target-semantic: `U_phys`, `V_T`, `Q`, `R`, `rho_A`, and `kappa_A`.
5. Stabilization choices: ridge and spectral cutoff.
6. The support bound is exact for `U_A` and generally outer for `U_phys`.
7. The nullspace term is informative about non-identifiability, but can be vacuous
   for an overly broad target family.
8. Orientation matters beyond rank and trace.
9. Stable geometry preserves the ideal meaning only when reported alongside it.
10. Stage 10 is not yet authorized; first pass the controllability and
    non-vacuity gate.
