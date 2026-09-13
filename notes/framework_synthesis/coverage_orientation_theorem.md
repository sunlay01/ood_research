# Stage 10: geometry orientation theorem

## 1. Status and scientific question

**Status: `ADVANCE`.** Stage 10 closes the environment-side geometry needed
before the Stage 11 risk-representation gate. The target family is held fixed
independently of the source operator. The question is whether source diversity
summaries (rank, trace, spectrum, condition number, or domain count) determine
OOD coverage. They do not: target-relative orientation supplies an independent
degree of freedom.

## 2. Fixed target-family setting

Work in (H=\mathbb R^d) with the Euclidean inner product. Let (A\succeq0)
be the source-exposure operator, (S_A=\operatorname{ran}(A)), and
(K_A=\ker(A)=S_A^\perp). For a fixed target shift subspace (V_T) and
radius (R\ge0),

\[
  \mathcal U_T=\{\delta\in V_T:\|\delta\|_2\le R\}.
\]

The Stage 9 radii are

\[
 \rho_A=R\|A^{\dagger/2}P_{S_A}P_{V_T}\|_{\rm op},\qquad
 \kappa_A=R\|P_{K_A}P_{V_T}\|_{\rm op}.
\]

Here (A,S_A) are source-derived, (V_T,R) are target-semantic, and finite
dimension/PSD self-adjointness are mathematical regularity assumptions.

## 3. Principal-angle exact theorem

For two subspaces (V,S), define the largest principal angle by

\[
 \sin\theta_{\max}(V,S):=
 \sup_{v\in V,\,\|v\|=1}\|P_{S^\perp}v\|.
\]

The supremum over the empty unit sphere is (0). If (dim V>dim S),
the convention is that one principal angle is (pi/2), so the displayed
quantity is (1). This is the standard largest-angle convention, expressed in
the projection form needed here.

### Theorem 10.1

For every (A\succeq0), (V_T), and (R\ge0),

\[
 \boxed{\kappa_A=R\sin\theta_{\max}(V_T,S_A)}.
\]

#### Proof

Since (K_A=S_A^\perp), (P_{K_A}=P_{S_A^\perp}). For any (x),

\[
 \|P_{K_A}P_{V_T}x\|
 \le \|P_{K_A}P_{V_T}\|_{\rm op}\|x\|.
\]

The range of (P_{V_T}) is (V_T), hence the operator norm is exactly

\[
 \|P_{K_A}P_{V_T}\|_{\rm op}
 =\sup_{v\in V_T,\|v\|=1}\|P_{S_A^\perp}v\|
 =\sin\theta_{\max}(V_T,S_A).
\]

The target ball is radially symmetric, so

\[
 \sup_{\delta\in\mathcal U_T}\|P_{K_A}\delta\|
 =R\|P_{K_A}P_{V_T}\|_{\rm op},
\]

which is the stated identity. In particular,

\[
 \boxed{\kappa_A=0\iff V_T\subseteq S_A}
\]

for (R>0); for (R=0), (kappa_A=0) trivially and the inclusion must be
read as a statement about the underlying unit-radius geometry.

## 4. Rank barrier

### Theorem 10.2

Let (q=\dim V_T) and (r=\operatorname{rank}(A)=\dim S_A). If (q>r),
then

\[
 V_T\cap K_A\ne\{0\}.
\]

Indeed, rank-nullity for the restriction (P_{S_A}|_{V_T}:V_T\to S_A)
gives

\[
 \dim\ker(P_{S_A}|_{V_T})\ge q-r>0,
\]

and this kernel is (V_T\cap S_A^\perp=V_T\cap K_A). A unit vector in this
intersection belongs to the radius-(R) target ball, so for (R>0),

\[
 \boxed{q>r\Longrightarrow \kappa_A=R}.
\]

This is a deterministic dimensional obstruction, not a probabilistic claim.

## 5. Source-domain-count barrier

For centered domain shifts (z_e=\delta_e-\bar\delta),

\[
 C_S=\frac1m\sum_{e=1}^m z_ez_e^\top.
\]

The vectors (z_e) sum to zero, so their span has dimension at most (m-1).
Since (\operatorname{ran}(C_S)=\operatorname{span}\{z_e\}),

\[
 \boxed{\operatorname{rank}(C_S)\le m-1}.
\]

Therefore (\dim V_T>m-1\) implies (\kappa_{C_S}=R). Complete coverage of a
(q)-dimensional target subspace requires the necessary condition

\[
 \boxed{m\ge q+1}.
\]

It is not sufficient: general-position and target-relevant excitation are
still required.

## 6. Exposed-direction strength theorem

Assume (V_T\subseteq S_A) and (R>0). Then (kappa_A=0), but coverage can
still be weak. Because (P_{S_A}P_{V_T}=P_{V_T}),

\[
 \boxed{\frac{\rho_A^2}{R^2}
 =\sup_{\substack{v\in V_T\\\|v\|=1}}v^\top A^\dagger v}.
\]

If (V_T) is (A)-reducing (equivalently here, invariant under the
self-adjoint (A)) and (A|_{V_T}) is positive definite, then

\[
 \boxed{\rho_A=\frac{R}
 {\sqrt{\lambda_{\min}(A|_{V_T})}}}.
\]

The invariance qualification matters: without it, (A|_{V_T}) is not an
operator on (V_T), and this restricted-eigenvalue formula is not valid.

## 7. Full-rank same-spectrum orientation theorem

Let (0<\lambda_{\min}<\lambda_{\max}), (R>0), and
(V_T=\operatorname{span}(e_1)\subset\mathbb R^2). Set

\[
 A_1=\operatorname{diag}(\lambda_{\max},\lambda_{\min}),\qquad
 A_2=\operatorname{diag}(\lambda_{\min},\lambda_{\max}).
\]

Both are full rank, have the same rank, spectrum, trace, and condition number.
Yet Theorem 10.3 gives

\[
 \rho_{A_1}=\frac R{\sqrt{\lambda_{\max}}},\qquad
 \rho_{A_2}=\frac R{\sqrt{\lambda_{\min}}},qquad
 \frac{\rho_{A_2}}{\rho_{A_1}}
 =\sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}.
\]

Since both operators are full rank, (kappa_{A_1}=kappa_{A_2}=0). Thus
orientation changes exposed-direction strength even when kernel ambiguity is
absent.

## 8. Fixed-spectrum orientation optimization

Let (A_U=U\Lambda U^\top), where

\[
 \Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_r,0,\ldots,0),
 \quad \lambda_1\ge\cdots\ge\lambda_r>0.
\]

The eigenvalues are fixed while (U) varies. If (q=\dim V_T>r), the rank
barrier gives (kappa_{A_U}=R) for every (U). If (q\le r), an orientation
with (V_T\subseteq\operatorname{ran}(A_U)) exists and gives (kappa=0).

Under complete directional coverage,

\[
 \frac{\rho_{A_U}^2}{R^2}
 =\max_{\substack{v\in V_T\\\|v\|=1}}v^\top A_U^{-1}v.
\]

In the eigenbasis, (A_U^{-1}) restricted to its range has eigenvalues

\[
 \mu_1\le\cdots\le\mu_r,qquad \mu_i=1/\lambda_i.
\]

Courant--Fischer, applied to (q)-dimensional subspaces of
\(\operatorname{ran}(A_U)\), gives

\[
 \min_{\dim W=q}\max_{v\in W,\|v\|=1}v^\top A_U^{-1}v
 =\mu_q=1/\lambda_q.
\]

Choosing (V_T) to be spanned by the (q) largest-eigenvalue directions of
(A_U) attains this value. Therefore the candidate formula is valid:

\[
 \boxed{\inf_{U:\,V_T\subseteq\operatorname{ran}(A_U)}\rho_{A_U}
 =\frac R{\sqrt{\lambda_q}}\qquad(q\le r).}
\]

The result is an orientation optimization theorem, not a claim that (m\ge q+1)
or (q\le r) alone guarantees excitation.

## 9. Negative controls

### 9.1 Isotropic source operator

For (A=cI), (c>0), every rotation leaves (A) unchanged. For every
target subspace ball,

\[
 \kappa_A=0,\qquad \rho_A=R/\sqrt c,
\]

independently of target orientation.

### 9.2 Rotationally invariant target family

For the ambient Euclidean ball ({\delta:\|\delta\|\le R\}), (kappa_A=0)
when (A) is full rank, while

\[
 \rho_A=R/\sqrt{\lambda_{\min}(A)}.
\]

This depends only on the spectrum. If (A) is singular but nonzero, the same
argument gives (\rho_A=R/\sqrt{\lambda_{\min}^+(A)}) and (kappa_A=R), again
with no eigenvector-orientation dependence. For (A=0), use the empty-maximum
convention: (\rho_A=0) and (kappa_A=R). Therefore orientation matters for
structured or anisotropic target geometry, not universally.

## 10. Ellipsoid extension

For (\mathcal U_Q=\{Q^{1/2}z:\|z\|\le1\}), assume (A) and (Q) are
simultaneously diagonal in one basis:

\[
 A=\operatorname{diag}(a_i),\qquad Q=\operatorname{diag}(q_i).
\]

Direct evaluation gives

\[
 \boxed{\rho_A^2=\max_{i:a_i>0}\frac{q_i}{a_i}},\qquad
 \boxed{\kappa_A^2=\max_{i:a_i=0}q_i},
\]

with the empty maximum interpreted as (0). Permuting the (a_i) while
holding their multiset fixed changes these maxima relative to a fixed (Q),
which is the ellipsoidal analogue of the subspace orientation theorem.

## 11. Deterministic tests

`notes/framework_synthesis/experiments/coverage_orientation_tests.py` checks:

1. the principal-angle projection identity and radius scaling;
2. the rank barrier;
3. the (m-1) centered-domain rank bound;
4. the full-rank same-spectrum counterexample;
5. fixed-spectrum orientation optimization in a small dimension;
6. isotropic and rotationally invariant-target negative controls;
7. commuting ellipsoid formulas.

The test file uses only explicit NumPy matrices and no data, random numbers,
training, neural networks, or benchmark datasets.

## 12. Lean verification status

The existing Lean Stage 9 files machine-check the coordinate support identity
and the finite-dimensional range/kernel decomposition. The new Stage 10 Lean
file checks a concrete two-dimensional hidden-direction witness, diagonal
line-strength ingredients for the same-spectrum example, and the isotropic
negative control. The general principal-angle, rank-nullity,
source-domain-count, and Courant--Fischer statements above are paper proofs;
the deterministic Python tests provide executable checks for their finite
matrix instances. No theorem is labelled Lean-verified beyond the exact
statements present in the Lean sources.

## 13. Scientific interpretation

\[
 \kappa_A\ \leftrightarrow\ \text{whether target-relevant directions are covered},
 \qquad
 \rho_A\ \leftrightarrow\ \text{how strongly covered directions are excited}.
\]

The hierarchy is therefore

\[
 \text{domain count}\not\Rightarrow\text{rank}\not\Rightarrow
 \text{target coverage}\not\Rightarrow\text{strong target excitation}.
\]

Relative source-target geometry, rather than scalar diversity magnitude, is the
environment-side quantity that enters the Stage 9 transfer certificate.

## 14. Decision gate

**ADVANCE.** The exact principal-angle formula, rank barrier, source-domain-count
necessary condition, full-rank same-spectrum orientation theorem, fixed-spectrum
optimization result, and negative controls are established. Stage 11 may now
begin its restricted risk-representation hard gate. No algorithm mapping or new
regularizer is authorized by this result.

## 15. Final scientific verdict

1. Proved on paper: projection/principal-angle identity, rank and domain-count barriers, strength formula, fixed-spectrum min-max result, and ellipsoid commuting formulas.
2. `kappa_A` measures target directions invisible to the source range.
3. `rho_A` measures inverse source excitation on covered target directions.
4. If `q > r`, a target direction is necessarily source-blind and `kappa_A = R`.
5. Centered `m`-domain exposure has rank at most `m-1`, so `m >= q+1` is necessary.
6. Same-spectrum full-rank operators can have different `rho_A`.
7. Orientation does not matter for isotropic sources or rotationally invariant target balls.
8. Python checks all listed finite matrix instances deterministically.
9. Lean verifies only the concrete coordinate/scalar statements in `lean/OodTheoryVerification/Stage10`.
10. Stage 11 is authorized; algorithm mapping and new regularizer design remain blocked.
