# Stage 11: risk representation hard gate

## 1. Status and scientific question

**Status: `REVISE`.** There are valid, nontrivial finite-dimensional supervised
states, including an exact quadratic-regression state. The representation and
its risk derivative semantics pass. The hard gate is not an unconditional
`ADVANCE`, however, because the Euclidean values of the split geometry
(ρ, κ, and especially the nullspace norm) change under a non-orthogonal
coordinate reparameterization. Before Stage 12, the theorem must declare its
Hilbert metric as part of the scientific model or be rewritten in a
metric-covariant gauge form.

## 2. Nontriviality criterion

A valid state is defined before selecting a learner (f), is finite-dimensional
or has a primitive approximation guarantee, and does not store the full law or
the risk vector. The state may discard distributional information, but the
report must identify the discarded information and show that it is irrelevant
for the declared loss/model class.

## 3. Exact finite-dimensional linear-span model

Let (phi:\mathcal X\times\mathcal Y\to\mathbb R^d) be measurable and
integrable under every admissible (P). Suppose the restricted loss class has
the fixed feature-span form

\[
 \ell(f(x),y)=b_f+g_f^\top\phi(x,y).
\]

Define the algorithm-independent state

\[
 \Psi(P)=\mathbb E_P[\phi(X,Y)].
\]

Linearity of expectation gives the exact identity

\[
 R_P(f)=b_f+g_f^\top\Psi(P),\qquad \eta_f(P)=0.
\]

The state is independent of (f), and labels enter directly through the joint
feature (phi(X,Y)). It discards all aspects of (P) not visible to the
chosen finite feature span. Consequently this is exact only for the declared
loss class; it is not an exact representation for arbitrary neural losses or
arbitrary supervised distributions.

## 4. Exact supervised quadratic model

For (f_w(x)=w^\top x) and squared loss,

\[
 (w^\top x-y)^2
 =w^\top(xx^\top)w-2w^\top(xy)+y^2.
\]

Assume (\mathbb E_P\|X\|^2<\infty), (\mathbb E_P|XY|<\infty), and
(\mathbb E_PY^2<\infty). Define

\[
 \Psi(P)=\big(M(P),c(P),s(P)\big)
 =\big(\mathbb E_P[XX^\top],\mathbb E_P[XY],\mathbb E_P[Y^2]\big).
\]

The matrix can be vectorized symmetrically (for (d) covariates this gives
(d(d+1)/2+d+1) coordinates). With the coordinate convention that the matrix
slot is paired by (\langle W,M\rangle_F=\operatorname{tr}(W^\top M)), set

\[
 g_w=(ww^\top,-2w,1),\qquad b_w=0.
\]

Then

\[
 \boxed{R_P(w)=\langle ww^\top,M(P)\rangle_F-2w^\top c(P)+s(P).}
\]

This is exact, finite-dimensional, and algorithm-independent. It distinguishes
conditional/label shift: (M(P)) may stay fixed while (c(P)=\mathbb E[XY])
changes. Both (c(P)) and (s(P)) are necessary in general; omitting (c)
loses the sign of the (X\)-(Y) relation, while omitting (s) loses the label
energy term. A state containing only (P_X), or only (M(P)), is insufficient.

In this model the source exposure operator is simply the covariance of the
moment-state vector (Psi(P_e)). Stage 9/10 then have a concrete meaning:

* (\kappa_A) prices conditional/moment directions never varied by the source
  domains;
* (\rho_A) prices weak source excitation of target-relevant moment directions;
* (S_A(w)=\sqrt{g_w^\top A g_w}) is the source-exposure-weighted change in
  squared-loss risk;
* (N_A(w)=\|P_{\ker A}g_w\|_2) is sensitivity in moment directions unseen by
  the source environments.

## 5. Local smooth/Taylor model

Let (\theta(P)\in\mathbb R^d) be a scientifically declared state and let
(R(f,\theta)) be (C^2) on a convex neighborhood containing the segment
between (\bar\theta) and (\theta). Taylor's theorem gives

\[
 R(f,\theta)=R(f,\bar\theta)+\nabla_\theta R(f,\bar\theta)^\top
 (\theta-\bar\theta)+\eta_f(\theta),
\]

with (g_f=\nabla_\theta R(f,\bar\theta)) and, if
\(
 \|\nabla_\theta^2R(f,\vartheta)\|_{\rm op}\le L_f
\)
along the segment,

\[
 |\eta_f(\theta)|\le \tfrac12L_f\|\theta-\bar\theta\|_2^2.
\]

This is a local approximation. A uniform (L_f\) requires an explicit model
class restriction; it cannot be hidden inside an unspecified
\(\epsilon_{\rm repr}\). The state parameterization is meaningful only when
its coordinates have declared statistical content (for example moments or a
mechanism parameter), not when chosen after seeing a particular predictor.

## 6. Risk-sensitivity semantics

For the exact affine models, if
\(
 \Psi(P_t)=\Psi(P_0)+t\delta+o(t)
\)
then

\[
 R_{P_t}(f)=R_{P_0}(f)+t\,g_f^\top\delta+o(t),
\]

so

\[
 \left.\frac d{dt}R_{P_t}(f)\right|_{t=0}=g_f^\top\delta.
\]

For the Taylor model this follows from the chain rule and the bounded quadratic
remainder. Thus (g_f) is genuinely risk sensitivity to state perturbations,
not merely an algebraic coefficient.

## 7. Learner-side sensitivity quantities

For the quadratic state and source covariance (A),

\[
 S_A(w)=\sqrt{g_w^\top A g_w},\qquad
 N_A(w)=\|P_{\ker A}g_w\|_2.
\]

The first is the RMS source-domain risk change predicted by the affine state
model. The second is sensitivity in state directions not excited by the source.
Two predictors can have the same source-risk vector but different (N_A): if
source shifts span only coordinate 1, (g_1=(0,1)) and (g_2=(0,2)) induce the
same source risk changes (zero) but different kernel sensitivities. Therefore
(N_A) is not source-identifiable from the source risk vector without extra
structure, even though it is well-defined once the full state representation
and metric are declared.

## 8. Identifiability attack

The exact moment state deliberately retains joint information. We tested the
following weaker states and found failures:

* **Marginal-only:** same (P_X), with (Y=X) versus (Y=-X), gives equal
  (P_X) and different squared risks for (w=1).
* **Incomplete moments:** the same (\mathbb E[X^2]) and (\mathbb E[Y^2])
  but opposite (\mathbb E[XY]) give risks (0) and (4) at (w=1).
* **Blind sensitivity:** source variation along one coordinate cannot identify
  the coefficient of an unexcited coordinate; (N_A) differs while source risk
  variation agrees.

These are formal limits, not bugs to be hidden in a residual.

## 9. Marginal/conditional counterexamples

The marginal-only counterexample proves that a generic supervised OOD state
cannot be (\Psi(P_X)) alone. Conditional/label information must enter through
joint moments or an explicitly conditional representation. In the quadratic
model, (\mathbb E[XY]) is the minimal cross-moment needed to distinguish the
two label mechanisms in the witness.

## 10. Coordinate/gauge audit

Let (\Psi'=T\Psi) for invertible (T). Risk preservation requires

\[
 g'=T^{-\top}g,\qquad \delta'=T\delta,\qquad A'=TAT^\top.
\]

The bilinear risk perturbation is invariant:

\[
 (g')^\top\delta'=g^\top\delta.
\]

Likewise (g'^\top A'g'=g^\top Ag), so the quadratic sensitivity

\[
 S_A(g)=\sqrt{g^\top Ag}
\]

is invariant under the paired congruence/dual transformation. However, the
Euclidean norm of a kernel component and the pseudoinverse-based radius are not
invariants under arbitrary non-orthogonal (T). A shear can change
\(\|P_{\ker A}g\|_2\) while preserving every bilinear risk value.

Therefore the current split certificate is scientifically meaningful only after
one of the following is made explicit:

1. the Hilbert inner product/units are part of the declared statistical model;
2. all target norms and projectors are transformed with the corresponding
   metric; or
3. the theorem is restated in a coordinate-free convex-gauge/support form.

Whitening or arbitrary rescaling cannot be called harmless preprocessing. This
is the reason for the `REVISE` decision rather than an unconditional advance.

## 11. Exact support versus split geometry

The exact target support (h_{\mathcal U}(g)=\sup_{\delta\in\mathcal U}g^\top\delta)
is the fundamental object. The Stage 9 expression

\[
 \rho_A S_A(g)+\kappa_A N_A(g)
\]

is exact for the independent product class whose range and kernel budgets are
separate. It can be strictly loose for a coupled physical family: the shifts
\(\{(1,1),(-1,-1)\}) have support zero against (g=(1,-1)), while the split
outer bound is (2). Thus ρ and κ should remain explanatory/task-side
coverage quantities unless the target family genuinely factorizes.

## 12. Residual audit

The linear-span and quadratic models have ε_repr=0. The Taylor model has the
primitive bound

\[
 \epsilon_{\rm repr}(f,\theta)
 \le \tfrac12L_f\|\theta-\bar\theta\|^2,
\]

valid only on the declared convex interpolation neighborhood. For a target
family with radius (r\), a uniform residual is at most (Lr^2/2) if
\(L=\sup_fL_f<\infty). If this residual is comparable to the target-risk
scale, the representation certificate is vacuous and the model must be
narrowed.

## 13. Deterministic tests

`notes/framework_synthesis/experiments/risk_representation_tests.py` checks:

1. linear-span and quadratic identities;
2. conditional/label and marginal-only counterexamples;
3. incomplete moments;
4. Taylor remainder;
5. bilinear coordinate invariance and (A'=TAT^\top);
6. non-invariance of Euclidean kernel sensitivity under shear;
7. blind sensitivity and exact-versus-split support.

It uses only explicit finite arrays and deterministic arithmetic. It passes with
`ALL STAGE-11 TESTS PASSED`.

## 14. Lean verification status

Lean verifies the exact low-dimensional squared-loss expansion and linear-state
identity in `lean/OodTheoryVerification/Stage11/Basic.lean`. The Taylor,
general coordinate-gauge, identifiability, and support statements are paper
proofs with deterministic executable witnesses; they are not labelled Lean
verified.

## 15. Failure modes

The hard failures are: marginal-only state, omission of conditional moments,
unbounded Taylor residual, predictor-dependent states, and treating (N_A) as
source-identifiable without extra structure. A further structural issue is the
metric dependence of Euclidean split geometry under arbitrary invertible
reparameterization.

## 16. Decision gate

**REVISE.** Stage 11 establishes a nontrivial algorithm-independent state and
an exact supervised finite-dimensional representation, with valid sensitivity
and residual semantics. Before Stage 12, revise the theorem interface to make
the metric/gauge explicit and coordinate-covariant, and state (N_A)'s
identifiability limitations. Do not begin algorithm mapping, new regularizers,
or the Stage 12 master theorem until this revision is audited.

## 17. Final scientific verdict

1. Successful state: `(E[XX^T], E[XY], E[Y^2])`, plus the finite feature-span state.
2. It is algorithm-independent and finite-dimensional.
3. The squared-loss linear-regression model is represented exactly.
4. Labels/conditionals enter through `E[XY]` and `E[Y^2]`.
5. `g_w=(ww^T,-2w,1)` and is the state-risk derivative.
6. Exact models have zero residual; Taylor residual is `L_f ||theta-theta_bar||^2/2`.
7. Marginal-only and incomplete-moment states fail explicit counterexamples.
8. `S_A(w)=sqrt(g_w^T A g_w)` is invariant under paired congruence.
9. `N_A` and pseudoinverse radii require an explicit metric/gauge; `N_A` is not source-identifiable in general.
10. Exact support is sharper than the split certificate for coupled target families.
11. Paper proofs, Python checks, and Lean checks are separated explicitly above.
12. Stage 12 is not authorized until the metric/gauge revision passes audit.
