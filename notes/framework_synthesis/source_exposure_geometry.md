# Candidate inner core: source-exposure geometry

Status: `OUR-CONJECTURE / PROBE`. This document is a proposed mathematical core
inside the existing typed bridge calculus. It is not yet the project's final
framework and does not claim that every regularizer already has the required
translation.

## 1. State construction

Let `D` be a space of domains and let `Psi: D -> H_D` be a domain embedding into
a Hilbert space (for example, a kernel mean embedding of the joint law or of a
declared feature marginal). For source domains `P_1,...,P_m`, define

`mu_e = Psi(P_e)`, `mu_bar = m^{-1} sum_e mu_e`, `delta_e = mu_e - mu_bar`,

and the source-exposure covariance operator

`C_S = m^{-1} sum_e delta_e tensor delta_e`.

The state is not just `C_S`. A target-relevant learner functional is a bounded
linear functional `g_f in H_D` together with an approximation residual `eta_f`:

`R_P(f) = b_f + <g_f, Psi(P)>_{H_D} + eta_f(P)`.

The linear representability condition and a bound on `eta_f` are explicit
assumptions. Without them, a domain embedding covariance has no reason to
control prediction risk.

The proposed inner state is therefore

`S_EX = (Psi, mu_bar, C_S, g_f, eta_f)`,

while the existing outer scaffold still carries `O_S`, the admissible target
family, estimation/optimization layers and source indistinguishability.

## 2. What the operator measures

For a centered linear task functional,

`m^{-1} sum_e (R_{P_e}(f)-bar R_S(f))^2`

equals ` <g_f, C_S g_f>` up to the representation residual and cross terms.
Thus V-REx is a native quadratic probe only under the displayed representability
assumption. The operator's nullspace consists of directions not varied by the
observed source domains; it is an exposure-blind subspace, not automatically an
invariant subspace.

For a source displacement `Delta_T = Psi(P_T)-mu_bar`, the first-order target
gap is

`R_T(f)-bar R_S(f) = <g_f, Delta_T> + eta_f(P_T)-bar eta_S(f)`.

This is the point at which a target family and a conditional/joint residual must
enter. `C_S` alone does not identify `Delta_T` or the target labels.

## 3. Exposure-induced target sets and support

A nontrivial target family can be defined by an explicit decomposition

`Delta_T = Delta_parallel + Delta_perp`,

with `Delta_parallel in ran(C_S)` and `Delta_perp in ker(C_S)`. For example,

`U_EX(rho,kappa) = { Delta: ||C_S^{dagger/2} Delta_parallel|| <= rho,
                                  ||Delta_perp|| <= kappa }`.

The corresponding support bound is

`sup_{Delta in U_EX} <g_f,Delta>
 <= rho ||C_S^{1/2} g_f|| + kappa ||Pi_{ker C_S} g_f||`.

This equation exposes a necessary design choice: the nullspace budget `kappa`
is the explicit price of unobserved exposure. A source-only method cannot make
that term disappear without a structural assumption (causal coverage,
conditional invariance, or a restricted target family).

The orientation of the ellipsoid matters. A claim that simply penalizing
`||Pi_{ran C_S} g_f||^2` is always “exposure-normalized” is not valid: it may
penalize well-exposed directions while leaving the exposure-blind component
uncontrolled. Any normalized penalty must specify whether it uses `C_S`, a
pseudoinverse, or a separate nullspace penalty and what target set each choice
supports.

## 4. Relation to concrete methods

| Method | Map into `S_EX` | Fidelity |
|---|---|---|
| ERM | `bar R_S(f)` / `g_f` only if a linear task representation is assumed | `NATIVE` for source risk; no OOD bridge by itself |
| V-REx | `Var_e R_e(f) ~= <g_f,C_S g_f>` | `EXACT` only under linear representability; otherwise `REQUIRES-TRANSLATION-THEOREM` |
| MMD | pairwise `||mu_e-mu_{e'}||^2`; average pairwise distance is proportional to `tr(C_S)` | `NATIVE` for trace-level geometry; full spectrum/operator control is additional |
| CORAL | covariance features define a finite-dimensional `Psi` and moment operator | `REQUIRES-TRANSLATION-THEOREM`; conditional residual remains |
| finite-group GroupDRO | support over a simplex of risk vectors, optionally pulled back through `g_f` | `NATIVE` for risk support; exposure-induced `U_EX` is a new target-set assumption |
| Wasserstein/f-DRO | a target set may be parameterized by a metric/operator, with support duality | `NATIVE` only after `U_EX` is declared; `C_S` does not determine the radius |
| ideal IRM/ICP | a mechanism constraint can restrict admissible `Delta_T` or `eta_f` | `REQUIRES-TRANSLATION-THEOREM`; causal semantics are not encoded by `C_S` |
| IRMv1 | derivatives `D_w g_{f,w}` are differential probes if the task map is smooth | `REQUIRES-TRANSLATION-THEOREM` (restricted functional models) |
| Fishr | requires a lifted covariance operator for per-example gradient features | `REQUIRES-TRANSLATION-THEOREM`; no general bridge in current evidence |

## 5. Master theorem candidate

Under bounded loss, the representability residual bound
`|eta_f(P)| <= epsilon_repr`, and `Delta_T in U_EX(rho,kappa)`, a candidate
population statement is

`R_T(f) <= bar R_S(f)
          + rho ||C_S^{1/2} g_f||
          + kappa ||Pi_{ker C_S} g_f||
          + 2 epsilon_repr`.

Finite samples add an RKHS/operator-estimation term and computation adds an
optimization/relaxation term. This is a proposed theorem architecture; proving
it for a concrete loss, embedding and target family is the decisive next step.
The conditional-label mismatch must either be included in `epsilon_repr` with a
valid theorem or displayed as a separate `Delta_cond`; it cannot be hidden in a
generic shift term.

## 6. Why this could be more than an interface

The possible contribution is not the covariance formula itself. Covariance
operators of embeddings, risk vectors, MMD and support functions are established
objects. The potentially new contribution is a theorem-backed composition in
which:

1. source-domain exposure defines a typed operator and an exposure-blind
   subspace;
2. learner/task functionals are represented in its dual geometry;
3. regularizers become probes of the same state with explicit approximation
   errors; and
4. a single target-support theorem exposes the price of unseen directions.

This is only a contribution if the composition yields a non-vacuous theorem and
an out-of-sample method mapping, not if it is merely a new notation for existing
MMD, V-REx or DRO bounds.
