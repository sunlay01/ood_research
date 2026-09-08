# 3A-T Theorem Statements

All formulas use `R_e(w)=w' M_e w - 2 m_e' w + c_e`.

## Theorem 1: exact source ellipsoid

Assume `M_S w* = m_S`. For every `delta`,

\[
R_S(w^*+\delta)-R_S(w^*)=\delta^\top M_S\delta.
\]

If `M_S` is positive definite, the source-good set is exactly
\[
\{w:R_S(w)-R_S(w^*)\leq\epsilon\}-w^*
=\{\delta:\delta^\top M_S\delta\leq\epsilon\}.
\]

Positive definiteness is needed for an ellipsoid interpretation, not for the
algebraic identity itself.

## Theorem 2: exact shift polynomial

Let `A_s=M_T-M_S`, `d_m=m_T-m_S`, and
`g_s=2 A_s w* - 2 d_m`. For every `delta`,

\[
\Delta_s(w^*+\delta)-\Delta_s(w^*)
=g_s^\top\delta+\delta^\top A_s\delta.
\]

Moment matrices are symmetric, hence `A_s` is symmetric. Without symmetry,
only the symmetric part contributes to the quadratic form.

## Theorem 3: ellipsoid support formula

Assume `M` is symmetric positive definite, `epsilon >= 0`, and `g` is a
finite vector. Then

\[
\sup_{\delta^\top M\delta\leq\epsilon}|g^\top\delta|
=\sqrt{\epsilon}\sqrt{g^\top M^{-1}g}.
\]

The supremum is finite and attained. For `epsilon>0` and `g != 0`, an
attaining witness is
\[
\delta^*=\frac{\sqrt{\epsilon}M^{-1}g}
{\sqrt{g^\top M^{-1}g}}.
\]
For `epsilon=0` or `g=0`, the value is zero and `delta=0` is a valid witness.

## Theorem 4: finite-budget relevance bound

Let
\[
V_s(\epsilon)=\sup_{\delta^\top M_S\delta\leq\epsilon}
|g_s^\top\delta+\delta^\top A_s\delta|,
\]
\[
L_s=\sqrt{g_s^\top M_S^{-1}g_s},
\quad
K_s=\sup_{\delta^\top M_S\delta\leq1}|\delta^\top A_s\delta|.
\]
Then `K_s` is finite and
\[
|V_s(\epsilon)-\sqrt{\epsilon}L_s|\leq\epsilon K_s.
\]
With `H_S=2M_S`, the leading term is
\[
\sqrt{2\epsilon}\sqrt{g_s^\top H_S^{-1}g_s}.
\]
For symmetric `A_s`,
\[
K_s=\|M_S^{-1/2}A_sM_S^{-1/2}\|_{op}.
\]

## Theorem 5: first-order-null exact scaling

If `g_s=0`, then for every `epsilon >= 0`,
\[
V_s(\epsilon)=\epsilon K_s.
\]
Thus `K_s>0` gives `Theta(epsilon)`, while `K_s=0` gives exact zero.

## Theorem 6: residual coupling

For `r*=Y-w*'X`, finite moments and the source normal equation imply
\[
g_s=-2E_T[Xr^*].
\]
The same `w*` is used in both source and target expectations.

## Proposition 7: source usage is not relevance

There are positive-definite source and target moment pairs with a source
optimum having `w*_j=0` but `g_{s,j} != 0`. The explicit two-dimensional
moment construction is audited in `proof_audit.md`.

## Proposition 8: common burden is not model discrimination

If `M_T=M_S`, `m_T=m_S`, and `c_T=c_S+kappa`, then
\[
\Delta_s(w)=\kappa,
\quad B_s=|\kappa|,
\quad V_s(\epsilon)=0.
\]

## Theorem 9: independent nuisance augmentation

Let
\[
M_{aug}=\begin{pmatrix}M_E&0\\0&M_N\end{pmatrix},
\quad
g_{aug}=\begin{pmatrix}g_E\\0\end{pmatrix},
\]
where `M_E` and `M_N` are positive definite. Then
\[
g_{aug}^\top M_{aug}^{-1}g_{aug}=g_E^\top M_E^{-1}g_E.
\]
Therefore leading relevance is invariant under adding source-orthogonal,
target-residual-uncoupled nuisance coordinates. This does not imply finite-
epsilon vulnerability invariance unless the added block of `A_s` is also
controlled.

## Corollary 10: source-metric coordinate invariance

For invertible `T`, set
\[
M'=T^{-\top}MT^{-1},\qquad g'=T^{-\top}g.
\]
Then `g'^T M'^{-1} g' = g^T M^{-1} g`. This scalar is invariant, but
coordinate singular vectors are not claimed to be invariant.

## Theorem 11: general smooth leading asymptotic

Assume `R_S` is `C^2` near `w*`, `nabla R_S(w*)=0`, and
`H_S=nabla^2 R_S(w*)` is positive definite. In addition, assume the global
source sublevel sets localize at `w*`: for some neighborhood `U` and all
sufficiently small `epsilon`,
\[
\{w:R_S(w)-R_S(w^*)\leq\epsilon\}\subset U.
\]
Assume `Delta_s` is differentiable at `w*`, with gradient `g_s`. Then, as
`epsilon` tends to zero,
\[
V_s(\epsilon)=\sqrt{2\epsilon}\sqrt{g_s^\top H_S^{-1}g_s}
 + o(\sqrt{\epsilon}).
\]

The proof uses a local quadratic sandwich for the source sublevel set and an
asymptotically feasible scaled maximizing direction. A Taylor slogan without
this feasibility step is insufficient.

## Theorem 12: stronger-regularity smooth expansion

Under Theorem 11, if the source remainder is `O(||delta||^3)` and the shift
remainder is `O(||delta||^2)` locally, then
\[
V_s(\epsilon)=\sqrt{2\epsilon}\sqrt{g_s^\top H_S^{-1}g_s}+O(\epsilon).
\]
If `g_s=0`, local boundedness of the shift Hessian alone suffices for
`V_s(epsilon)=O(epsilon)`; the sharper coefficient requires a specified
second-order expansion.
