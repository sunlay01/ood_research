# Analytic Theory

This is the paper-proof record for the quadratic core. It uses one
normalization throughout:

\[
R_e(w)=w^\top M_e w-2m_e^\top w+c_e,
\qquad M_e=E_e[XX^\top],\quad m_e=E_e[XY].
\]

Assume `M_S` is symmetric positive definite and `w*` satisfies
`M_S w* = m_S`. Then `H_S = 2 M_S`.

## T3A-T1: exact source ellipsoid

For `w = w* + delta`, direct expansion gives

\[
\begin{aligned}
R_S(w^*+\delta)-R_S(w^*)
&=2\delta^\top M_Sw^* -2m_S^\top\delta+\delta^\top M_S\delta\\
&=\delta^\top M_S\delta.
\end{aligned}
\]

The linear term vanishes because `M_S w* = m_S`. Therefore

\[
\mathcal F_\epsilon-w^*
=\{\delta:\delta^\top M_S\delta\leq\epsilon\}
=\{\delta:\tfrac12\delta^\top H_S\delta\leq\epsilon\}.
\]

This is an exact algebraic identity, not a Taylor approximation.

## T3A-T2: exact shift polynomial

Let `Delta M = M_T-M_S`, `Delta m=m_T-m_S`, and `Delta c=c_T-c_S`.
Then

\[
\Delta_s(w)=w^\top\Delta M w-2\Delta m^\top w+\Delta c.
\]

With `w=w*+delta`, subtracting `Delta_s(w*)` leaves

\[
\Delta_s(w^*+\delta)-\Delta_s(w^*)
=(2\Delta M w^*-2\Delta m)^\top\delta
+\delta^\top\Delta M\delta.
\]

Define `g_s=2 Delta M w*-2 Delta m` and `A_s=Delta M`.
This identity is exact for every `delta`.

## T3A-T3: ellipsoid support

For `M \succ 0`, define `\langle x,y\rangle_M=x^\top My`. Then

\[
g^\top\delta=\langle M^{-1}g,\delta\rangle_M.
\]

Cauchy-Schwarz gives

\[
|g^\top\delta|\leq
\sqrt{g^\top M^{-1}g}\sqrt{\delta^\top M\delta}.
\]

The upper bound is attained for `g != 0` and `epsilon > 0` by

\[
\delta^*=\frac{\sqrt\epsilon M^{-1}g}{\sqrt{g^\top M^{-1}g}},
\]

because `delta*^T M delta* = epsilon`. Hence

\[
\sup_{\delta^\top M\delta\leq\epsilon}|g^\top\delta|
=\sqrt\epsilon\sqrt{g^\top M^{-1}g}.
\]

The independent whitening proof sets `u=M^{1/2}\delta` and
`a=M^{-1/2}g`. The constraint becomes `||u||_2 <= sqrt(epsilon)` and the
objective becomes `|a^T u|`. Euclidean Cauchy-Schwarz gives the same value,
with witness `u*=sqrt(epsilon)a/||a||_2`.

The code reports `sqrt(g_s^T H_S^-1 g_s)`, so the leading vulnerability is

\[
\sqrt{2\epsilon}\sqrt{g_s^\top H_S^{-1}g_s}
=\sqrt\epsilon\sqrt{g_s^\top M_S^{-1}g_s}.
\]

## T3A-T4: finite-epsilon two-sided bound

Define

\[
L_s=\sqrt{g_s^\top M_S^{-1}g_s},
\qquad
K_s=\sup_{\delta^\top M_S\delta\leq1}|\delta^\top A_s\delta|.
\]

For every feasible `delta`, homogeneity gives

\[
|g_s^\top\delta|\leq\sqrt\epsilon L_s,
\qquad
|\delta^\top A_s\delta|\leq\epsilon K_s.
\]

The triangle inequality yields

\[
V_s(\epsilon)\leq\sqrt\epsilon L_s+\epsilon K_s.
\]

Using the linear maximizing witness from T3A-T3 and
`|a+b| >= |a|-|b|` gives

\[
V_s(\epsilon)\geq\sqrt\epsilon L_s-\epsilon K_s.
\]

Consequently,

\[
|V_s(\epsilon)-\sqrt\epsilon L_s|\leq\epsilon K_s.
\]

Since `H_S=2M_S`, this is equivalently

\[
\left|V_s(\epsilon)-\sqrt{2\epsilon}
\sqrt{g_s^\top H_S^{-1}g_s}\right|\leq\epsilon K_s.
\]

For symmetric `A_s`, whitening shows

\[
K_s=\|M_S^{-1/2}A_sM_S^{-1/2}\|_{\mathrm{op}}.
\]

The implementation computes the exact finite value through two global
quadratic trust-region solves, not by treating the leading expression as the
finite-epsilon answer.

## T3A-T5: first-order-null scaling

If `g_s=0`, T3A-T2 reduces the response to `delta^T A_s delta`. Set
`delta=sqrt(epsilon)u`. Then

\[
V_s(\epsilon)=\epsilon
\sup_{u^\top M_Su\leq1}|u^\top A_su|
=\epsilon K_s.
\]

Therefore `K_s > 0` implies `Theta(epsilon)`, while `K_s=0` implies exact
zero vulnerability. In particular, `g_s=0` does not by itself imply zero
finite-epsilon vulnerability.

## T3A-T6: residual coupling

For `r*=Y-w*^T X`,

\[
E_e[Xr^*]=E_e[XY]-E_e[XX^\top]w^*=m_e-M_e w^*.
\]

Subtracting source from target and using `M_Sw*=m_S` gives

\[
g_s=-2(E_T[Xr^*]-E_S[Xr^*])=-2E_T[Xr^*].
\]

This identity is the relevance criterion. It does not identify a causal
mechanism and it does not equate source feature usage with OOD relevance.

## Counterexample

Take

\[
M_S=\begin{pmatrix}2&0\\0&1\end{pmatrix},\quad
m_S=\begin{pmatrix}1\\0\end{pmatrix},\quad
M_T=\begin{pmatrix}2&1\\1&2\end{pmatrix},\quad
m_T=\begin{pmatrix}1\\1\end{pmatrix}.
\]

Then `w*=(1/2,0)^T`, but

\[
g_s=2(M_T-M_S)w^*-2(m_T-m_S)=(0,-1)^\top.
\]

Thus a zero source coefficient does not imply zero leading relevance.
