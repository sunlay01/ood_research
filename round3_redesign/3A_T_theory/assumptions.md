# 3A-T Assumptions

This track audits the completed 3A population quadratic benchmark. It does
not alter the benchmark or its conclusions.

The fixed convention is

\[
R_e(w)=w^\top M_e w-2m_e^\top w+c_e,
\qquad M_e=E_e[XX^\top],\quad m_e=E_e[XY].
\]

For the source environment, `M_S` is symmetric positive definite and
`w*` satisfies `M_S w* = m_S`. The Hessian convention is `H_S=2M_S`.

The quadratic results are finite-dimensional, population statements. Every
moment used below is assumed finite. Target quantities are evaluated after
the source optimum has been fixed; they are not used for source fitting or
model selection.

The smooth extension uses a separate local statement. There `R_S` is twice
continuously differentiable near `w*`, `nabla R_S(w*)=0`, and
`H_S=nabla^2 R_S(w*)` is positive definite. The shift `Delta_s` is at least
continuously differentiable near `w*`; a local bounded Hessian or quadratic
remainder is added only for the `O(epsilon)` conclusion.

The singular-source discussion is exploratory and is not used to extend the
main theorems beyond positive-definite source geometry.
