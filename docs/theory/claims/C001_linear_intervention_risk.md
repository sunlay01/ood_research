# C001: Linear Task-Preserving Intervention Risk

## Status

`PROVED_INTERNAL / NUMERICALLY_VERIFIED`

## Statement

Let

\[
C\sim\mathcal N(0,I),\quad U=LC+\xi,\quad
Y=\beta^\top C+\epsilon_Y,
\]

where \(\xi\) and \(\epsilon_Y\) are independent, zero mean, and
\(\operatorname{Var}(\epsilon_Y)=\sigma_Y^2\).  In environment \(e\), let

\[
A=R_eC+\mu_e+\eta_e,\qquad
\operatorname{Cov}(\eta_e)=\Sigma_{A,e},
\]

with independent \(\eta_e\).  An affine predictor observes only
\(X=(U,A)\):

\[
f_w(X)=w_0+w_U^\top U+w_A^\top A.
\]

The structural equation for \(Y\) is fixed in all environments; only the
nuisance mechanism varies.

Define \(D=(1,C,\xi,A)\), \(M_e=\mathbb E_e[DD^\top]\), and

\[
b(w)=(-w_0,\;\beta-L^\top w_U,\;-w_U,\;-w_A).
\]

Then

\[
R_e(w)=\sigma_Y^2+b(w)^\top M_eb(w),
\qquad
R_T(w)-R_S(w)=b(w)^\top(M_T-M_S)b(w).
\]

Both are `exact equalities`.

## Proof

Substituting \(U=LC+\xi\) into \(Y-f_w(X)\) gives

\[
Y-f_w(X)=b(w)^\top D+\epsilon_Y.
\]

The noise is independent and centered, so the cross term vanishes after
squaring and taking the expectation.  Subtract the two resulting risk
expressions for the transport identity.

## Exact Robust Formulas

For the correlation family

\[
\mathcal I_{\rm corr}=
\{R:\|R-R_0\|_F\le r,\;\mu=0,\;\Sigma_A=Q\},
\]

the exact worst-case risk is

\[
\sigma_Y^2+b_0^2+b_\xi^\top\Sigma_\xi b_\xi+b_A^\top Qb_A+
\bigl(\|b_C+R_0^\top b_A\|+r\|b_A\|\bigr)^2.
\]

For the moment family with fixed \(R_0\),

\[
\|\mu-\mu_0\|\le r_\mu,
\qquad
\|\Sigma_A-\Sigma_{A,0}\|_{\rm op}\le r_\Sigma,
\]

the exact worst-case risk is

\[
\begin{aligned}
\sigma_Y^2
&+\bigl(|b_0+b_A^\top\mu_0|+r_\mu\|b_A\|\bigr)^2
+b_\xi^\top\Sigma_\xi b_\xi\\
&+\|b_C+R_0^\top b_A\|^2
+b_A^\top\Sigma_{A,0}b_A+r_\Sigma\|b_A\|^2.
\end{aligned}
\]

The upper bounds are attained by choosing the mean perturbation parallel to
the signed residual and a rank-one covariance perturbation in the \(b_A\)
direction.

## Risk References

\[
R_S^{X,*}=\inf_{f\in\mathcal F_{\rm affine}}R_S(f),
\qquad
R_C^*=\sigma_Y^2.
\]

Thus robust causal excess is not source excess plus degradation alone:

\[
R_{\mathcal I}(f)-R_C^*
= [R_S(f)-R_S^{X,*}]
+[R_S^{X,*}-R_C^*]
+\sup_{\iota\in\mathcal I}[R_\iota(f)-R_S(f)].
\]

## Verification

`tests/test_intervention_linear.py` verifies the transport identity and both
scalar worst-case formulas.  The implementation is
`src/ood_repr_reg/intervention_linear.py`.
