# 3C Theory

Let \(H_S\succ0\), \(a_j=\nabla\Omega_j(w^*)\), and
\(J_j=\nabla^2\Omega_j(w^*)\).  For a locally quadratic objective,

\[
\mathcal J(\delta)=\frac12\delta^TH_S\delta+\lambda a_j^T\delta+
\frac\lambda2\delta^TJ_j\delta,
\]

the stationary point is

\[
\delta_{j,\lambda}=-\lambda(H_S+\lambda J_j)^{-1}a_j
\]

whenever the metric is positive definite.  With
\(b_j=H_S^{-1/2}a_j\), \(K_j=H_S^{-1/2}J_jH_S^{-1/2}\), and
\(q_s=H_S^{-1/2}g_s\), this gives

\[
H_S^{1/2}\delta_{j,\lambda}=-\lambda(I+\lambda K_j)^{-1}b_j,
\]

and

\[
S_{j,s}(\lambda)=-\lambda q_s^T(I+\lambda K_j)^{-1}b_j.
\]

Pure curvature is measured by

\[
\rho_{j,s}(\lambda)=
\frac{q_s^T(I+\lambda K_j)^{-1}q_s}{q_s^Tq_s}.
\]

Its derivative at zero is \(-c_{j,s}\), where
\(c_{j,s}=q_s^TK_jq_s/(q_s^Tq_s)\).  If \(K_jq_s=0\), the ratio is exactly
one.  For PSD \(K_j\), the ratio is nonincreasing and converges to the
normalized projection onto \(\ker K_j\).  These are response-level quadratic
theorems, not semantic mechanism claims.
