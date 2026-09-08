# Analytic Derivatives

For `R_w(t)=w^T M_XX(t) w - 2 w^T m_XY(t) + m_Y2(t)`, the implementation expands every
structural parameter as an affine polynomial in `t`. Products such as
`Gamma(t) Sigma_C(t) Gamma(t)^T` and `mu_A(t) mu_A(t)^T` are then expanded exactly. The
coefficient of `t^k`, multiplied by `k!`, is the directional derivative `D^k R_w[v^k]`.

This is a polynomial calculation, not an empirical regression of derivatives. The resulting
lifted statistic vector is paired with the same 21-dimensional model lift as in 3A.

