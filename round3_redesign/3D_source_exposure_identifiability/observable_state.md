# Observable State

For `X` including the intercept, define

`psi_e = (svec(M_e), m_e, c_e)`, where `M_e=E[X X']`,
`m_e=E[X Y]`, and `c_e=E[Y^2]`. The `svec` convention weights every
off-diagonal entry by `sqrt(2)`, so Euclidean products equal Frobenius
products.

For source optimum `w*` and `H_S=2 M_S`, a state difference produces

`q(delta) = H_S^(-1/2) (2 Delta M w* - 2 Delta m)`.

The scalar `c_e` is retained for exact risk reconstruction but is in the
kernel of the predictor gradient map. It may therefore increase observable
state rank without increasing response exposure.
