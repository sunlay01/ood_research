# Nonlinear Stress Family

The helper `nonlinear_environment_state` evaluates exact second moments for
\(S_j=\rho_jY+\alpha_j(Y^2-1)+\varepsilon_j\).  It is a stress calculation
for the moment pipeline, not an extension of the linear-Gaussian theorem.
The identities \(E[Y(Y^2-1)]=0\) and \(E[(Y^2-1)^2]=2\) are used explicitly.

Any failure of additive response under a nonlinear parameterization must be
reported as an interaction residual.  No nonlinear result is used to assign
mechanism names.
