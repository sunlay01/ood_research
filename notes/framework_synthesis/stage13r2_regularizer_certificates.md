# Stage 13R.2: Regularizer-to-Transfer Certificates

## Fixed endpoint and state

The endpoint is the established one-sided excess-risk transfer measure

```text
T_Gamma(S || T) = sup_f [E_T(f)-E_S(f)].
```

The source state is the finite-dimensional excess-risk representation from the
companion note, with one externally declared target family `U` and support
`h_U(g)=sup_{delta in U}g(delta)`.

## Certificate table

| Method | Source statistic | Transfer certificate under its declared family | Status |
|---|---|---|---|
| ERM | source mean excess risk | no control of `h_U(g_f)` beyond the external family assumption | `NO-CONTROL` |
| V-REx | `(1/m)sum_e g_f(delta_e)^2 = Var_e E_e(f)` | `T^U <= rho sup_f sqrt(Var_e E_e(f)) + 2 eps` for exposed ellipsoid `U` | `EXACT-STATISTIC / CONDITIONAL-CERTIFICATE` |
| GroupDRO | `max_e E_e(f)` | exact support for `U=conv{delta_e}`; no guarantee outside observed hull | `EXACT-UNDER-HULL` |
| MM-REx | bounded affine coefficient support | exact support for declared bounded affine extrapolation family | `EXACT-UNDER-FAMILY` |
| fixed-state MMD | state displacement norm | dual norm upper bound on `h_U(g_f)` for a fixed RKHS witness and radius | `UPPER-BOUND` |
| CORAL | covariance/second-moment projection | controls only the projected component; conditional residual remains | `PROJECTION-RELAXATION` |
| ideal IRM | common source optimum/mechanism restriction | shrinks admissible target family/fiber under structural identifiability | `SURROGATE-WITH-CONDITIONS` |
| IRMv1/Fishr | parameter gradient or gradient covariance | not certified in the minimal excess-risk state | `NO-CONTROL` |

The left-hand endpoint is unchanged across all rows. A method-specific target
family is not silently substituted for the fixed family; when a row is exact,
the exactness is explicitly restricted to the family named in that row.

## V-REx certificate derivation

For exact affine excess risk,

```text
E_e(f)-E_barS(f)=g_f(delta_e),
Var_e E_e(f)=g_f(A g_f).
```

For `U_rho={delta in S: delta^T A^dagger delta <= rho^2}`, dual Cauchy--Schwarz
gives

```text
sup_{delta in U_rho} g_f(delta) = rho sqrt(g_f(A g_f)).
```

Therefore

```text
T_Gamma^{U_rho}(S)
  <= rho sup_{f in Gamma} sqrt(Var_e E_e(f)) + 2 epsilon_repr.
```

The supremum over predictors is important: a per-predictor V-REx penalty is a
certificate for the selected predictor, while a uniform transfer measure over
`Gamma` needs a uniform bound or a restricted candidate class.

## GroupDRO and MM-REx

For `U_conv=conv{delta_e}`, affine transfer gives

```text
sup_{delta in U_conv} g_f(delta) = max_e g_f(delta_e),
```

and hence the finite-group robust transfer quantity is the worst source excess
risk. If the target is outside the convex hull, this identity says nothing about
that target.

For

```text
B_r={beta: 1^T beta=0, ||beta||_2<=r},
delta(beta)=sum_e beta_e delta_e,
```

the affine support is `r ||(g_f(delta_e))_e - mean||_2`, matching the declared
MM-REx extrapolation family. Clipping/nonnegativity constraints change the
support and must be stated explicitly.

## MMD/CORAL/IRM limitations

Fixed-state MMD bounds the dual action of `g_f` only for the fixed witness class
and target radius; marginal MMD does not control conditional label shifts.
CORAL observes a covariance projection in the supervised state, so a label or
cross-moment shift can leave its penalty unchanged while changing transfer risk.
Ideal IRM may reduce the compatible mechanism fiber, but this requires a
structural identifiability theorem. These are useful theorem interfaces, not
unconditional exact translations.

## Relation to prior work

Zhang et al. 2021 already define transfer measure and a target-error bound.
Hemati et al. ICCV 2023 explicitly analyze Hessian/gradient alignment as upper
bounds on transfer measure and discuss CORAL, IRM, V-REx, Fish and Fishr.
Chen et al. UAI 2025 extend transfer measure to multiple sources and prove
parameter-side moment-alignment bounds. The present package does not repeat
those results; it adds the source-observability quotient, sharp certificate
minimality, and blind-direction impossibility at the transfer-measure endpoint.
