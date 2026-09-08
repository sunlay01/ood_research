# 3E-B Joint Information-Regularization Affine Regret

## Verdict

`3E-B-JOINT-AFFINE-PASS`

The primary pair is the coupled 3A/3D tangent.  3A supplies the response
metric and 3D supplies the source observation operator.  The source-hidden
emergent `U` direction is retained in the primary benchmark; a U-exposed
observation is a separately labelled information intervention.

## Exact finite-dimensional object

For `L_u(z)=1/2||z||^2 + (A u)^T z`, a concrete regularizer contributes the
affine policy `z0 + (A + Pi O_S)u`.  Its reported total regret is
`1/2 sup_(||u||<=1)||z0+(A+Pi O_S)u||^2`.  The irreducible information floor is
`1/2 ||A P_ker(O_S)||_op^2 = 0`.  `total_excess` in the CSV is
absolute affine regret minus that floor, clipped at zero only for roundoff;
it is not target-risk excess.

## Numerical audit

- methods and valid lambdas: 15
- coupled world tangent: `4D`
- source observation shape: `(275, 4)`
- primary hidden-U floor: `0`
    - U-exposed floor: `n/a`
- factorization residual: `2.91e-17`
- maximum IFT finite-difference error: `2.27e-08`
- maximum secular versus independent constrained optimizer error: `2.83e-07`

The secular-equation solver includes the repeated-top-eigenvalue hard case and
is independently checked by multi-start constrained optimization.  Every
concrete policy in the primary table is above the information floor.

## Decomposition and limits

Static-only, adaptive-only, and interaction values are counterfactual
diagnostics and are not assumed additive.  The recoverable residual reports
the response action remaining after the source-invisible component is removed;
it does not assert target-risk control.  The same `(b,K)` can coexist with
different `Pi`, so frozen curvature alone does not determine the joint affine
action.

No 3B semantic module, 3C semantic/control assumption, target-risk oracle,
causal interpretation, finite-sample guarantee, or universal DG theorem is
used.  The Lean extension remains `LEAN-PARTIAL` because IFT, pseudoinverse
operator norms, and general trust-region maximization are not claimed as
formally verified here.
