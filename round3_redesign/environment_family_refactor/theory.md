# Family-relative geometry

Let `Theta` be a declared environment family and let `theta` be its reference
point.  The family tangent is the finite-dimensional space `U_theta` with the
metric in `TangentSpec`.  For the task-optimal response map `q` and the
source task-state map `s_S`,

`A_theta = D q(theta)` and `O_S,theta = D s_S(theta)`.

The source-only information floor is

`1/2 || A_theta P_(ker O_S,theta) ||_op^2`.

For a source-adaptive local learner with `Pi_theta = D_y z_theta`, the affine
policy is

`z(u) = z0 + (A_theta + Pi_theta O_S,theta) u`.

These statements are conditional on the declared family and its metric.  If a
shift is outside the tangent, it is family omission; if it lies in the tangent
and is source-null but response-active, it is source-information failure; if it
is source-visible but the learner response is wrong, it is algorithm failure.

The source-reference span is invariant because replacing a reference subtracts
one source state from every column without changing the affine difference
span.  The source-induced realization uses a least-norm Jacobian pullback and
reports any residual instead of treating an unrealizable state mode as legal.

Under an invertible coordinate recoding, `A`, `O_S`, and the metric must be
transported together.  Otherwise the uncertainty set has changed, rather than
only its coordinates.
