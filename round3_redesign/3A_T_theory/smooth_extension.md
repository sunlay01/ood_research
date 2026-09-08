# Smooth Extension Proof

Let `R_S` be `C^2` near `w*`, with zero gradient and positive-definite
`H=H_S`. Assume also that the global source sublevel sets localize at `w*`:
for some neighborhood `U` of `w*`, all sufficiently small source sublevels
are contained in `U`. This assumption is necessary because a merely local
positive Hessian does not exclude a distant point with the same source risk.
Continuity of the Hessian gives, for every sufficiently small
`eta>0`, a radius `r_eta` such that for `||delta||<r_eta`,
\[
\frac12(1-eta)\delta^TH\delta
\le R_S(w^*+\delta)-R_S(w^*)
\le\frac12(1+eta)\delta^TH\delta. \tag{1}
\]

This is the required local sublevel-set sandwich. Since the smallest
eigenvalue of `H` is positive, (1) implies every sufficiently small feasible
delta has norm `O(sqrt(epsilon))`.

Assume `Delta` is differentiable at `w*`. Then
\[
\Delta(w^*+\delta)-\Delta(w^*)=g^T\delta+o(||delta||).
\]
For the upper bound, localization puts every feasible perturbation inside the
neighborhood where (1) holds. Apply the differentiability remainder uniformly
on the shrinking feasible set and use the upper support of the ellipsoid from
the sandwich. This gives
\[
V(epsilon)\le\sqrt{2epsilon/(1-eta)}\sqrt{g^TH^{-1}g}+o(sqrt(epsilon)).
\]

For the lower bound when `g != 0`, let
`d=H^{-1}g/sqrt(g^T H^{-1}g)`. The point
`delta_t=t d` has quadratic source cost `t^2/2`. Choose
`t=sqrt(2epsilon)(1-o(1))` so that the upper side of (1) makes it feasible.
The radial factor can be chosen so the point is feasible by the upper side of
(1), while it tends to one as `epsilon` tends to zero. The shift expansion
then gives the matching lower limit. Letting `eta` tend
to zero proves
\[
V(epsilon)=sqrt(2epsilon)sqrt(g^TH^{-1}g)+o(sqrt(epsilon)).
\]
If `g=0`, differentiability alone yields only `o(sqrt(epsilon))`. If the
shift has locally bounded Hessian, Taylor's theorem gives a uniform
`O(||delta||^2)` remainder, and the source sandwich gives `V(epsilon)=O(epsilon)`.

If the source and shift remainders are respectively `O(||delta||^3)` and
`O(||delta||^2)`, the same feasible-direction argument and the source radial
correction give the sharper `O(epsilon)` error around the leading term.

Thus `C^2` source regularity alone does not justify the stronger finite-budget
remainder; the shift's second-order control is an additional assumption.
