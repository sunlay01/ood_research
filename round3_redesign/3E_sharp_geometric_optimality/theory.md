# 3E-C theory

Let `P` project onto `ker(O_S)` and `Q=I-P`. Define
`A_irr=A P`, `A_rec=A Q`, and `E=A_rec+Pi O_S`. Then `E P=0` and
`A_irr Q=0`, while `P O_S^*=0`. Hence the cross operators vanish and
`(A_irr+E)(A_irr+E)^*=A_irr A_irr^*+E E^*`.

Writing `alpha=||A_irr||` and
`S_slack=alpha^2 I-A_irr A_irr^*`, the adaptive-only affine policy reaches
the source-information floor exactly when `E E^* <= S_slack`. The full policy
reaches it exactly when this condition also holds and `z0=0`. If `alpha=0`,
the slack is zero, so adaptive optimality requires `E=0`.

The static steering tax follows from evaluating the policy on the two
indistinguishable worlds `v` and `-v` attaining the top invisible singular
direction. Their average squared response is `||z0||^2+||Av||^2`, so the
larger of the two is at least `||z0||^2+alpha^2`; after multiplying by one
half this gives `R_j >= R_info + 1/2||z0||^2`. This is a symmetric
uncertainty-ball lower bound, not a universal DG-risk statement.

The iff proof has no hidden base-point recentering: `z0` is the actual affine
offset and `E` is the complete recoverable residual. Finite-dimensional
compactness supplies a unit `v` attaining `||A P||_op`. If `alpha=0`, the
slack operator is zero and `E E* <= 0` is equivalent to `E=0`.
