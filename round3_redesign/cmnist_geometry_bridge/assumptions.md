# Assumptions

This track is an empirical finite-sample bridge, not a proof that the
Gaussian population theorems hold for neural networks.  The primary model is
a learned ERM encoder followed by a frozen-representation, trainable linear
head and empirical squared loss.

Color probabilities are integrated analytically over a fixed grayscale bank.
This removes Bernoulli resampling noise from finite differences.  The target
distribution is used only for post-hoc oracle geometry and accuracy.

The source-induced family contains only source correlation directions.  The
mechanism-defined family adds a declared target-admissible hidden correlation
direction.  An exposed version changes the source design for that same
direction.  Norms are conditional on the declared Euclidean tangent metric
and source-risk Hessian response metric.
