# C/K factorial diagnostic

The CMNIST local C/K pilot completed for IRMv1, V-REx and Fishr at step 300, but it fails the local trust-region calibration gate. A single absolute epsilon in an 8-D random parameter subspace produced enormous C-driven displacements for IRMv1 (up to 2.9e5 functional norm) and V-REx (2.7e3), while K-driven effects were orders of magnitude smaller. This indicates that the projected operator estimates and step scaling are not commensurate across methods; the resulting dominance pattern is numerical, not evidence for forcing dominance.

Fishr has comparable C and K magnitudes, but the interaction changes across seeds. No forcing/filtering conclusion is accepted.

Required correction before scientific interpretation: normalize each cell to a common parameter trust-region norm (or use a whitened Fisher/Hessian metric), verify finite-difference linearity at epsilon/2 and epsilon/4, and reject near-singular projected Hessians. Then repeat h=1 before any h>1 rollout.
