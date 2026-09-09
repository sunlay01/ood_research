# Representation Protocol

Five ERM CNN representation seeds are trained with the existing CMNIST
generator and source-only schedule.  The encoder is then frozen.  A fixed
source-training grayscale bank is encoded as paired red and green representations.  The bank
is reduced to its empirical non-degenerate support by an SVD; this records and
removes exact/near-dead feature coordinates rather than adding damping.

The bridge head uses `x_phi=(1,phi(x))` and empirical squared loss.  Full CNN
cross-entropy results in `CMNIST-VIS-001` remain a separate exploratory audit.
