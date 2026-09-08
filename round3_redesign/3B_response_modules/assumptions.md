# 3B Assumptions

This track starts from the frozen 3A/3A-T source Hessian and first-order
shift gradients. It studies response modules, not latent semantics.

The primary response for a shift `s` is
\[
u_s=H_S^{-1/2}g_s,
\]
and the intrinsic Gram matrix is `G_ij = g_i^T H_S^{-1} g_j`.

The benchmark uses population Gaussian moment states. Source fitting is done
before any target probes are inspected. Mechanism, intervention-family and
exposure labels are post-hoc metadata only.

First-order-null probes are filtered from the primary discovery analysis and
reported as a separate boundary. A module is a stable subspace of response
vectors; its dimension is not asserted to equal a semantic factor count.
