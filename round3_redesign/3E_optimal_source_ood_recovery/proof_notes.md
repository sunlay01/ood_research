# Proof and Audit Notes

The minimax lower bound is a two-world argument and therefore applies to an
arbitrary deterministic estimator, not merely a linear estimator.  Compactness
of the unit sphere in `ker O` supplies a maximizer in finite dimensions.  If
`ker O` is zero, the convention is `alpha=0` and the pseudoinverse recovery is
exact.

The implementation computes `alpha` from the orthonormal null-space basis
`N`, as `sigma_max(A N)`.  The dense projector formula `||A N N^T||_op` is an
independent numerical audit only.  The pseudoinverse residual is checked as
`A(I-O^dagger O)u`, not by fitting a recovery map on target data.

The 3D Gaussian pair is independently reconstructed from distinct plus/minus
target records.  Its reported equality to `2 alpha` is only for the declared
one-dimensional embedding whose unit perturbations are those two worlds; it
does not make a metric-free claim that an arbitrary raw world pair is global
extremal.

Lean formalizes the finite linear kernel/factorization core in the shared
workspace.  The arbitrary-deterministic minimax proof and the pseudoinverse
operator-norm equality remain paper proofs because formalizing the required
finite-dimensional norm and function-space APIs is deliberately outside this
small core.

The coupled numerical audit does not alter the abstract theorem.  It declares a
metric, then differentiates the frozen 3D observable state to obtain `O` and
the frozen 3A source-whitened response to obtain `A`.  The finite-difference
step-stability audit is independent of target risk; target environments enter
only to define the response Jacobian, never the source observation map.  The
three 3B/3C objects excluded from the construction are semantic clusters,
oracle labels, and regularizer-control geometry.
