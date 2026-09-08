# Source-induced family

Given observed source environments `e_i`, form task-complete states
`psi(e_i)=(svec(M_i),m_i,c_i)` and contrasts `psi(e_i)-psi(e_0)`.  An SVD
provides an orthonormal state-span basis.  At the reference environment, a
finite-difference parameter Jacobian `J_psi` is used to compute

`delta_eta = J_psi^dagger d`.

The residual `||J_psi delta_eta-d||` is recorded for every mode.  Only modes
within the declared realization tolerance enter the family tangent; any other
mode is listed as unrealizable.  This separates state-span rank from legal
realizable rank.

The default source-induced family has four modes for the relation-exposed
two-shortcut source design.  Duplicating a source environment does not increase
the span.  A new mean contrast can increase raw source-state rank.  A
risk-null noise contrast can also increase raw state rank without adding a
response direction.  These facts describe the uncertainty family, not a claim
that the source-induced family is universally more robust.
