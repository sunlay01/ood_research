# Singular Source Geometry Boundary

This is a boundary analysis, not an extension of the main theorem package.

If `M_S` is only positive semidefinite, the source excess identity remains
\[
R_S(w^*+\delta)-R_S(w^*)=\delta^TM_S\delta,
\]
but the source-good set has flat directions `ker(M_S)`. Source optima are
generally nonunique: adding a kernel vector preserves source risk whenever
the normal equation is solvable.

If the target linear response has a nonzero component on `ker(M_S)`, then a
zero-source-cost direction can produce nonzero target differentiation at no
source budget. With an unrestricted parameter domain this can make the
finite-budget supremum infinite or at least `O(1)` for every positive budget,
depending on the target quadratic term and parameter constraints.

The pseudoinverse expression
\[
g^TM_S^+g
\]
is meaningful only after restricting perturbations to `Range(M_S)` or
requiring `g` to annihilate `ker(M_S)`, equivalently
`g in Range(M_S)`. Even then it describes the range component and does not
control arbitrary kernel motion. A complete singular theory therefore needs
an explicit quotient, minimum-norm source optimizer, or a parameter constraint
that removes the flat directions.
