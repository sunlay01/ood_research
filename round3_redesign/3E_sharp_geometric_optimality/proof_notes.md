# 3E-C proof notes

These notes state the finite-dimensional proof used by the Python audit. All
operators are real matrices with compatible finite dimensions. The uncertainty
set is the Euclidean unit ball and the response loss is
`R(z; B) = 1/2 ||z + B u||^2`.

## Projection decomposition

Let `P` be the orthogonal projector onto `ker O_S` and let `Q = I - P`.
Because `P` projects onto the kernel, `O_S P = 0`. Taking adjoints gives
`P O_S^* = 0`. Define

`A_irr = A P`, `A_rec = A Q`, and `E = A_rec + Pi O_S`.

Then `E P = A Q P + Pi O_S P = 0`, while `A_irr Q = A P Q = 0`.
Consequently

`A_irr E^* = A P (A Q + Pi O_S)^* = 0`,

because `P Q = 0` and `P O_S^* = 0`. Taking adjoints also gives
`E A_irr^* = 0`. Hence

`(A_irr + E)(A_irr + E)^* = A_irr A_irr^* + E E^*`.

This is an operator cross-term identity. It does not say that the two
response image subspaces have a semantic interpretation or are orthogonal in
an arbitrary ambient representation.

## Adaptive-only sharp condition

Put `alpha = ||A_irr||_op` and
`S = alpha^2 I - A_irr A_irr^*`. Since the largest eigenvalue of
`A_irr A_irr^*` is `alpha^2`, `S` is positive semidefinite. For `z0 = 0`,

`R_adap = 1/2 ||A_irr + E||_op^2`

and the information floor is `R_info = alpha^2/2`. The cross-term identity
implies

`R_adap <= R_info`
iff
`(A_irr + E)(A_irr + E)^* <= alpha^2 I`
iff
`E E^* <= S`.

The reverse inequality `R_adap >= R_info` follows by restricting the input
to `ker O_S`: there `E u = 0` and the policy is `A_irr u`. Therefore the
inequality is an equality exactly under the displayed PSD condition.

If `alpha = 0`, then `A_irr = 0` and `S = 0`; `E E^* <= 0` is equivalent to
`E = 0`. This is the complete-information degeneration.

## Generalized slack ratio

Diagonalize the PSD operator as `S = U diag(s_i) U^*`. If the range of `E`
is not contained in the support of `S`, the comparison is not finite and the
reported ratio is `+infinity`. Under support compatibility, define

`rho_slack = || S^{dagger/2} E ||_op^2`.

Then `E E^* <= S` iff `rho_slack <= 1`. This follows after conjugating the
PSD inequality by `S^{dagger/2}` on the support; the nullspace component is
zero by compatibility.

## Static steering tax and full iff

In finite dimensions the operator norm of `A_irr = A P` is attained. Choose a
unit `v` in `ker O_S` with `||A v|| = alpha`. The worlds `v` and `-v` have
the same source observation. For any fixed offset `z0`, their squared losses
are

`1/2 ||z0 + A v||^2` and `1/2 ||z0 - A v||^2`.

Their average is `1/2 ||z0||^2 + 1/2 alpha^2`, so the larger one is at least
`R_info + 1/2 ||z0||^2`. This is the static steering lower bound. It is tight
for aligned examples, and can be strict when the recoverable component adds
an orthogonal or otherwise adverse response.

If the full policy has worst-case value equal to `R_info`, the lower bound
forces `z0 = 0`. The adaptive-only equivalence then forces `E E^* <= S`.
Conversely, `z0 = 0` and that PSD condition give equality. Thus

`R_full = R_info` iff `z0 = 0` and `E E^* <= S`.

The proof uses only compactness/attainment of a finite-dimensional unit sphere
and the PSD characterization of the operator norm. It does not use target
risk, semantic labels, or regularizer exposure containment.

## Counterexample boundaries

`E = 0` is sufficient but not necessary: a nonzero `E` can fit inside the
spectral slack. Conversely, `E = 0` does not rescue a nonzero `z0`, because
the static tax remains. Equal frozen quadratic `(b, K)` pairs need not have
equal exact source Jacobians or affine actions. These distinctions are
recorded in the generated counterexample file rather than collapsed into a
single regularizer ranking.

## Numerical status

The Python implementation uses orthogonal null-space bases and an exact
finite-dimensional shifted-ball solver for the reported regrets. The
orthogonality, PSD gaps, theorem fixtures, legal environment search, and
small-lambda diagnostics are independently checked by the dedicated pytest
file. Lean formalizes the compact algebraic core only; operator norms,
top-eigenvalue arguments, and full PSD-order APIs remain explicitly marked
`LEAN-PARTIAL`.
