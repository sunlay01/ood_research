# Optimal Source-Only Recovery

Let `O : U -> Y_S` and `A : U -> R` be linear maps between finite-dimensional
real Hilbert spaces.  Put `P_0 = P_{ker O}` and

\[
\alpha(A,O)=\lVert A P_0\rVert_{\rm op}.
\]

For every deterministic recovery rule `Phi : Im(O) -> R`, including nonlinear
rules,

\[
\sup_{\lVert u\rVert\leq1}\lVert Au-\Phi(Ou)\rVert\geq\alpha(A,O).
\]

Choose a unit right-singular maximizer `v in ker O` of `A|_{ker O}`.  The two
worlds `v` and `-v` have the same source observation, so the triangle
inequality gives

\[
2\max\{\lVert Av-\Phi(0)\rVert,
\lVert-Av-\Phi(0)\rVert\}
\geq 2\lVert Av\rVert=2\alpha.
\]

The Moore--Penrose rule `Phi*(y)=A O^dagger y` attains the bound because

\[
Au-\Phi^*(Ou)=A(I-O^\dagger O)u=A P_0u.
\]

Therefore

\[
\inf_{\Phi:\operatorname{Im}O\to R}
\sup_{\lVert u\rVert\leq1}\lVert Au-\Phi(Ou)\rVert
=\alpha(A,O)=\sigma_{\max}(AN),
\]

where the columns of `N` are any orthonormal basis of `ker O`.

## Consequences

* `alpha = 0` iff `ker O subseteq ker A` iff `A = A_tilde O` for a linear
  `A_tilde` on `Im(O)`.
* The source-equivalent unit-ball ambiguity diameter is `2 alpha`.
* If `ker O_2 subseteq ker O_1`, then `alpha(A,O_2) <= alpha(A,O_1)`.
* An invertible recoding of source observations leaves `ker O`, hence
  `alpha`, unchanged.  World and response coordinate changes preserve it only
  when they are isometries (or when the declared metrics are transformed too).
* Rank, source count, and structural ambiguity dimension do not determine
  `alpha`; the orientation and magnitude of `A` on the invisible subspace do.

This is a source-response recovery result.  It does not state a target-risk
lower bound, causal identification result, latent-factor recovery result, or a
universal domain-generalization theorem.

## Coupled 3A/3D instantiation

The primary 3E audit fixes an eight-dimensional standardized environment-world
tangent.  `O` is the stack of source-only central-difference derivatives of
the frozen 3D task-complete states.  `A` is the central-difference Jacobian of
the frozen 3A source-whitened response map at the declared relation-exposed
source reference.  Both maps act on the same standardized coordinates.

The hidden-emergent `U` coordinate is intentionally source-null and
response-active in the primary design.  A separate U-exposed source
observation is introduced only in the nested information ladder.  This makes
the predicted decrease in `alpha` an information-geometry statement with
fixed `A`, rather than a change in the response target.  Duplicate and
risk-null/noise additions are included as non-improving controls.  The final
identity observation is explicitly a mathematical full-information endpoint,
not an attainable source-design claim.
