# Round 3 3A-T: Theory Closure Report

## Final verdict

`3A-T-THEORY-AND-FORMAL-PASS`

The deterministic population quadratic theory is closed under its explicit
positive-definite assumptions. The independent SymPy checks pass, the
existing trust-region and relevance regression tests pass, and the smooth
extension is proved with the required source-sublevel localization condition.
The deterministic quadratic core was independently formalized in Lean 4 with
mathlib. All retained theorem modules compile without placeholders.

## Theorem status

The following are `theorem` or `proposition` results in the mathematical
record: exact source ellipsoid, exact shift polynomial, ellipsoid support,
finite-budget bound, first-order-null scaling, residual coupling, source-use
counterexample, common-burden proposition, independent nuisance leading
invariance, and source-metric coordinate invariance.

The smooth leading asymptotic is a theorem under `C^2` source regularity,
positive-definite Hessian, shift differentiability, and localization of the
global source sublevel sets. The `O(epsilon)` refinement is conditional on
local quadratic shift control and the stated source remainder. The singular
source case remains a boundary analysis, not a theorem of the main package.

No claim here is a causal mechanism theorem, a finite-sample confidence bound,
or a deep-network theorem.

## Proof dependency graph

\[
\text{Source Ellipsoid}
\rightarrow
\text{Ellipsoid Support}
\rightarrow
\text{Leading Relevance}
\]

\[
\text{Shift Polynomial}
+\text{Ellipsoid Support}
+\text{Quadratic Curvature}
\rightarrow
\text{Finite-}\epsilon\text{ Bound}
\]

\[
g_s=0\rightarrow\text{Exact }\epsilon\text{-Scaling}
\]

\[
\text{Moment Residual Identity}
+\text{Source Normal Equation}
\rightarrow\text{Residual Coupling}
\]

\[
\text{Block-Separable Source Geometry}
+\text{Zero Nuisance Gradient}
\rightarrow\text{Nuisance Leading-Invariance}
\]

## Main mathematical conclusions

With `H_S=2M_S`, the implementation's scalar and the theory's scalar are
related by
\[
L_s=\sqrt{g_s^TM_S^{-1}g_s}
=\sqrt{2}\,\texttt{leading\_relevance}.
\]
This is the critical normalization audit; no factor-of-two mismatch remains.

The leading source-risk-neighborhood discrimination is
\[
\sqrt{2\epsilon}\sqrt{g_s^TH_S^{-1}g_s}.
\]
The exact finite-budget correction is controlled by
\[
K_s=\|M_S^{-1/2}A_sM_S^{-1/2}\|_{op}
\]
for symmetric `A_s`. If `g_s=0`, vulnerability is exactly `epsilon K_s`,
not necessarily zero.

Independent nuisance augmentation leaves the leading dual quadratic form
unchanged when the source matrix is block diagonal and the target residual
gradient has a zero nuisance block. It can still change finite-budget
vulnerability through the target quadratic block `A_s`.

## Adversarial boundaries closed

- A local positive Hessian alone is insufficient for a global smooth
  sublevel statement; localization was added explicitly.
- `epsilon=0` and `g=0` use the zero witness, not the nonzero witness formula.
- Positive definiteness is required for compact ellipsoid support and finite
  inverse-metric relevance.
- `w_j^*=0` does not imply `g_{s,j}=0`; the audited two-dimensional example
  has `w*=(1/2,0)` and `g=(0,-1)`.
- A common target-only constant can make burden nonzero while vulnerability
  is exactly zero.
- In the singular case, kernel directions can have zero source cost and
  nonzero target response; pseudoinverse formulas require extra restrictions.

## Independent checks

`symbolic_checks.py` passed all seven checks: source excess, shift
polynomial, residual coupling, source-use counterexample, common burden,
block nuisance invariance, and the `M` versus `H` factor-of-two relation.

The targeted theory-closure and previous 3A relevance tests passed: `24
passed`. The full repository suite also passed: `139 passed`. The theory
track does not modify existing Python interfaces.

## Lean status

`LEAN-CORE-PASS`. Lean 4.33.1 and mathlib v4.33.1 are pinned by toolchain and
exact manifest commits. `lake build OODRelevance` compiled the source-excess
and shift identities, exact Euclidean support maximum after whitening,
first-order-null scaling, arbitrary-dimensional direct-sum nuisance
invariance, and the common-burden/source-usage counterexamples. Six project
`.olean` files were produced. No retained source contains `sorry`, `admit`, a
custom axiom, or a `True` placeholder.

## Frozen scope

This track does not establish source-exposure canonicalization, mechanism
semantics, clustering, response-order structure, regularizer control, deep
network transfer, or finite-sample generalization. Those remain outside the
3A-T closure.
