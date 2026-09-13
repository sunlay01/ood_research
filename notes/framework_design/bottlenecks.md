# Mathematical bottlenecks

## B1. Empirical regularizer to population bridge

For MMD, CORAL, IRMv1 and Fishr, the unresolved statement is an explicit high-probability inequality of the form

`Omega_hat_j(f) <= eps  =>  B_j(f) <= psi_j(eps, n, E)`.

The required witness or derivative class must be measurable and have finite complexity. This is not currently proved for unrestricted deep objectives.

## B2. Bridge object to target risk

For marginal alignment the exact decomposition contains a conditional-label term. A theorem claiming `R_T-R_S <= phi(MMD)` without a conditional assumption is contradicted by the binary channel-swap construction. The missing relation is a conditional/invariant mechanism theorem, not a better MMD constant.

## B3. Source exposure to target-family control

Finite sources do not identify arbitrary target conditionals or meta-law tails. The unresolved quantity is either a coverage/radius assumption, a fresh-domain quantile, or an impossibility lower bound. Removing it changes the problem.

## B4. Algorithm fidelity

IRMv1 differentiates only a classifier coordinate; Fishr uses per-example gradient covariances; MLDG uses a bilevel update. The missing relation to ideal IRM, causal invariance or target risk requires separate translation and optimization-error terms.

## B5. Complexity and tightness

Any positive theorem must show finite loss/witness/derivative complexity and be non-vacuous on a simple construction. The two-world shortcut example is a required tightness and falsification check.
