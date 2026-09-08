# Audit: C002-IRM Scalar-Scale Blind Direction

## Claim

The standard scalar-scale IRMv1 penalty does not calibrate robust causal excess
over the C001 correlation ball, even when source observational excess is zero.

## Correctness

`PASS`.

- The source ERM, zero IRMv1 derivatives, and target excess are exact rational
  calculations recorded in `docs/theory/claims/C002_irmv1_blind_direction.md`.
- The implementation independently evaluates population moments and is covered
  by an executable unit test.
- Since source ERM also has nonnegative penalty zero, it is a global minimizer
  of `R_S + lambda * Omega_IRMv1` for every `lambda >= 0` in the fixed identity
  representation class.

## Assumption Scope

`MAJOR LIMIT`.

The result is for population squared loss, affine prediction, identity
representation, two source environments, and a bounded conditional-correlation
intervention family.  It refutes a general certificate claim for a class that
contains this setting; it does not characterize all learned-representation
implementations of IRMv1.

## Prior-Art Audit

`HIGH-RISK COLLISION`.

- Kamath et al., AISTATS 2021, prove that practical linear IRM/IRMv1 can fail
  even at population level and can generalize worse than ERM in simple
  Colored-MNIST-like environments.
- Their checked construction is a two-bit discrete environment analysis, not
  the present continuous Gaussian risk-transport identity or the exact
  source-ERM/zero-penalty/correlation-ball equality.
- This distinction is insufficient to claim a new IRMv1 failure result without
  a complete citation-chain audit.  The counterexample is retained as a
  necessary negative control and a falsification of the proposed certificate.

## Claim Inflation Check

`FAIL` for paper novelty, `PASS` for internal theory.

Do not present C002-IRM alone as a new IRMv1 theorem.  Its project value is to
rule out the positive-certification branch and force any future positive result
to use a different regularizer, a stronger intervention/observability premise,
or a different notion of calibration.

## Recommendation

`KILL` the standalone IRMv1 certificate route.  Keep C001, C004a, and C002-IRM
as the audited baseline against which any later regularizer-specific result is
measured.
