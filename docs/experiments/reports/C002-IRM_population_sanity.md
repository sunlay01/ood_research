# CAL-001: C002-IRM Population Sanity Check

## Purpose

Independently evaluate the analytic C002-IRM counterexample using population
moments, rather than sampled training or target data.

## Predeclared Model

\[
L=\beta=1,\quad \operatorname{Var}(\xi)=2,
\quad \operatorname{Var}(\eta)=1/50,
\quad \sigma_Y=1/10,
\]

with source nuisance relations \(0.7\) and \(-0.1\), and an admissible target
relation \(-1\) in the unit correlation ball around zero.

## Command

```bash
PYTHONPATH=src python -m ood_repr_reg.intervention_report
```

## Expected Result

- source ERM: `(0, 0.25, 0.833333...)`;
- source risk equals source observational oracle risk `0.51`;
- standard scalar-scale IRMv1 penalty is numerically zero;
- target causal excess equals `127/48`;
- full gradient-alignment penalty is strictly positive.

## Observed Result

The command reproduced all expected values.  The full test suite also verifies
the exact transport identity, both intervention-ball formulas, the vector
source-unobservability construction, and this counterexample.

## Decision

`PASS_THEOREM_SANITY_CHECK`.  This supports the internal algebra only.  The
literature audit classifies the standalone IRMv1 result as high-risk prior-art
overlap, so it is not a paper claim.
