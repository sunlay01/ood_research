# TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT Preregistered Design

The audit freezes each verified nonlinear encoder and re-optimizes only its
65-dimensional final linear head from the saved head initialization.

- Methods: `ERM`, `IRMv1`; seeds: `10..14`.
- Source tangent basis: env0 color flip, env1 color flip, shared label noise.
- Report directions: env0, env1, label, common color, antisymmetric color.
- Centered probability finite differences: `0.01`, `0.02` with common random numbers.
- Source-only quantities: refined head, `H_S`, `O_S`, `Pi`, all gates.
- Held-out evaluation quantities: `A` and post-hoc finite-response checks only.
- Main metric: `||A + Pi O_S||_F`; all thresholds are in `provenance.json`.

Interpretation ceiling: this is a frozen-encoder head-block local response
audit. It is not a full-network response, new algorithm, causal claim, or
universal DG result.
