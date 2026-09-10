# TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR Preregistered Design

This repair invalidates the old audit interpretation: it omitted the source term in A, used thresholded finite differences, made O method-dependent, and treated a new head-only equilibrium as the successful full learner.

It also supersedes repair commit `b6c9eaf`: that run represented theta as absolute probabilities and then added the base probabilities a second time, producing invalid source/evaluation mixtures including `p=1.1`.

- Primary tangent basis: `source_env0_color`, `source_env1_color`, `shared_label_noise`.
- Coordinates are displacements from source `(0.2,0.1)`, evaluation `(0.9,0.9)`, label noise `0.25`; G-1 validates the zero-displacement world and every mixture weight.
- Primary worlds: smooth four-outcome empirical expectations; no thresholded world is a primary derivative.
- G0 tangent-linearity tolerance: `0.02`; G0 failure stops the audit.
- Primary O: concatenated source risk gradients only, method-independent by construction.
- Pi_head: H0 is diagnostic only; H1 is a separate frozen-encoder equilibrium diagnostic.
- Pi_full: exact Adam-state reconstruction, smooth source-only continuation, delta `0.01`, K `1,5,20`, common continuation batches and replay.
- Full mismatch: `NOT_ESTABLISHED`; no scalar surrogate is invented.
