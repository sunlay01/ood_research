# Baseline Fidelity

Primary protocol: `official_colored_mnist_reversed_color_mlp`.

When `primary_protocol=official_colored_mnist_reversed_color_mlp`, the interpreted comparison uses the official IRMv1 Colored MNIST reversed-color protocol: binary label `digit < 5`, 25% label noise, train color-flip probabilities `0.2/0.1`, target color-flip probability `0.9`, 2-channel 14x14 MLP, BCE-with-logits, L2 weight penalty `0.001`, penalty annealing, and whole-loss rescaling after anneal.

The older CNN local-response path remains available as a non-authoritative experimental path using source correlations `[0.9, 0.8]`, OOD target correlation `0.1`, label noise `0.25`, `SmallCMNISTCNN`, Adam optimizer family, seed handling, and counterfactual color probe.

This Task 3 run uses a bounded local sample/epoch budget recorded in `preregistered_design.json`: profile `main`, train per environment `2500`, epochs `8`, batch size `128`. The generator and architecture are not redesigned to favor the new method.

All methods share the same source data, source validation data, target evaluation data, seeds, optimizer family, checkpoint fractions, and source-only selection rule. Target outcomes are read after the preregistered design is written and after source-only beta/checkpoint selection. Full candidate-grid target rows are retained post-hoc so weak source-only selection cannot hide a stronger baseline candidate.

Before the primary comparison is interpreted, `baseline_recovery_gate.json` must show that the reversed-color CMNIST protocol has a non-trivial IRMv1 target baseline and does not allow ERM to pass through as a high-accuracy clean-digit learner.
