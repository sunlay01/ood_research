# Baseline Fidelity

The run reuses the existing CMNIST binary label, red/green coloring, source correlations `[0.9, 0.8]`, OOD target correlation `0.1`, `SmallCMNISTCNN`, Adam optimizer family, seed handling, and counterfactual color probe.

This Task 3 run uses a bounded local sample/epoch budget recorded in `preregistered_design.json`: profile `main`, train per environment `512`, epochs `4`, batch size `128`. The generator and architecture are not redesigned to favor the new method.

All methods share the same source data, source validation data, target evaluation data, seeds, optimizer family, checkpoint fractions, and source-only selection rule. Target outcomes are read after the preregistered design is written and after source-only beta/checkpoint selection.
