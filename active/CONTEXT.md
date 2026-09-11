# TASK-AOPI-MULTIMETHOD-MECHANISM-SURVEY Context

This isolated audit compares ERM, IRMv1, VREX, and CORAL under corrected CPU-minimal CMNIST. The R5 base world is `(0.2, 0.1, 0.9, 0.25, 0.25)` and tangent coordinates are displacements from that base.

The old audits remain unchanged and are not runtime inputs to this survey. All four methods use common source rows, initialization, and schedule. Response differences are descriptive only.

Trusted inputs:

- `configs/task3_aopi_multimethod_mechanism_survey.json` and corrected CPU-minimal source-generating semantics.
- Accepted ERM/IRM reconstruction manifest for seeds `10..14` as an F0 anchor.

Source-only construction:

- Smooth four-outcome expectation derivatives on five R5 probability coordinates.
- A includes `D grad(R_T - R_S)`; O contains only source risk gradients and no method penalty.
- Pi_full uses K=1,5,20 source-only continuations with plus/minus/control replay.

Evaluation-only construction:

- Held-out evaluation worlds enter A, evaluation banks, and post-hoc performance only.
- Target rows cannot select source fit, direction, continuation, normalization, grouping, or method inclusion.

Canonical state remains unchanged; completion writes only `active/STATE_DELTA.md`.
