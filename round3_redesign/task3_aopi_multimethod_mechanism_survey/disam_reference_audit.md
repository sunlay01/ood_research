# DISAM reference audit

Status: `DEFERRED`.

The task requires exact Domain-Inspired SAM perturbation calibration from source-domain loss imbalance/convergence degree. This implementation does not replace DISAM with `SAM + VREX` or any surrogate. `DISAM` is retained in `candidate_methods` and receives `admitted_to_training_panel=false`, `admitted_to_pi_full=false` until the exact domain-aware update can be implemented faithfully under the frozen common harness.
