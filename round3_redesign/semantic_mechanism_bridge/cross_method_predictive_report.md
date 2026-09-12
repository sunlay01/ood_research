# Cross-method mechanism-specific predictive audit

For each semantic direction and finite shift, a one-feature ridge model maps the source-defined local loss response slope to finite-shift loss change. Evaluation is leave-one-method-out across seven methods (ERM, CORAL, IRMv1, V-REx, Fishr, Full-BIRM, LoRA-BIRM); target accuracy is never a predictor.

This is a diagnostic of whether local response is informative for the same semantic mechanism at finite scale. It is not yet a causal mechanism result: the finite outcome uses the same fixed model and data-generating coordinate, and the BIRM variants have a distinct official checkpoint interface.
