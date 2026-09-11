# FAD reference audit

Status: `DEFERRED`.

The task requires exact zeroth-order and first-order Flatness-Aware Minimization for Domain Generalization update semantics. This implementation does not replace FAD with SAM or a generic flatness penalty. `FAD` is retained in `candidate_methods` and receives `admitted_to_training_panel=false`, `admitted_to_pi_full=false` until the exact update can be implemented without changing frozen CMNIST/data/model semantics.
