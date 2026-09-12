# A/O-conditioned semantic predictive audit

Five methods (ERM, CORAL, IRMv1, V-REx, Fishr) and five seeds were evaluated at checkpoint 300. For each source semantic coordinate, the predictor uses source-defined task sensitivity `A`, source observability `O_S`, and the complete response matrix `R` (three semantic response vectors, Gram entries, and source/counterfactual/clean bank allocation). The outcome is an H=20 continuation under a finite semantic shift (alpha=0.1), measured relative to a zero-shift continuation from the same checkpoint and schedule.

## Results

The leave-one-method-out audit is in `aopi_semantic_incremental_predictive_audit.csv`. For `source_label_noise`, adding full response geometry to `(A,O_S)` reduces RMSE from 16.52 to 0.89, while `A` alone has RMSE 16.74. This is the first non-tautological incremental signal in this line: response geometry carries information about the finite continuation outcome beyond A/O for this mechanism.

For `source_env0_color` and `source_env1_color`, response geometry does not improve over A-only or A/O baselines (RMSE increases). Therefore the signal is mechanism-specific, not a universal response representation. It is also not evidence that label-noise response is causal for OOD accuracy: the outcome is a local continuation displacement, not a held-out target metric or an intervention on the trained algorithm.

## Boundary

BIRM and LoRA-BIRM remain outside this full-network predictive audit because their available official artifacts expose a separate head-only continuation interface. They are retained in the static A/O/head analysis and must be added only through a declared common-space adapter.

Current status: `MECHANISM_SPECIFIC_INCREMENTAL_SIGNAL_PENDING_INDEPENDENT_REPLICATION`. The label-noise result warrants a preregistered replication with an independently generated finite outcome and an additional negative control; color mechanisms currently provide a counterexample to a universal response-geometry claim.
