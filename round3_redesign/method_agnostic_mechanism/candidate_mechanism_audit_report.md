# CMNIST candidate mechanism audit

All features are source-side; target is merged only for external ranking audit.

- FISHR / early / cancellation_efficiency: rank agreement=1.000, pairs=3, counterexamples=none.
- FISHR / early / mean_adjacent_cosine: rank agreement=0.333, pairs=3, counterexamples=11-13;12-13.
- FISHR / early / clean_to_source: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- FISHR / early / cf_source_ratio: rank agreement=0.333, pairs=3, counterexamples=11-13;12-13.
- FISHR / late / cancellation_efficiency: rank agreement=0.667, pairs=3, counterexamples=11-12.
- FISHR / late / mean_adjacent_cosine: rank agreement=0.333, pairs=3, counterexamples=11-13;12-13.
- FISHR / late / clean_to_source: rank agreement=0.667, pairs=3, counterexamples=11-12.
- FISHR / late / cf_source_ratio: rank agreement=0.333, pairs=3, counterexamples=11-13;12-13.
- FISHR / middle / cancellation_efficiency: rank agreement=0.667, pairs=3, counterexamples=12-13.
- FISHR / middle / mean_adjacent_cosine: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- FISHR / middle / clean_to_source: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- FISHR / middle / cf_source_ratio: rank agreement=0.333, pairs=3, counterexamples=11-13;12-13.
- IRMv1 / early / cancellation_efficiency: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- IRMv1 / early / mean_adjacent_cosine: rank agreement=0.667, pairs=3, counterexamples=12-13.
- IRMv1 / early / clean_to_source: rank agreement=1.000, pairs=3, counterexamples=none.
- IRMv1 / early / cf_source_ratio: rank agreement=1.000, pairs=3, counterexamples=none.
- IRMv1 / late / cancellation_efficiency: rank agreement=1.000, pairs=3, counterexamples=none.
- IRMv1 / late / mean_adjacent_cosine: rank agreement=0.667, pairs=3, counterexamples=12-13.
- IRMv1 / late / clean_to_source: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- IRMv1 / late / cf_source_ratio: rank agreement=0.667, pairs=3, counterexamples=11-12.
- IRMv1 / middle / cancellation_efficiency: rank agreement=0.333, pairs=3, counterexamples=11-12;11-13.
- IRMv1 / middle / mean_adjacent_cosine: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- IRMv1 / middle / clean_to_source: rank agreement=1.000, pairs=3, counterexamples=none.
- IRMv1 / middle / cf_source_ratio: rank agreement=0.667, pairs=3, counterexamples=11-12.
- VREX / early / cancellation_efficiency: rank agreement=0.333, pairs=3, counterexamples=11-12;11-13.
- VREX / early / mean_adjacent_cosine: rank agreement=0.333, pairs=3, counterexamples=11-12;12-13.
- VREX / early / clean_to_source: rank agreement=0.333, pairs=3, counterexamples=11-12;12-13.
- VREX / early / cf_source_ratio: rank agreement=0.667, pairs=3, counterexamples=12-13.
- VREX / late / cancellation_efficiency: rank agreement=1.000, pairs=3, counterexamples=none.
- VREX / late / mean_adjacent_cosine: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- VREX / late / clean_to_source: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- VREX / late / cf_source_ratio: rank agreement=0.333, pairs=3, counterexamples=11-12;12-13.
- VREX / middle / cancellation_efficiency: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- VREX / middle / mean_adjacent_cosine: rank agreement=0.667, pairs=3, counterexamples=12-13.
- VREX / middle / clean_to_source: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- VREX / middle / cf_source_ratio: rank agreement=0.000, pairs=3, counterexamples=11-12;11-13;12-13.
- ALL / late / cancellation_efficiency: rank agreement=-0.461, pairs=9, counterexamples=cross-method pooled correlation.
- ALL / late / mean_adjacent_cosine: rank agreement=-0.884, pairs=9, counterexamples=cross-method pooled correlation.
- ALL / late / clean_to_source: rank agreement=-0.812, pairs=9, counterexamples=cross-method pooled correlation.
- ALL / late / cf_source_ratio: rank agreement=-0.073, pairs=9, counterexamples=cross-method pooled correlation.

## Decision

No candidate passes a method-independent mechanism test. A candidate must hold within methods and across methods; these statistics show method-conditioned behavior and counterexamples. Observable transfer is only a proxy because valid dynamic common-space A_t/O_t pullback is unavailable in these artifacts.

## BIRM / LoRA-BIRM boundary

BIRM histories are sparse checkpoints and head-only/frozen-encoder runs. They do not contain per-step functional vectors, so cancellation, adjacent cosine, and state-transition hypotheses cannot be validly tested against the full-network trajectories. Their available head/representation features are retained in `cmnist_rephead_method_features.csv` as a separate numeric family; no cross-layer mechanism claim is made.
