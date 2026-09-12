# Semantic mechanism-specific behavior

This audit uses small semantic response slopes as predictors and held-out finite shifts of the same data-generating mechanism as outcomes. It includes ERM, CORAL, IRMv1, V-REx, Fishr, Full-BIRM and LoRA-BIRM over five seeds. BIRM variants use official final checkpoints; they do not share the survey optimizer continuation state and are therefore compared at model-response level only.

The output is organized by mechanism (`source_color_common`, `source_color_contrast`, `source_label_noise`) and finite shift alpha, rather than pooled target accuracy. The current artifact is descriptive: no cross-method predictive model or intervention claim is asserted until a held-out method audit is run.
