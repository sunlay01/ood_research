# TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL

- task_id: `TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL`
- written_at: `1789123822.40197`
- git_head: `412f4ae2711d1701f265b98c96bda39a48a39fb9`
- branch: `task3-cmnist-local-response`
- config_sha256: `e821f5a159e25601c0087784be9d9e783784ea8ff38228280f48ebcb7ac04f01`
- methods: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'SPECTRAL_NORM_REG', 'SPECTRAL_REG_2024', 'SVB_ORTHDNN', 'STABLE_RANK_NORM', 'SAM', 'ASAM']`
- candidate_methods: `['ERM', 'IRMv1', 'VREX', 'CORAL', 'FISHR', 'MLDG', 'SPECTRAL_NORM_REG', 'SPECTRAL_REG_2024', 'SVB_ORTHDNN', 'STABLE_RANK_NORM', 'SVD_SPARSE', 'SAM', 'ASAM', 'FAD', 'DISAM']`
- seeds: `[10, 11, 12, 13, 14]`
- base_world: `[0.2, 0.1, 0.9, 0.25, 0.25]`
- primary_basis: `['e1', 'e2', 'e3', 'e4', 'e5']`
- delta: `0.01`
- horizons: `[1, 5, 20]`
- target_use: `A, post-hoc performance and evaluation functional banks only`
- descriptive_only: `True`
- verdict_ceiling: `SPECTRAL-FLATNESS-PANEL-PARTIAL`
- source_only_variant_selector: `{'enabled': True, 'calibration_seed': 10, 'selection_metric': 'source_mean_loss', 'tie_breaker': 'source_mean_accuracy', 'source_accuracy_floor': 0.55, 'canonical_selector': 'best_method_specific_source_geometry_then_source_loss', 'variants': {'SPECTRAL_NORM_REG': [0.0, 0.001, 0.01, 0.1, 1.0], 'SPECTRAL_REG_2024': [0.0, 0.001, 0.01, 0.1, 1.0], 'SVB_ORTHDNN': [{'svb_factor': 0.05, 'projection_frequency': 1}, {'svb_factor': 0.05, 'projection_frequency': 10}, {'svb_factor': 0.05, 'projection_frequency': 100}, {'svb_factor': 0.1, 'projection_frequency': 100}, {'svb_factor': 0.2, 'projection_frequency': 100}], 'STABLE_RANK_NORM': [8.0, 16.0, 32.0, 48.0, 64.0], 'SAM': [0.0, 0.01, 0.05, 0.1, 0.2], 'ASAM': [0.0, 0.1, 0.5, 1.0, 2.0]}}`

## Fixed interpretation ceiling

This is a common-budget source/evaluation response survey. It does not claim semantic mechanism recovery, causality, a new algorithm, theory validation, or a universal DG taxonomy. Methods are admitted to A/O/Pi only through source-only training and continuation fidelity, never through target performance. Paper references define algorithms; paper benchmark reproduction is explicitly not the goal.
