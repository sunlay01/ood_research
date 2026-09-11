# TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL

- task_id: `TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL`
- written_at: `1789120398.762412`
- git_head: `a0d991d9ae609a52ba3861be32ef18d3406072de`
- branch: `task3-cmnist-local-response`
- config_sha256: `bb460b86aad7f7817fef107ef2dd21a14a7c76c29dab6a6d35d8d01ff8a845e0`
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

## Fixed interpretation ceiling

This is a common-budget source/evaluation response survey. It does not claim semantic mechanism recovery, causality, a new algorithm, theory validation, or a universal DG taxonomy. Methods are admitted to A/O/Pi only through source-only training and continuation fidelity, never through target performance. Paper references define algorithms; paper benchmark reproduction is explicitly not the goal.
