# CMNIST semantic direction registry

Directions are fixed from `smooth_world5.py` before inspecting target metrics.
The primary source-side coordinates are `source_env0_color`,
`source_env1_color`, and `source_label_noise`; derived coordinates are fixed
linear combinations. Evaluation color/noise are post-hoc readouts only.

| id | definition | use |
|---|---|---|
| source_env0_color | perturb source environment 0 color flip probability | source mechanism |
| source_env1_color | perturb source environment 1 color flip probability | source mechanism |
| evaluation_color | perturb evaluation color probability | post-hoc only |
| source_label_noise | perturb source label noise | source mechanism |
| evaluation_label_noise | perturb evaluation label noise | post-hoc only |
| source_color_common | equal source color perturbation | derived source |
| source_color_contrast | opposite source color perturbation | derived source |
| color_global / color_shift | fixed derived color combinations | diagnostic |
| noise_global / noise_shift | fixed derived noise combinations | diagnostic |

These coordinates describe data-generation interventions and are not inferred
from target performance. C/K factorial responses remain local response
interventions; they are not themselves semantic mechanisms.
