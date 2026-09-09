# Method Definitions

`ERM`: source cross-entropy only.

`IRMv1`: standard scalar-risk-gradient penalty on source environments.

`V-REx`: variance of source cross-entropy risks.

`HEAD_GRADIENT_VARIANCE_SURROGATE`: mean squared deviation of per-source head gradients from their source mean. NOT A REPRODUCTION OF IGA OR FISH.

`LOCAL_RESPONSE`: the same centered head-gradient disagreement weighted by the stop-gradient inverse damped source head Hessian/Gauss-Newton metric.

`RANDOM_METRIC`: the LOCAL_RESPONSE form with a random SPD metric matched to the real metric's eigenvalue multiset, trace, and Frobenius scale.

`SHUFFLED_LOCAL_RESPONSE`: the LOCAL_RESPONSE penalty after shuffling source environment identity while preserving batch sizes.
