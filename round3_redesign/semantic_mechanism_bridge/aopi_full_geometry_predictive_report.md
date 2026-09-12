# Full A/O geometry, mechanism specificity, and schedule transfer

All rows use logical checkpoint C_300; local response geometry is measured on probe schedule A and finite-shift outcomes on independent schedule B. Both are forked from the same complete C_300 bundle.

## Predictive audit

semantic_direction                 predictor                   cv  n  folds      rmse  fold_rmse_mean  outcome_sd
 source_env0_color           A_scalar_target leave_one_method_out 25      5  1.369480        1.327269    6.130107
 source_env0_color           A_scalar_target   leave_one_seed_out 25      5  1.281148        1.238732    6.130107
 source_env0_color          AO_scalar_target leave_one_method_out 25      5  1.806980        1.645085    6.130107
 source_env0_color          AO_scalar_target   leave_one_seed_out 25      5  1.291464        1.278501    6.130107
 source_env0_color           A_full_geometry leave_one_method_out 25      5  1.231128        1.128531    6.130107
 source_env0_color           A_full_geometry   leave_one_seed_out 25      5  1.323146        1.300191    6.130107
 source_env0_color          AO_full_geometry leave_one_method_out 25      5  2.470260        2.239117    6.130107
 source_env0_color          AO_full_geometry   leave_one_seed_out 25      5  1.485763        1.467596    6.130107
 source_env0_color       AO_full_plus_R_full leave_one_method_out 25      5  3.260726        2.547563    6.130107
 source_env0_color       AO_full_plus_R_full   leave_one_seed_out 25      5  1.397536        1.266332    6.130107
 source_env0_color AO_full_plus_R_color_only leave_one_method_out 25      5  3.128587        2.175550    6.130107
 source_env0_color AO_full_plus_R_color_only   leave_one_seed_out 25      5  1.504193        1.357993    6.130107
 source_env0_color AO_full_plus_R_noise_only leave_one_method_out 25      5  4.382468        3.351759    6.130107
 source_env0_color AO_full_plus_R_noise_only   leave_one_seed_out 25      5  1.295650        1.271982    6.130107
 source_env0_color           method_identity   leave_one_seed_out 25      5  1.200938        1.156841    6.130107
 source_env1_color           A_scalar_target leave_one_method_out 25      5  1.075474        1.054099    5.931877
 source_env1_color           A_scalar_target   leave_one_seed_out 25      5  1.220933        1.217202    5.931877
 source_env1_color          AO_scalar_target leave_one_method_out 25      5  1.438161        1.371732    5.931877
 source_env1_color          AO_scalar_target   leave_one_seed_out 25      5  1.261667        1.257839    5.931877
 source_env1_color           A_full_geometry leave_one_method_out 25      5  1.231383        1.216143    5.931877
 source_env1_color           A_full_geometry   leave_one_seed_out 25      5  1.423325        1.419034    5.931877
 source_env1_color          AO_full_geometry leave_one_method_out 25      5  1.561575        1.489520    5.931877
 source_env1_color          AO_full_geometry   leave_one_seed_out 25      5  1.622415        1.618337    5.931877
 source_env1_color       AO_full_plus_R_full leave_one_method_out 25      5  1.931303        1.605833    5.931877
 source_env1_color       AO_full_plus_R_full   leave_one_seed_out 25      5  1.342033        1.287080    5.931877
 source_env1_color AO_full_plus_R_color_only leave_one_method_out 25      5  1.352642        1.171166    5.931877
 source_env1_color AO_full_plus_R_color_only   leave_one_seed_out 25      5  1.242501        1.191815    5.931877
 source_env1_color AO_full_plus_R_noise_only leave_one_method_out 25      5  3.407538        2.693641    5.931877
 source_env1_color AO_full_plus_R_noise_only   leave_one_seed_out 25      5  1.338446        1.318741    5.931877
 source_env1_color           method_identity   leave_one_seed_out 25      5  1.226613        1.196399    5.931877
source_label_noise           A_scalar_target leave_one_method_out 25      5 13.683050       12.069987    5.944723
source_label_noise           A_scalar_target   leave_one_seed_out 25      5  5.973996        5.972750    5.944723
source_label_noise          AO_scalar_target leave_one_method_out 25      5 14.265238       12.271373    5.944723
source_label_noise          AO_scalar_target   leave_one_seed_out 25      5  5.893311        5.887469    5.944723
source_label_noise           A_full_geometry leave_one_method_out 25      5  1.668400        1.267621    5.944723
source_label_noise           A_full_geometry   leave_one_seed_out 25      5  0.919216        0.857453    5.944723
source_label_noise          AO_full_geometry leave_one_method_out 25      5  4.159505        3.231908    5.944723
source_label_noise          AO_full_geometry   leave_one_seed_out 25      5  1.003821        0.965027    5.944723
source_label_noise       AO_full_plus_R_full leave_one_method_out 25      5  3.560618        2.768228    5.944723
source_label_noise       AO_full_plus_R_full   leave_one_seed_out 25      5  1.391695        1.319105    5.944723
source_label_noise AO_full_plus_R_color_only leave_one_method_out 25      5  3.853201        2.910465    5.944723
source_label_noise AO_full_plus_R_color_only   leave_one_seed_out 25      5  1.231614        1.160681    5.944723
source_label_noise AO_full_plus_R_noise_only leave_one_method_out 25      5  5.065063        3.775792    5.944723
source_label_noise AO_full_plus_R_noise_only   leave_one_seed_out 25      5  1.022135        1.000971    5.944723
source_label_noise           method_identity   leave_one_seed_out 25      5  0.556262        0.511961    5.944723

`A_full_geometry` contains norms and normalized within-codomain Gram entries for all three A columns. `AO_full_geometry` adds the corresponding O_S block without taking cross-codomain inner products. The response blocks use norms, within-response cosines, and source/counterfactual/clean allocation fractions; diagonal Gram entries are not included as redundant features.

The label-noise negative-control comparison is the key adjudication: if color-only response geometry predicts the label-noise outcome as well as noise-only response geometry, the signal is consistent with a method fingerprint. A noise-specific gain that survives the full A/O geometry baseline and independent schedule is the narrower mechanism-matched result. These statistics remain predictive evidence and do not establish causality by themselves.

## Observed label-noise pattern

Under leave-one-method-out, the RMSE values are A_scalar_target=13.683, AO_scalar_target=14.265, A_full_geometry=1.668, AO_full_geometry=4.160, AO_full_plus_R_full=3.561, AO_full_plus_R_color_only=3.853, AO_full_plus_R_noise_only=5.065.
Under leave-one-seed-out, the corresponding full-geometry and response controls are A_full_geometry=0.919, AO_full_geometry=1.004, AO_full_plus_R_full=1.392, AO_full_plus_R_color_only=1.232, AO_full_plus_R_noise_only=1.022, method_identity=0.556.
The apparent gain is therefore not stable across the two holdouts: A_full_geometry already outperforms scalar A/O, method identity is a strong seed-held-out baseline, and adding response geometry does not improve the independent-schedule result beyond the complete A/O geometry. The present audit does not identify a stable mechanism-matched Pi O_S increment.