# A/O-conditioned semantic predictive audit

Five methods and five seeds; logical checkpoint C_300 (after updates 0..299), legacy trainer checkpoint key 299; finite shift alpha=0.1, continuation horizon 20.
All A/O geometry, local response forks, and finite-shift outcome forks were materialized from the same complete model/optimizer/algorithm-state bundle. The initial model, optimizer, algorithm-state, and bank-logit hashes passed the hard state gate for every method/seed.

Leave-one-method-out results:

semantic_direction  model  n  methods  lo_method_rmse  outcome_sd
 source_env0_color A_only 25        5        1.319311    6.142356
 source_env0_color     AO 25        5        1.902852    6.142356
 source_env0_color   AO_R 25        5        1.157510    6.142356
 source_env1_color A_only 25        5        1.092764    5.938999
 source_env1_color     AO 25        5        1.312153    5.938999
 source_env1_color   AO_R 25        5        1.890069    5.938999
source_label_noise A_only 25        5       13.867450    5.840032
source_label_noise     AO 25        5       14.012023    5.840032
source_label_noise   AO_R 25        5        1.323103    5.840032

The full response model is evidence beyond A/O only if its leave-one-method-out RMSE improves over AO without using target outcomes. This audit remains a predictive response test; it does not by itself establish a causal semantic mechanism.