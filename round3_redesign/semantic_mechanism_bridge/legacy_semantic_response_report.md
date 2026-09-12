# Legacy semantic response report

Five-seed source-defined response reconstruction for ERM, IRMv1, V-REx, Fishr and CORAL. Target accuracy is attached only post-hoc.

## K=1

method                CORAL      ERM   FISHR   IRMv1    VREX
semantic_direction                                          
source_env0_color   18.9578  19.3545  0.2582  0.3670  0.5674
source_env1_color   19.0522  19.4401  0.2335  0.6730  0.5720
source_label_noise   6.4664   6.6325  0.0363  7.3967  0.0840

## K=5

method                 CORAL       ERM   FISHR    IRMv1    VREX
semantic_direction                                             
source_env0_color   163.1323  164.4657  3.0378   2.7118  5.6341
source_env1_color   163.3437  164.7687  2.6744   3.4473  5.5741
source_label_noise   54.6241   55.6414  0.2749  53.6606  0.4444

## K=20

method                 CORAL       ERM    FISHR    IRMv1    VREX
semantic_direction                                              
source_env0_color   135.7958  136.6288  17.0444   6.7157  5.6852
source_env1_color   135.8505  136.8864  15.2231   4.9646  5.9218
source_label_noise  125.5321  128.0204   0.8223  44.0039  2.6275

## Interpretation

This artifact establishes a semantic descriptive table, not a mechanism. It does not select an axis using target outcomes, and it does not claim that color response suppression is sufficient for OOD success.
