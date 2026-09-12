# A/O-conditioned semantic response audit

This audit restores the A/O main line. Semantic response is joined to the task sensitivity operator A and source observability operator O_S for the same method, seed and opaque direction. A and O_S remain in their declared codomains; no invalid A+O projection is formed.

For source-side semantic directions, the report records raw task sensitivity, source observability, response magnitude, and ratios. These are descriptive coordinates, not a causal decomposition.

## K=1

method                   CORAL      ERM   FISHR    IRMv1    VREX
semantic_direction                                              
source_color_common    22.2336  22.1978  0.1419   0.7064  0.0205
source_color_contrast   0.4410   0.4440  2.6331   0.2085  1.1318
source_env0_color      15.7419  15.7215  1.9475   0.3562  0.7985
source_env1_color      15.7074  15.6775  1.7766   0.6449  0.8023
source_label_noise     18.2194  18.4737  0.3150  12.4933  0.3318

## K=5

method                    CORAL       ERM    FISHR    IRMv1     VREX
semantic_direction                                                  
source_color_common    192.8005  189.9594   1.9648   4.1946   0.1940
source_color_contrast    1.9437    1.9666  30.4271   1.0167  11.2184
source_env0_color      136.7043  134.6885  22.8116   2.6874   7.9923
source_env1_color      135.9797  133.9785  20.2167   3.2625   7.8733
source_label_noise     132.1639  132.5997   2.7050  90.6180   1.8377

## K=20

method                    CORAL       ERM     FISHR    IRMv1     VREX
semantic_direction                                                   
source_color_common    160.9009  158.7019    9.7617   6.8765   0.6399
source_color_contrast    4.8569    5.0273  175.8171   4.2529  11.5536
source_env0_color      113.8224  112.1871  130.6199   6.8943   8.0278
source_env1_color      113.8326  112.3652  118.0215   5.0491   8.3367
source_label_noise     463.6886  458.5014    8.4127  73.8286  10.6681

## BIRM/LoRA-BIRM boundary

Official BIRM and LoRA-BIRM checkpoints are not merged into this full-network table. Their existing A/O/Pi artifact is head-only and lacks the full-network continuation state; it remains a separate representation/head analysis.

## Verdict

`NO-POSITIVE-SEMANTIC-MECHANISM-CANDIDATE`: this audit supplies the missing A/O-conditioned semantic coordinates, but it does not yet test held-out finite behavior or an intervention that changes one semantic component while holding A/O fixed.
