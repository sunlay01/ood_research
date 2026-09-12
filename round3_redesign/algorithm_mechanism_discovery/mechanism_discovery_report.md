# Data-first mechanism discovery report

This is a discovery and falsification artifact. Discovery features do not contain target accuracy or target loss. Target outcomes enter only in the external help/hurt audit.

## Inputs and scope

- Full-network rows: 20 across ERM, IRMv1, VREX, FISHR.
- Representation/head rows: 10 across IRMv1, Full-BIRM, LoRA-BIRM.
- BIRM/LoRA rows are kept in a separate feature space because their current Pi is fixed-encoder/head-only.

## Numeric families

- Full-network selected k=2, silhouette=0.639, bootstrap pair agreement=0.998 +/- 0.023.
- Representation/head selected k=2, silhouette=0.359, bootstrap pair agreement=0.904 +/- 0.186.
- Full-network cluster composition: {0: 'FISHR, IRMv1, VREX', 1: 'ERM'}.
- Representation/head cluster composition: {0: 'Full-BIRM (Official), IRMv1', 1: 'IRMv1'}.
- Cluster IDs are numeric family IDs. They are not semantic mechanism labels.

## External help/hurt audit

                 model  n  accuracy  balanced_accuracy                                                                                                                                                                                                                                                 folds
       method_identity 20       1.0                1.0 [{"n_test": 4, "seed": 10, "test_accuracy": 1.0}, {"n_test": 4, "seed": 11, "test_accuracy": 1.0}, {"n_test": 4, "seed": 12, "test_accuracy": 1.0}, {"n_test": 4, "seed": 13, "test_accuracy": 1.0}, {"n_test": 4, "seed": 14, "test_accuracy": 1.0}]
           source_risk 20       1.0                1.0 [{"n_test": 4, "seed": 10, "test_accuracy": 1.0}, {"n_test": 4, "seed": 11, "test_accuracy": 1.0}, {"n_test": 4, "seed": 12, "test_accuracy": 1.0}, {"n_test": 4, "seed": 13, "test_accuracy": 1.0}, {"n_test": 4, "seed": 14, "test_accuracy": 1.0}]
      numeric_geometry 20       1.0                1.0 [{"n_test": 4, "seed": 10, "test_accuracy": 1.0}, {"n_test": 4, "seed": 11, "test_accuracy": 1.0}, {"n_test": 4, "seed": 12, "test_accuracy": 1.0}, {"n_test": 4, "seed": 13, "test_accuracy": 1.0}, {"n_test": 4, "seed": 14, "test_accuracy": 1.0}]
identity_plus_geometry 20       1.0                1.0 [{"n_test": 4, "seed": 10, "test_accuracy": 1.0}, {"n_test": 4, "seed": 11, "test_accuracy": 1.0}, {"n_test": 4, "seed": 12, "test_accuracy": 1.0}, {"n_test": 4, "seed": 13, "test_accuracy": 1.0}, {"n_test": 4, "seed": 14, "test_accuracy": 1.0}]

The method-identity baseline is expected to be strong in this first pass because the existing four-method table has one training configuration per method. This is a confounding warning, not evidence of a mechanism.
The current predictive audit confirms this warning: method identity, source risk, and numeric geometry all obtain perfect leave-one-seed-out accuracy because the help label is almost identical to the method split. No beyond-identity mechanism evidence is claimed.

## Counterexamples

Candidate counterexample pairs: 16. Same-method close-geometry target gaps and different-method close-target geometry gaps are retained for adversarial review.

## Interpretation ceiling

The first pass can establish stable response-geometry families only. Forcing/filtering/interaction names require common-base C/K counterfactuals. A family that disappears after controlling method identity, lambda, or environment family is treated as a confounder.
