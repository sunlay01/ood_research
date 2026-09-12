# Leakage-free lagged response-descriptor pilot

At step t, descriptors are lagged to predict the next functional update at t+1. Scaling is fitted within each held-out-seed training fold. MultiTaskLasso imposes group sparsity across the full response vector. This remains a descriptor-predictability experiment, not causal mechanism identification; matched interventions are required.

- FISHR: held-out vector R2=0.2017±0.1182.
- IRMv1: held-out vector R2=0.2791±0.0734.
- VREX: held-out vector R2=0.1901±0.0704.
