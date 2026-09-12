# Revision: mechanism identification versus trajectory description

The previous candidate audit is **observational**. It compares quantities measured after training with target accuracy. Even with five seeds, this cannot identify a causal mechanism: method, optimizer state, representation state, and the measured quantity are jointly determined by training. Pairwise rank agreement is therefore exploratory evidence only.

A valid forcing/filtering claim requires a matched fork intervention. At checkpoint `(theta_t, optimizer_state_t)`, clone the complete state and replay the same minibatches/randomness while changing exactly one component:

- forcing fork: `B + lambda*C` versus `B`, or a common-base replacement of `C`;
- filtering fork: fixed forcing with `K=0` versus `K=K_j` using a local HVP/CG quadratic step;
- coupling fork: hold `Delta H` fixed while changing the readout `W`, and conversely.

The estimand is the paired change in short-horizon functional response and subsequent source/target behavior. Without this intervention, `(C,K)->Pi` is a theoretical coordinate system and the CMNIST path tables are descriptive geometry. The current evidence does not support a method-independent scalar mechanism or a causal forcing/filtering decomposition.

The five-seed trajectory artifacts are retained for power and variance planning, but must not be presented as mechanism validation until matched forks are run.
