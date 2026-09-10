# TASK3-CMNIST-CPU-MINIMAL Context

Current task:

- Run a CPU-only minimal ColoredMNIST proof-of-concept for `I` versus damped detached `H^-1` on the same head-gradient disagreement.

Required context:

1. Frozen 3A-3E theory is NOT reopened.
2. Old `task3_baseline_fidelity` is diagnostic history only.
3. Facebook native IRM anchor is ERM approximately `0.168` and IRMv1 approximately `0.667` target accuracy.
4. Current task is a CPU-minimal proof-of-concept.
5. `GRAD` is our own controlled ablation, not IGA/Fish/Fishr.
6. Only `I` versus `H^-1` is being tested.

Implementation constraints:

- No beta grid.
- No checkpoint selection.
- No target-based tuning.
- No DomainBed, IGA, Fish, Fishr, V-REx, CORAL, MLDG, random metric, shuffled local response, RESP2, `A_rec`, `E`, or `rho_slack` estimator.

Historical reopen: none.
