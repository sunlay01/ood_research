# Protocol fixed before running (2026-11-06)
Original paired logistic model, n=512, k=1, R=R0+R1, Delta=R0-R1.
Primary seeds 21,22,23; replication seeds 24 through 32 inclusive. No condition-based filtering.
One fixed initialization RNG1000, N(0,.1^2 I4), shared across datasets.
Actual ERM gradient flow for 100 time units; penalty lambda=10000, flow for 100 more.
Record t=0,.01,.1,1,3,10,30,100 plus a fixed dense grid.
Primary fraction: empirical S/(U+S), U=mean(aD+d)^2, S=mean(b+hZ)^2.
Also record population component fraction (b²+h²)/(a²+d²+b²+h²).
Exact heterogeneity share h² Var(Z)/S, not a measure of learned features.
Source plateau: abs(d meanR/dt)/meanR <= threshold and abs(Delta)<=1e-3,
sustained on all remaining grid points up to time100. Thresholds .01,.001,.0001.
Report net and maximum rho gain after plateau entry; material net gain >.001 (0.1 pp).
Also fixed window10-30; do not select window by the sign of rho.
Cancellation: raw per-sample signed response and b/h Shapley contributions to Delta.
Decompose rho velocity into risk, penalty, tangent risk, and remaining normal motion.
Primary-seed interventions:
(1) restart at empirical shape-only optimum; retain exact source data and penalty;
(2) remove h coordinate, starting from same warm state with h=0;
(3) analytically average color assignments given Y, retaining D,Y,Z and same warm start.
An auxiliary exactly constrained flow starts at shape-only optimum.
Seed22 lambda sensitivity: 1000 and100000, same ERM warm state.
These interventions test finite-sample color noise and availability of heterogeneous
responses. They do not establish neural feature-learning or CMNIST mechanisms.

