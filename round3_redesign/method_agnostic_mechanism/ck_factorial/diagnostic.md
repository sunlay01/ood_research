# Corrected C/K factorial smoke

The previous factorial run is invalid and excluded. This corrected smoke fixes: checkpoint semantics (`theta_300` is after updates 0..299 and uses batch 300), the Newton minus sign, inclusion of L2 in the base risk/Hessian, common parameter trust-region radius, and projected solve instrumentation. The frozen implementation (commit `5dfb625`) was run on seeds 10--14 for each method.

At this stage the output is only a numerical smoke. The implementation now uses an orthonormal float64 subspace, a single common alpha per radius (so raw C/K amplitude is retained), and an automatic vector radius-scaling audit at rho, rho/2 and rho/4. Positive-definiteness, conditioning and solve-residual gates are recorded for every cell; derived contrasts inherit validity from all required cells. This is still not a scientific mechanism claim.

## Corrected smoke outcome

The corrected implementation uses one common scalar alpha across all four cells, preserves raw operator amplitude, includes signed eigenvalues and negative-eigenvalue counts, and evaluates radii 1e-3, 5e-4, and 2.5e-4. Any contrast involving an indefinite, ill-conditioned, non-linear or unresolved cell is marked `derived_valid=false` and must not be interpreted as forcing/filtering. Across five seeds, all Fishr cells pass every gate. IRMv1 and V-REx K-on cells are indefinite for all five seeds, so their filtering and interaction contrasts are invalid by construction; their C-only contrasts pass. This is a stable numerical property of the chosen projection and checkpoint, but it does not establish a forcing mechanism or imply causal algorithm semantics.
