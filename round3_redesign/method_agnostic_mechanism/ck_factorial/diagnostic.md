# Corrected C/K factorial smoke

The previous factorial run is invalid and excluded. This corrected smoke fixes: checkpoint semantics (`theta_300` is after updates 0..299 and uses batch 300), the Newton minus sign, inclusion of L2 in the base risk/Hessian, common parameter trust-region radius, and projected solve instrumentation. It runs one seed per method before expansion.

At this stage the output is only a numerical smoke. The implementation now uses an orthonormal float64 subspace, a single common alpha per radius (so raw C/K amplitude is retained), and an automatic vector radius-scaling audit at rho, rho/2 and rho/4. Positive-definiteness, conditioning and solve-residual gates are recorded for every cell; derived contrasts inherit validity from all required cells. This is still not a scientific mechanism claim.

## Corrected smoke outcome

The corrected implementation uses one common scalar alpha across all four cells, preserves raw operator amplitude, includes signed eigenvalues and negative-eigenvalue counts, and evaluates radii 1e-3, 5e-4, and 2.5e-4. Any contrast involving an indefinite, ill-conditioned, non-linear or unresolved cell is marked `derived_valid=false` and must not be interpreted as forcing/filtering. In particular, IRMv1 K-on cells are indefinite in the smoke and are rejected by the PD gate. The smoke therefore validates the numerical apparatus only; it does not support an IRMv1 forcing-dominant (or any other) mechanism claim. Multiple seeds remain required after the numerical gates pass.
