# Corrected C/K factorial smoke

The previous factorial run is invalid and excluded. This corrected smoke fixes: checkpoint semantics (`theta_300` is after updates 0..299 and uses batch 300), the Newton minus sign, inclusion of L2 in the base risk/Hessian, common parameter trust-region radius, and projected solve instrumentation. It runs one seed per method before expansion.

At this stage the output is only a numerical smoke. A scientific interpretation requires finite-difference radius scaling, condition-number gates, and multiple seeds.

## Corrected smoke outcome

The corrected implementation now uses one common scalar alpha across all four cells, preserves raw operator amplitude, includes signed eigenvalues and negative-eigenvalue counts, and evaluates radii 1e-3, 5e-4, and 2.5e-4. Fishr, V-REx, and IRMv1 `(00)/(10)` cells show near-linear norm scaling (approximately 1/2 and 1/4). Several K-on cells are indefinite (`min_eigenvalue < 0`), so their filtering interpretation is rejected by the positive-definite gate. IRMv1 `(11)` also fails the radius-linearity check. The smoke therefore validates parts of the numerical apparatus but does not yet pass the scientific C/K gate.
