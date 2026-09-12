# Corrected C/K factorial smoke

The previous factorial run is invalid and excluded. This corrected smoke fixes: checkpoint semantics (`theta_300` is after updates 0..299 and uses batch 300), the Newton minus sign, inclusion of L2 in the base risk/Hessian, common parameter trust-region radius, and projected solve instrumentation. It runs one seed per method before expansion.

At this stage the output is only a numerical smoke. A scientific interpretation requires finite-difference radius scaling, condition-number gates, and multiple seeds.
