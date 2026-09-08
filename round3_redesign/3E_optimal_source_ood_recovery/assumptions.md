# 3E Assumptions

3E concerns finite-dimensional Euclidean spaces only.  A world perturbation is
`u in U`, source information is `O u`, and the desired output is the
source-whitened OOD response `A u`.  The world norm is explicitly declared;
changing it changes the numerical value of the recovery error.

`O : U -> Y_S` and `A : U -> R` are deterministic linear maps.  `R` carries
the Euclidean coordinates induced by the frozen 3A source-whitened response
norm.  The source-only recovery theorem allows every deterministic map from
`Im(O)` to `R`; no linearity, continuity, probabilistic prior, target-risk
oracle, semantic label, or regularizer is assumed.

The primary numerical pair is not a representative abstract matrix.  Its
world space has the declared standardized coordinates `S1/S2 relation`,
`S1/S2 mean`, `S1/S2 variance`, independent-noise variance, and target-emergent
`U` coupling.  It uses the Euclidean world metric with magnitude-one scales
`0.20, 0.20, 0.35, 0.35, 0.30, 0.30, 0.50, 0.75`, respectively.  This is a
benchmark-level local schedule, not a canonical causal parameterization.

For that primary pair, 3A supplies the fixed source-whitened response norm and
its local vulnerability interpretation.  3D supplies the task-complete source
state observation: stacked central-difference derivatives of
`psi_e=(svec(M_e),m_e,c_e)` across registered source environments.  The source
observation construction does not read target risk, labels, clusters, or 3C
regularizer geometry.  3B and 3C are therefore exclusions/boundaries rather
than theorem inputs.

Alpha is conditional on this declared world metric.  It is invariant under
consistent source recodings and world/response isometries, but not under an
arbitrary world reparameterization with the Euclidean metric silently held
fixed.  Neither alpha nor the recovery theorem is a target-risk or universal
domain-generalization lower bound.

The nonlinear exercise reports only Jacobians at a reference world and is
labelled `LOCAL-TANGENT-DIAGNOSTIC-ONLY`.  It is not a global nonlinear
recovery theorem or a finite-sample result.
