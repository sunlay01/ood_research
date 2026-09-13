# Route C — Existing robust/DRO target-set theorem

- **Problem instance:** P5, GroupDRO/f-DRO/Wasserstein DRO.
- **Formalization:** declared uncertainty set and support/dual risk.
- **Reuse/change:** reuse standard robust optimization and uniform convergence; no framework change.
- **Representation:** source risk vector or distributional uncertainty set.
- **Required result:** justify target inclusion and quantify radius/metric misspecification.
- **Assumptions:** predeclared set geometry, support, radius, integrability and finite source complexity.
- **Source-estimable:** empirical robust objective and per-domain risk deviations.
- **Assumption-controlled:** target membership, radius and metric.
- **Irreducible:** risk outside the set and unseen conditional shifts.
- **Failure case:** target outside `U`, or radius large enough to make the bound trivial.
- **Solved if successful:** a valid worst-case guarantee for a named deployment uncertainty class.
