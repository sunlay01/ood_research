# Algorithm → bridge object → target functional

| Algorithm/object | Layer A: algorithmic object | Layer B: bridge object actually used | Layer C: target functional |
|---|---|---|---|
| ERM | mean empirical source risk | within/domain concentration | `Q_Pi` or source risk |
| V-REx | source risk variance | risk-vector dispersion | restricted DG proxy |
| GroupDRO | max/group weighted risk | simplex support function | `sup_{w in W} w·R` |
| MMD/CORAL/DANN | marginal representation penalty | IPM/moment witness | target risk + discrepancy/oracle |
| Ideal IRM | simultaneous optimality constraint | invariant conditional / identifiability set | intervention/DG risk |
| IRMv1 | classifier-gradient penalty | TV variation or derivative functional | OOD functional, restricted |
| Fishr | gradient covariance matching | derivative covariance statistic | no proved generic target functional |
| ICP/anchor | invariant conditional/anchor residual | SCM identifiability or primal-dual shift object | intervention robust risk |
| f-/Wasserstein DRO | robust objective | uncertainty set support/dual variable | worst-case risk in `U` |
| MLDG | bilevel update | Taylor/meta-test functional | DG proxy, restricted |
| Tri-Space | latent decomposition | localized discrepancy terms | target risk with latent assumptions |

The same Layer C theorem can sometimes be reused for different algorithms, but only after an algorithm-specific Layer A→B lemma is proved.
