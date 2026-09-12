# Literature semantic decomposition audit

This bounded audit records primary papers whose operational distinctions can be mapped to the fixed CMNIST generator. It is a source for candidate observation coordinates, not evidence that a latent causal mechanism has been identified.

| Paper | Decomposition | Operational definition | Required supervision/intervention | CMNIST mapping | Source-only usable? |
|---|---|---|---|---|---|
| Arjovsky et al., *Invariant Risk Minimization* (2019) | invariant predictor vs environment-dependent association | shared optimal predictor across environments | environment labels and multi-environment risks | digit label task vs source color correlation | Yes |
| Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation* (2021) | stable risk vs environment-varying risk | penalize dispersion of environment risks | environment labels | source color/noise coordinates | Yes |
| Sun & Saenko, *Deep CORAL* (2016) | domain-specific feature statistics vs aligned statistics | match means/covariances across domains | domain labels | source color changes as domain statistic shift | Yes |
| Li et al., *Learning to Generalize: Meta-Learning for Domain Generalization* (2018) | meta-train versus meta-test domain variation | differentiate through held-out domain simulation | multiple source domains | source environment color correlation | Yes |
| Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization* (2021) | transferable signal vs domain-specific variation | controlled domain-generalization comparison | domain labels; benchmark shifts | fixed CMNIST shift registry | Yes |
| Sagawa et al., *Distributionally Robust Neural Networks* (2020) | minority/group signal vs majority shortcut | worst-group risk | group annotations | color-label groups | Yes |
| Peters, Bühlmann & Meinshausen, *Causal Inference by Using Invariant Prediction* (2016) | invariant causal mechanism vs unstable association | conditional law invariant over environments | environments plus causal assumptions | digit mechanism candidate, color association | Partly |
| Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* (2019) | latent factors vs nuisance factors | non-identifiability without inductive bias | interventions/inductive bias | negative control against blind recovery | Limitation only |
| Ahmed et al., *DecAug: Augmenting Models for Domain Generalization* (2021) | category-related vs context-related representation | category/context perturbation and probes | category labels and controlled augmentation | clean digit task vs counterfactual color response | Partly |

## Audit boundary

CMNIST semantic axes are fixed by `smooth_world5.py` before post-hoc target outcomes are read. Causal, invariant, content, style, context, and nuisance are not interchangeable. Decompositions are imported only when their operation is represented by an existing coordinate; support/diversity shifts remain `NOT-OPERATIONAL-IN-CURRENT-CMNIST`. Blind clustering is not used to define semantic axes.
