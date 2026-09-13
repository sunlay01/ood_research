# Reading queue

## Tier 1: foundational theory

| Citation | Link | Why / question | Inspect | Priority |
|---|---|---|---|---|
| Ben-David et al. (2010) | https://doi.org/10.1007/s10994-009-5152-4 | DA decomposition | Theorem 2, HDeltaH and lambda | high |
| Mansour, Mohri & Rostamizadeh (2009) | https://arxiv.org/abs/0902.3430 | loss-class discrepancy | discrepancy definition/theorem | high |
| Blanchard, Lee & Scott (2011) | https://papers.nips.cc/paper/4312-generalizing-from-several-related-classification-tasks-to-a-new-unlabeled-sample | source-only DG family | domain sampling assumptions | high |
| Blanchard et al. (2021) | https://jmlr.org/papers/v22/20-305.html | marginal transfer extension | admissible marginal family | high |
| Gretton et al. (2012) | https://jmlr.org/papers/v13/gretton12a.html | MMD/RKHS proof language | concentration theorem | high |
| Bousquet & Elisseeff (2002) | https://jmlr.org/papers/v2/bousquet02a.html | stability limits | uniform stability theorem | medium |
| Bartlett et al. (2017) | https://doi.org/10.1073/pnas.1618451114 | norm/spectral complexity | margin bound | medium |
| Duchi & Namkoong (2021) | https://doi.org/10.1214/20-AOS2004 | f-DRO duality | robust risk dual | high |
| Esfahani & Kuhn (2018) | https://doi.org/10.1287/opre.2017.1673 | Wasserstein robust theory | duality and finite sample | high |
| Germain et al. (2020) | https://doi.org/10.1016/j.neucom.2019.10.105 | PAC-Bayes transfer | disagreement bound | medium |
| Peters, Bühlmann & Meinshausen (2016) | https://doi.org/10.1111/rssb.12167 | causal invariance | ICP theorem/assumptions | high |
| Wang, Bai, Yang, Xu & Liang (2026) | https://jmlr.org/papers/volume27/25-0399/25-0399.html | direct-sum Tri-Space representation and fine-grained target-risk bound | Theorems 1--3; Assumptions 1--2; Definition 4 localized discrepancy | high |

## Tier 2: algorithm-specific theory

| Citation | Link | Question | Priority |
|---|---|---|---|
| Arjovsky et al. (2019) | https://arxiv.org/abs/1907.02893 | ideal IRM vs IRMv1 Eq. 3 | high |
| Rosenfeld et al. (2021) | https://openreview.net/forum?id=BbNIbVPJ-42 | IRM counterexamples/failure | high |
| Kamath, Tangella, Sutherland & Srebro (2021) | https://proceedings.mlr.press/v130/kamath21a.html | limits of practical IRMv1 / invariance capture | high |
| Krueger et al. (2021) | https://proceedings.mlr.press/v139/krueger21a.html | V-REx extrapolation | high |
| Rame et al. (2022) | https://proceedings.mlr.press/v162/rame22a.html | Fishr gradient variance object | medium |
| Sun & Saenko (2016) | https://arxiv.org/abs/1607.01719 | CORAL moment object | medium |
| Li et al. (2018) | https://arxiv.org/abs/1710.03463 | bilevel/meta objective | medium |
| Lai & Wang (2024) | https://proceedings.mlr.press/v235/lai24c.html | TV functional interpretation of IRM | high |
| Wang, Wu & Zhang (2024) | https://doi.org/10.1609/aaai.v38i14.29497 | training-domain minimax lower bound | high |
| Cao & Chen (2024) | https://doi.org/10.1609/aaai.v38i10.28994 | PAC-Bayes Mixup DG bound | medium |

## Tier 3: unified/recent directions

| Citation | Link | Why | Priority |
|---|---|---|---|
| Gulrajani & Lopez-Paz (2021) | https://arxiv.org/abs/2007.01434 | empirical taxonomy and limits of DG | medium |
| Rothenhäusler et al. (2021) | https://doi.org/10.1111/rssb.12410 | anchor/robust shift unification | medium |
| Shui, Wang & Gagné (2022) | https://doi.org/10.1007/s10994-021-06080-w | unified INV representation regularization | high |
| Wang & Veitch (2022) | https://openreview.net/forum?id=-l9cpeEYwJJ | causal unified invariance view | high |
| Rivasplata et al. (2018) | https://papers.neurips.cc/paper/8134-pac-bayes-bounds-for-stable-algorithms-with-instance-dependent-priors | PAC-Bayes stability (not DA) | medium |
| Wang, Bai, Yang, Xu & Liang (2026) | https://jmlr.org/papers/volume27/25-0399/25-0399.html | Tri-Space latent representation; fine-grained invariance/diversity bound | high |
| Liu, Yu, Wang & Liao (2024) | https://doi.org/10.1109/ISIT57864.2024.10619471 | information-theoretic OOD bound | high |
| Liu, Yu, Wang & Liao (2025) | https://doi.org/10.1109/TIT.2025.3598722 | information-theoretic OOD + SGLD | high |
| Wu, Liu, Cui & Wu (2024) | https://doi.org/10.52202/079017-2325 | multicalibration beyond covariate shift | medium |
