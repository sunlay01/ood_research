# Prior Art Exact Object Audit

| paper | exact equation/object | same as our proposed object? | equivalent only under assumptions? | different? | implementation implication | novelty implication |
|---|---|---:|---:|---:|---|---|
| MLDG, Li et al. 2018, https://arxiv.org/abs/1710.03463 | meta-train/meta-test objective; first-order variants involve gradient alignment across domains | no | no exact inverse-H metric found in this bounded audit | yes | include only if faithful compact implementation is added; otherwise do not relabel GRAD/LR as MLDG | `RELATED-BUT-DIFFERENT` |
| Fish, Shi et al. 2021, https://arxiv.org/abs/2104.09937 | inter-domain gradient matching / gradient dot-product style update | no | related first-order gradient matching | yes | HEAD_GRADIENT_VARIANCE_SURROGATE is a neutral baseline, not automatically Fish | `RELATED-BUT-DIFFERENT` |
| Fishr, Rame et al. 2021, https://arxiv.org/abs/2109.02934 | domain-level gradient variance matching; connects gradient variance to Fisher/Hessian motivation | no | related through Fisher/gradient-variance geometry | yes | do not call LOCAL_RESPONSE Fishr; use as strong related baseline family | `RELATED-BUT-DIFFERENT` |
| Hessian Alignment / classifier-head Hessian analyses, e.g. https://arxiv.org/abs/2308.11778 | Hessian/gradient structure for DG/generalization analysis | unresolved exact implementation match | possible only after equation-level comparison | yes in this bounded audit | no novelty claim; record as closest Hessian-geometry neighbor | `UNRESOLVED` |
| Moment/curvature alignment family | moment or Hessian matching rather than inverse-H gradient-disagreement penalty | no | no | yes | keep internal name `LOCAL_RESPONSE` | `NO-EXACT-MATCH-FOUND` for this exact object within the bounded checked set |

Bounded conclusion: the implemented object is reported under a neutral internal name. No algorithmic novelty claim is made by this task.
