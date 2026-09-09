# Baseline Method Identity

Pinned upstreams are recorded in `upstream_manifest.json`.

| method | upstream authority | required identity | current status |
|---|---|---|---|
| ERM | Facebook IRM and DomainBed | source empirical risk minimization only | `pinned_and_controlled` |
| IRMv1 | Facebook IRM Colored MNIST | scalar scale penalty, anneal at step 100, penalty weight 10000, whole-loss rescale after anneal | `recovered` |
| IGA | DomainBed `b93c22a1cfc3b2428398272c1a116c8de1f4139e` | full-network per-environment gradients, mean-gradient penalty, default penalty 1000 | `native_completed` |
| Fish | DomainBed `b93c22a1cfc3b2428398272c1a116c8de1f4139e` plus YugeTen/fish mechanism | clone, sequential inner-domain updates, carried inner optimizer state, outer interpolation with meta_lr 0.5 | `native_completed` |
| HEAD_GRADIENT_VARIANCE_SURROGATE | local Task3 diagnostic only | head-only gradient variance; fixed penalty-subset history | NOT A REPRODUCTION OF IGA OR FISH |

The old public label `UNPRECONDITIONED_GRAD_ALIGN` is superseded by `HEAD_GRADIENT_VARIANCE_SURROGATE` in this gate. Preserved old numeric rows must be read only as a negative diagnostic, not as IGA/Fish evidence.
