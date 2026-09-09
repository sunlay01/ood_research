# TASK3-BASELINE-FIDELITY-RECOVERY Context

Current question:

- Are the ERM, IRMv1, IGA, and Fish baselines faithful enough to unblock a later Task 3 local-response experiment?

Required upstream authorities:

- `facebookresearch/InvariantRiskMinimization@fc185d0f828a98f57030ba3647efc7394d1be95a`
- `facebookresearch/DomainBed@b93c22a1cfc3b2428398272c1a116c8de1f4139e`
- `YugeTen/fish@333efa24572d99da0a4107ab9cc4af93a915d2a9`
- optional Fishr only: `alexrame/fishr@7b8fdf1e0b15226ded9b58efd37698e74e616ab7`

Required baseline identities:

- Facebook IRM Colored MNIST: `digit < 5`, label noise `0.25`, train color flips `[0.2, 0.1]`, target flip `0.9`, 2-channel 14x14 MLP, BCE-with-logits, Adam, L2 `0.001`, `501` steps.
- IRMv1: scalar scale penalty, anneal at step `100`, penalty weight `10000`, whole-loss rescale after anneal.
- IGA: DomainBed full-network per-environment gradients, mean-gradient penalty, default penalty `1000`.
- Fish: clone model, sequential inner-domain updates, carried inner optimizer state, outer interpolation/meta-update with default `meta_lr=0.5`.

Known exclusions:

- Do not run or interpret `LOCAL_RESPONSE`, `RANDOM_METRIC`, `SHUFFLED_LOCAL_RESPONSE`, `RESP2`, or any new regularizer.
- Do not use target data or target metrics for baseline training, hyperparameter choice, checkpointing, or method selection.
- Do not claim a Task 3 scientific verdict; the required status is `Task 3 scientific verdict: NOT YET RUN`.
- `HEAD_GRADIENT_VARIANCE_SURROGATE` is a local head-only diagnostic and is not IGA or Fish.

Relevant code entrypoints:

- `src/ood_repr_reg/task3_baseline_fidelity/`
- `src/ood_repr_reg/run_task3_baseline_fidelity.py`
- `tests/test_baseline_fidelity.py`

Historical reopen: none.
