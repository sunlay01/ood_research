# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT Preregistered Design

task_id: `TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT`
git_head_before_run: `50d435b403bbda00427b9cce38258066eacc40db`
config_sha256: `a1083f49401660ecbb1f1b19e75848ad24612750b34e5255131945ebbbbbe60d`
methods: `ERM`, `IRMv1`
seeds: `10,11,12,13,14`

## Question

Decompose the corrected CPU-minimal ERM/IRMv1 target-accuracy gap into representation content versus final-head color usage.

## Fixed Inputs

- Config: `configs/task3_cmnist_cpu_minimal.json`
- Reference corrected run table: `round3_redesign/task3_cmnist_cpu_minimal/results/main_runs.csv`
- Data/model/trainer: `src/ood_repr_reg/task3_cmnist_cpu_minimal/`

## Reconstruction Gate

If exact final checkpoints are absent, reconstruct only ERM/IRMv1 seeds `10..14` with the corrected CPU-minimal trainer, then require final source env0 accuracy, source env1 accuracy, target accuracy, and prediction-color agreement to match the existing corrected run within `1e-6`.

## Counterfactual Probe

Use the full held-out target split post-hoc. Recover grayscale by summing the two 14x14 color channels and construct red `(g,0)` and green `(0,g)` inputs. Original target color is not used in the intervention.

## Diagnostics

- Representation content: whitened latent color response, task signal, task/color overlap, balanced clean accuracy.
- Head usage: scalar-logit color response, sigmoid probability color response, task-head margin, counterfactual consistency, counterfactual flip rate.

## Descriptive Category Rule

Before observing this audit's diagnostics, fixed descriptive gates are: representation is material if IRMv1 has >=1.25x task signal, >=5pp balanced clean accuracy gain, or <=0.80x latent color response in at least 4/5 paired seeds. Head usage is material if IRMv1 has <=0.80x scalar/probability color response, >=5pp higher counterfactual consistency, or >=5pp lower flip rate in at least 4/5 paired seeds. Both material means `MIXED-DECOMPOSITION`; neither means `DESCRIPTIVE-INCONCLUSIVE`.

## Interpretation Ceiling

This audit is descriptive only. It does not establish causality, source identifiability, a new objective, frozen theory validation, novelty, or restoration of old CMNIST empirical results.
