# TASK-AOPI-MULTIMETHOD-MECHANISM-SURVEY

task_id: `TASK-AOPI-MULTIMETHOD-MECHANISM-SURVEY`

goal: `Run an isolated descriptive A/O/Pi survey of ERM, IRMv1, VREX, and CORAL on corrected CMNIST.`

state_write_authorized: false

scientific_status: validation / falsification gate only

methods: ERM, IRMv1, VREX, CORAL

primary seeds: 10, 11, 12, 13, 14

inputs: `configs/task3_aopi_multimethod_mechanism_survey.json`, corrected CPU-minimal data/model code, and the accepted ERM/IRM reconstruction manifest.

hard constraints:

- Use the R5 displacement world with smooth four-outcome expectations and e1..e5 as primary basis.
- Define O from source env0/env1 risks only, without method-specific penalties.
- Use source-only training and Pi_full continuation; target/evaluation is post-hoc only.
- Keep every fixed method and seed even when target performance is poor.
- Do not add OURS, Fish/Fishr, MLDG, HA, CMA, target tuning, or canonical-state edits.

outputs: `round3_redesign/task3_aopi_multimethod_mechanism_survey/`

Interpretation ceiling: descriptive response differences only; no semantic mechanism recovery, causal claim, theory validation, or new algorithm claim.
