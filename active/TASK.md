# TASK3-CMNIST-CPU-MINIMAL

task_id: `TASK3-CMNIST-CPU-MINIMAL`

goal: `Run a CPU-only ColoredMNIST proof-of-concept comparing GRAD identity-metric head-gradient disagreement with LOCAL_RESPONSE inverse-Hessian-metric head-gradient disagreement.`

state_write_authorized: false

Task 3 scientific status: NOT YET DECIDED

allowed files:

- `active/TASK.md`
- `active/CONTEXT.md`
- `active/STATE_DELTA.md`
- `configs/task3_cmnist_cpu_minimal.json`
- `src/ood_repr_reg/task3_cmnist_cpu_minimal/`
- `src/ood_repr_reg/run_task3_cmnist_cpu_minimal.py`
- `tests/test_task3_cmnist_cpu_minimal.py`
- `round3_redesign/task3_cmnist_cpu_minimal/`

Stage A gate:

- Run only `ERM` and `IRMv1` on seeds `0,1,2`.
- Continue only if IRMv1 mean target accuracy is at least `0.50`, ERM mean target accuracy is at most `0.35`, and IRMv1 minus ERM is at least `0.20`.

Stage B methods:

- `ERM`
- `IRMv1`
- `GRAD`
- `LOCAL_RESPONSE`

Historical reopen: none.
