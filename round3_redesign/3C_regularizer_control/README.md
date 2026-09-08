# 3C Regularizer Control

Run with:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_round3r_3c
```

The primary output is the per-response, per-method, per-lambda control matrix.
Exposure and intervention labels are post-hoc fields only.  The six 3B blind
clusters are never used as semantic units.
