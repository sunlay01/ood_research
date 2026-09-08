# Environment-Family Layer Refactor

Run the audit with:

```bash
PYTHONPATH=src python -m ood_repr_reg.run_round3r_environment_family
```

The immutable result tables are written below `results/`.  The runner keeps
the legacy compatibility path and the source-induced family separate.
