# CMNIST-VIS-001 Independent Code Audit

Act only as a read-only experiment code auditor. Do not edit files, run the
main experiment, reinterpret the research question, or suggest a larger study.

Read:

- `docs/experiments/CMNIST-VIS-001_preregistered.md`
- `configs/cmnist_vis_001_smoke.json`
- `configs/cmnist_vis_001_main.json`
- `src/ood_repr_reg/cmnist_feature_probe.py`
- `src/ood_repr_reg/run_cmnist_feature_probe.py`
- `tests/test_cmnist_feature_probe.py`

Check the following:

1. Color counterfactuals preserve grayscale shape and labels.
2. Source correlation, target sign flip, and split separation are implemented
   as registered.
3. ERM, joint L2, scalar-scale IRMv1, and CORAL objectives match their stated
   definitions and receive identical seed-specific source data and schedules.
4. No target or counterfactual-probe quantity enters training or strength
   selection. The full path may be reported without selecting a strength.
5. Latent color response is computed after balanced-probe whitening; prediction
   color response equals the squared binary-logit-margin change under the exact
   color counterfactual.
6. Task signal, task/color overlap, balanced accuracy, and trajectory axes are
   described no more strongly than their implementation supports.
7. Saved figures are derived from the same numeric path records as the metrics.
8. Tests cover the highest-risk identities and a network-free data path.

Return exactly one verdict followed by concise evidence:

```text
AUDIT_PASS
REVISE_CODE_ONCE
STOP_CODE_INVALID
```

Use `REVISE_CODE_ONCE` only for bounded implementation defects. Use
`STOP_CODE_INVALID` for leakage or a design/implementation mismatch that makes
the probe uninterpretable.
