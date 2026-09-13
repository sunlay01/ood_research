# Minimal population experiment

This is a deterministic falsification probe for the source-exposure geometry
proposal. It uses exact finite-dimensional embeddings, so no sampling claim is
being made.

Run:

```bash
python notes/framework_synthesis/experiments/exposure_geometry_tests.py
```

The script checks the `r^T K^dagger r` projection identity, quadratic scaling of
`Var_e(R_e)` under `delta_e -> epsilon delta_e`, invariance of the pseudoinverse
quantity, and separation of matched-V-REx source systems with different
exposure rank. A failure is a reason to stop and repair the formalization before
adding finite-sample or optimizer claims.

Expected qualitative outcome:

| system | `rank(C_S)` | matched V-REx | exposed term | nullspace term |
|---|---:|---:|---:|---:|
| rank-1 | 1 | yes | `1/sqrt(2)` | `1/sqrt(2)` |
| rank-2 | 2 | yes | `1/sqrt(2)` | `0` |

For `rho=1` and `kappa=0.25`, the resulting support bounds are approximately
`0.883883` and `0.707107`. The gap is generated solely by the unexposed
direction, which scalar V-REx cannot see.

This probe supports the distinction between source-risk variance and
target-relevant sensitivity. It does not prove that V-REx, MMD, and DRO are
already exact corollaries. That requires separate translation lemmas and an
explicit target-family calibration theorem.
