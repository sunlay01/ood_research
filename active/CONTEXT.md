# TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK Context

This isolated expansion keeps the previously repaired CMNIST A/O/Pi survey semantics and broadens the algorithm panel. The R5 base world remains `(0.2, 0.1, 0.9, 0.25, 0.25)` and tangent coordinates remain displacements from that base.

Trusted invariant components:

- Corrected CPU-minimal CMNIST data/model semantics and source batch schedule.
- Smooth four-outcome expectation world, task response `A = H_S^(-1/2) D grad(R_T - R_S)`, source-only method-independent `O`, functional banks, normalization, and blind grouping.
- ERM/IRMv1 reference hashes from the accepted counterfactual audit manifest.

Algorithm structure:

- `algorithms/erm.py`, `irmv1.py`, `vrex.py`, and `coral.py` remain behavior-preserving.
- New methods live in dedicated files: `fishr.py`, `mldg.py`, `weight_nuclear.py`, `feature_nuclear.py`; `stable_rank.py` is diagnostic-only.
- `method_trainer.py` and `full_response.py` call algorithm-owned step interfaces rather than reimplementing method math.

Evaluation boundaries:

- Target/evaluation data may enter A, evaluation functional banks, and post-hoc performance only.
- Target rows cannot select source fit, direction, continuation, normalization, grouping, rank coefficient, or method inclusion.

Canonical state remains unchanged; completion writes only `active/STATE_DELTA.md`.
