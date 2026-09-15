# Counterfactual Repairability / Controllability

This directory contains the Phase-0 validation of the research turn toward
counterfactual repairability.  It is an experimental layer on top of the
frozen `A : U -> Z` and `O_S : U -> Y` response/source-information objects.

The first object is the centered adaptive regret

\[
R_{\rm adap}(\Pi)=\tfrac12\lVert A+\Pi O_S\rVert_{\rm op}^2.
\]

For a declared finite-dimensional response-controller class \(\mathcal C\),
the code computes `R_info`, `R_class_star`, the control gap, an optimizer, a
membership residual, and the inherited spectral-slack certificate.  It does
not assign semantic labels such as representation failure, and it does not
use target risk.

## Phase-0 entry points

- `src/ood_repr_reg/counterfactual_repairability/classes.py`: response-space
  controller classes, functional budgets, basis transport, and Jacobian
  induction.
- `src/ood_repr_reg/counterfactual_repairability/optimize.py`: class-optimal
  centered repair and solver certificates.
- `src/ood_repr_reg/counterfactual_repairability/bridges.py`: actual-`Pi`
  membership and `J_B L` locality bridge.
- `src/ood_repr_reg/counterfactual_repairability/diagnostics.py`: three toy
  checks, random nested classes, coordinate transport, and grid audit.
- `formalization/OODRelevance/RestrictedRepair.lean`: abstract Lean check of
  class-optimality inheritance, nested-class monotonicity, decomposition, and
  the zero-observation identity.

Run the deterministic Phase-0 report from the repository root:

```bash
PYTHONPATH=src python round3_redesign/task3_counterfactual_repairability/scripts/run_phase0.py
```

The generated JSON and Markdown report live under `results/`.  This stage must
pass before any 220-row panel or CMNIST intervention bridge is attempted.

## Scientific boundary

The Phase-0 result is an internal consistency/falsification gate.  It is not a
target-risk theorem, an algorithm claim, a causal mechanism claim, or evidence
that the class definitions match a training procedure.
