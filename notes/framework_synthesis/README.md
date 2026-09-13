# Framework synthesis entry point

This directory is the output of the bottom-up synthesis round requested in
`gpt6_bottom_up_ood_framework_prompt_final.md`.

Read in this order:

1. `method_cards.md` — exact training objects and evidence labels;
2. `mechanism_primitives.md` — bottom-up primitive extraction;
3. `common_structure_hypotheses.md` — candidate compression hypotheses;
4. `synthesis_attack.md` — counterexamples and assumption attacks;
5. `revised_synthesis.md` — what survived and what did not;
6. `candidate_framework_01.md` — the typed bridge-calculus candidate;
7. `framework_stress_test.md` — method-by-method reconstruction;
8. `final_framework.md` and `validation_agenda.md` — current proposal and next tests.
9. `source_exposure_geometry.md` and `exposure_geometry_novelty_audit.md` — the
   proposed inner mathematical core and its novelty/feasibility audit.
10. `regularization_ood_bound_methodology.md` and
    `minimal_population_experiment.md` — the finite-dimensional mother bound
    and its falsification probe.
11. `operator_gauge_master_theorem.md` — the candidate PSD-operator and convex-
    gauge duality theorem that separates sensitivity from target coverage.
12. `coverage_orientation_theorem.md` — Stage 10's target-relative orientation,
    rank/domain-count barriers, fixed-spectrum optimization, and negative controls.

Status: this is a research architecture, not an accepted universal theorem.
The current evidence supports several irreducible mechanism types and requires
method-specific `Omega -> B` translation lemmas.

The exposure-geometry files remain a `PROBE`: the finite-dimensional affine
support bound survives deterministic checks, but the typed calculus is not
superseded until non-vacuous method translation lemmas and an out-of-sample
optimizer mapping are established.
