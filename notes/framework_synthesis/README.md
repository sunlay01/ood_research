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
13. `risk_representation_hard_gate.md` and
    `experiments/risk_representation_tests.py` — Stage 11's restricted exact
    supervised representation, identifiability attacks, residual audit, and
    metric/gauge revision gate.
14. `stage11r_metric_gauge_revision.md` and
    `experiments/risk_representation_revision_tests.py` — the completed
    primal/dual, exposure-operator, annihilator-quotient, support-first, and
    metric-covariance revision.
15. `stage12_population_master_theorem.md` and
    `experiments/stage12_population_theorem_tests.py` — the finite-dimensional
    theorem package: exact transfer, sharp support, quotient ambiguity,
    blind-direction impossibility, and exposed-span domination.
16. `stage13_regularizer_translation.md` and
    `experiments/stage13_regularizer_translation_tests.py` — method-level
    translations, mechanism typing, cross-method consequences, and the
    `PARTIAL-UNIFICATION` gate.
17. `stage13r_environmental_sensitivity_master.md`,
    `stage13r_moment_alignment_separation.md`, and
    `stage13r_novelty_audit.md` — the frozen environmental-sensitivity master
    functional, its path/local-to-global theorem, separation from Moment
    Alignment, and the exact-overlap audit. Deterministic checks are in
    `experiments/stage13r_esf_master_tests.py`; the stable algebraic Lean slice
    is in `lean/OodTheoryVerification/Stage13R/Basic.lean`.
18. `stage13r1_master_invariance_audit.md`,
    `stage13r1_excess_vs_moment_alignment.md`,
    `stage13r1_fixed_tangent_reaudit.md`, and
    `stage13r1_novelty_audit.md` — the nuisance/invariance hard gate, the
    excess-risk equivalence analysis, the fixed physical tangent comparison,
    and the prior-art audit. Deterministic checks are in
    `experiments/stage13r1_master_invariance_tests.py`; stable algebraic
    identities are in `lean/OodTheoryVerification/Stage13R1/Basic.lean`.
19. `master_functional_search.md` and `master_functional_discrepancy_audit.md`
    — the first post-ESF search over transfer, risk-landscape, robust-regret,
    and optimality-map candidates, including the exact reduction of the natural
    sup-norm quotient to classical loss-class discrepancy. Deterministic checks
    are in `experiments/master_functional_search_tests.py`.

Status: this is a research architecture, not an accepted universal theorem.
The current evidence supports several irreducible mechanism types and requires
method-specific `Omega -> B` translation lemmas.

The exposure-geometry files remain a `PROBE`: the finite-dimensional affine
support bound survives deterministic checks, but the typed calculus is not
superseded until non-vacuous method translation lemmas and an out-of-sample
optimizer mapping are established. Stage 11R is complete with decision
`ADVANCE-TO-STAGE-12`: the exact quadratic state is valid and the split geometry
is explicitly subordinate to a declared metric/gauge, while support and
annihilator statements remain coordinate-free.
Stage 12 is now complete with decision `ADVANCE-TO-STAGE-12.5`; only the exact
novelty audit is authorized next.
Stage 13 is complete with decision `PARTIAL-UNIFICATION`: several methods share
the same state and exact/restricted bridges, but IRMv1 and Fishr prevent a
universal exact claim.
Stage 13R is complete with decision `REVISE-ESF-MASTER`: the environmental-side
sensitivity functional is coherent and separated from parameter-side Moment
Alignment, but source-only global control requires an explicit path envelope or
coverage assumption, and derivative-rich methods remain restricted. No Stage 14,
new regularizer, or large benchmark is authorized yet.
Stage 13R.1 supersedes that provisional status with decision
`FAIL-ESF-AS-INDEPENDENT-MASTER`: raw ESF fails additive-nuisance invariance,
excess ESF is Moment Alignment-equivalent at the endpoint/path level, and the
pairwise quotient is relative-risk only without a new fixed-tangent theorem.
The valid Stage 12/13 results remain preserved; no Stage 14 or new regularizer
is authorized.
The subsequent master-functional search is now active. Its first screen stops
transfer and optimizer-diameter candidates, and marks the risk-landscape
quotient `REVISE`: the natural sup-norm quotient is exactly half classical
pairwise loss discrepancy, so a non-arbitrary alternative geometry and a new
cross-method theorem are required before any master claim.
