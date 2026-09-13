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

Status: this is a research architecture, not an accepted universal theorem.
The current evidence supports several irreducible mechanism types and requires
method-specific `Omega -> B` translation lemmas.
