# Literature directory protocol

`literature/ledger/` is the authoritative, compact entry point for the OOD/DG theory line. A future model should read the files below in this order before opening any paper or historical note:

1. [`../research_program.md`](../research_program.md)
2. [`../notes/foundations/problem_formulation.md`](../notes/foundations/problem_formulation.md)
3. [`../notes/foundations/representation_methodology.md`](../notes/foundations/representation_methodology.md)
4. [`../notes/foundations/open_questions.md`](../notes/foundations/open_questions.md)
5. [`ledger/family_map.md`](ledger/family_map.md)
6. [`ledger/representation_map.csv`](ledger/representation_map.csv)
7. [`ledger/compact_papers.csv`](ledger/compact_papers.csv)
8. [`ledger/theorem_map.csv`](ledger/theorem_map.csv)
9. [`ledger/unresolved_questions.md`](ledger/unresolved_questions.md)

## Retrieval policy

- `ledger/` is the current authoritative knowledge base. It is the default literature input.
- `papers/` is the original-evidence layer: PDFs, source URLs, and the download manifest. Open a PDF only when a ledger entry or unresolved question requires decisive verification.
- `ledger/details/` contains selective detailed cards for important papers. These are secondary retrieval, not default input.
- Other files directly under `literature/` are historical or intermediate research notes. They are retained for provenance and may contain useful reasoning, but must not be recursively ingested by default. Consult them only when the ledger points to a specific unresolved issue or when auditing how a conclusion was reached.

## Historical/intermediate notes

The following files are retained but are not authoritative replacements for the ledger:

- `audit_round2.md`
- `deep_read_round3.md`
- `formalization_lessons.md`
- `generalization_proof_techniques.md`
- `irm_theory_audit.md`
- `regularization_taxonomy.md`
- `regularization_taxonomy.csv`
- `theorem_anatomy_table.csv`
- `theory_literature_synthesis.md`
- `unified_formalization_frameworks.md`

The files `provenance.md`, `reading_queue.md`, `references.bib`, and `papers/download_manifest.csv` remain operational metadata. They are not substitutes for the compact ledger.

## Authority rule

When an older note conflicts with the ledger, treat the ledger's compact entry and evidence pointers as the current working position, then verify the cited detailed card or original PDF. Do not silently merge duplicate summaries.
