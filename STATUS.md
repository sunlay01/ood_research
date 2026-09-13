# OOD theory status

## Current branch

`ood-theory` is clean and synchronized with `origin/ood-theory`.

## Epistemic status

The project has completed two evidence stages:

1. [`notes/evidence/ood_bridges/`](notes/evidence/ood_bridges/) — algorithm-specific OOD/DG proof bridges.
2. [`notes/evidence/framework_construction/`](notes/evidence/framework_construction/) — cross-domain lessons on how mature fields construct formal theories.

The resulting construction guide is [`formalization_framework_construction_guide.md`](notes/evidence/framework_construction/synthesis/formalization_framework_construction_guide.md). It ends with `CONSTRUCTION-GUIDE-READY` and does not select the project's final framework.

## What is authoritative

For a future GPT-6 design pass, read in this order:

1. `research_program.md`
2. `notes/foundations/*`
3. `notes/evidence/ood_bridges/extraction/*`
4. `notes/evidence/ood_bridges/synthesis/final_report.md`
5. `notes/evidence/framework_construction/extraction/*`
6. `notes/evidence/framework_construction/synthesis/formalization_framework_construction_guide.md`

Do not recursively read `archive/` by default. It contains failed or superseded reasoning retained for provenance. In particular, `archive/failed_formalization_round1/` is marked **FAILED — PREMATURE FRAMEWORK CONSTRUCTION**.

## Proof status

`proofs/active/` is intentionally empty. Existing source-only, robust, and conditional theorem scaffolds are in `proofs/baselines/`; their audit is in `proofs/audits/`. No theorem has been promoted as the project's active result.

## Next authorized step

Use both evidence bases to generate and compare candidate architectures under a new design task. Keep candidates under `notes/framework_design/`, with requirements, stress tests, comparison and decision separated from evidence.
