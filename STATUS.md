# OOD theory status

## Current branch

`ood-theory` contains the completed problem-first formalization audit. The
working decision is `YES-WITH-LOCAL-THEOREM`; no new universal representation
has been accepted.

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

## Current design result

The design audit is recorded in `notes/framework_design/`. Read
`adequacy_audit.md`, `problem_instances.md`, `reuse_test.md`,
`information_sufficiency.md`, `bottlenecks.md`, the route files,
`comparison.md`, `novelty_check.md`, and `decision.md` in that order. Existing
IPM, conditional/causal, DRO, and meta-domain formalisms are adequate for the
declared problem instances; the remaining work is an algorithm-local theorem,
an explicit ambiguity term, or an impossibility result.

The subsequent bottom-up synthesis is recorded in
[`notes/framework_synthesis/`](notes/framework_synthesis/). Its entry point is
`README.md`; it starts from exact method cards, attacks candidate primitives,
and proposes a typed bridge calculus as a research architecture. This does not
promote a theorem to `proofs/active/` and does not replace the problem-first
audit.

An exposure-geometry probe is documented in
`notes/framework_synthesis/source_exposure_geometry.md` and
`exposure_geometry_novelty_audit.md`. It treats the typed calculus as an outer
scaffold and tests a source-domain covariance operator plus a target-relevant
task functional as a possible inner core; it is not yet accepted as the final
theory.

## Next authorized step

Choose one concrete theorem route, preferably a bounded-loss MMD/conditional
bound or a restricted IRMv1 translation. Keep `proofs/active/` empty until a
proof is independently audited and promoted.
