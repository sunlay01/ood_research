# Intentionally unmoved legacy paths

The source of truth remains in the existing repository layout to preserve imports, artifact hashes, and historical provenance. In particular, `src/`, `configs/`, `tests/`, `artifacts/`, `round3_redesign/`, `formalization/`, `papers/`, and `docs/` are intentionally not mass-moved in this refactor.

The new routing directories do not duplicate these files. Use `migration_map.md` to find the canonical path. A future physical move requires its own reproducibility audit and is outside this refactor.
