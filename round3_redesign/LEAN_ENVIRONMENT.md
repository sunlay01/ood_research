# Lean Environment

The repository uses the official `elan` installation at `~/.elan/bin` and
pins the current formalization projects to Lean 4.33.1 and mathlib v4.33.1.
The project manifests resolve mathlib to the same commit:

```text
0df444a360eaa60ab8c11dca51a86af692955474
```

Use the shared shell setup before working in any Lean project:

```bash
source scripts/lean_env.sh
```

The setup exports `MATHLIB_CACHE_DIR`, which is mathlib's official shared
directory for downloaded `.ltar` build artifacts. This cache is safe to share
between projects with the same Lean/mathlib revision.

Each Lake workspace must still keep its own `.lake/packages` and
`.lake/build`. Those directories contain project-specific dependency checkouts
and build outputs; sharing one mutable build directory can mix manifests or
root hashes and create harder-to-debug failures.

For a new or repaired workspace, use the pinned manifest and fetch the
official compiled cache before building:

```bash
cd round3_redesign/3A_T_theory/lean
lake exe cache get
lake build OODRelevance

cd ../../3C_regularizer_control/lean
lake exe cache get
lake build OODRegularizer
```

Do not run `lake update` during normal builds. It intentionally resolves newer
dependency revisions and can change the manifest. Run it only when a deliberate
dependency upgrade is being made and the resulting manifest is reviewed.

The old `3A_T_theory/lean/.lake2` directory is not a dependency of the current
`lakefile.toml`; it is a stale alternate workspace and should not be used.
