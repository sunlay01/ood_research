# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND Protocol

## Scope

This is an isolated first-round capability audit for corrected CPU-minimal ColoredMNIST. It evaluates only `C_coverage`, `C_separate`, and `C_select` on frozen ERM/IRMv1 encoders from seeds `10..14`.

## Guardrails

- No new regularizer.
- No algorithm claim.
- No source-identifiability claim.
- No causal/additive decomposition claim.
- No theory validation.
- No D/E execution.

## Input Gate

The checkpoint manifest, checkpoint SHA256 values, config SHA256, model architecture, parameter hashes, and corrected ERM/IRMv1 target gap are checked before capability metrics are computed. Verified mean target accuracy: ERM `0.109800`, IRMv1 `0.669140`, gap `0.559340`.

## Experiments

- A coverage: source ridge probe plus diagnostic oracle clean ridge coverage.
- B separability: color-response subspace removal at ranks `[0, 1, 2, 4, 8, 16, 32, 64]`.
- C selection: head-only ERM, head-only IRMv1, and diagnostic oracle clean heads with frozen encoders.
