# Reusable bridge-theorem patterns

## Pattern 1: discrepancy transfer

If the loss class is contained in a witness class, then
`|R_Q(f)-R_P(f)| <= D_G(Q,P)` (or the standard HΔH bound with `lambda*`). Reusable across HΔH, loss-class IPM and MMD; target discrepancy/joint error remains.

## Pattern 2: uncertainty support

For `Q in U`, `R_Q(f) <= sigma_U(ell_f)`. If `U` is a finite source-mixture set, empirical robust risk differs from population robust risk by the maximum per-domain uniform deviation. Reusable across GroupDRO and declared-set DRO.

## Pattern 3: two-level domain concentration

For `P_e iid~Pi`, `g_f(P)=R_P(f)`, bound `E_Pi g_f` by empirical domain risks plus domain-level and within-domain complexity. Reusable for ERM/V-REx/GroupDRO only as a target-family theorem, not arbitrary-target control.

## Pattern 4: conditional decomposition plus contraction

`R_Q-R_P = covariate term + conditional-label term`; under invariant conditional mechanisms and a contraction coefficient, the conditional term vanishes or is contracted. Reusable for ICP/anchor/INV/Tri-Space only under their structural assumptions.

## Pattern 5: objective translation

`Omega_algorithm -> population functional` via variational/Taylor analysis, then apply Pattern 1–4. Reusable only in restricted smooth/convex models; no generic Fishr/deep-optimizer theorem is established.

**Conclusion:** Patterns 1–4 are reusable outer theorems. Pattern 5 is a method-specific translation layer.
