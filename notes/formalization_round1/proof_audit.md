# Independent proof audit

## Theorem audited

`proofs/source_group_robust_bound.md` and `proofs/typed_conditional_shift_bound.md`.

## Findings

1. **No hidden target samples.** Both empirical bounds use only labeled source
   samples. The target appears only through the declared mixture family or the
   radii `(rho_m,rho_c)`.
2. **Target family is explicit.** The finite-mixture theorem covers only
   reweightings of observed environments. The typed theorem covers only Q with
   the stated marginal and conditional budgets. Neither is an arbitrary-target
   DG theorem.
3. **Conditional term is necessary.** The channel-swap counterexample keeps
   source risks and marginal feature operators fixed while changing target risk
   by one. Therefore `rho_c` (or a causal/SCM assumption implying it) cannot be
   removed.
4. **Regularizer fidelity is limited.** The proofs establish concentration and
   duality, not a bridge from V-REx, MMD, IRMv1 or Fishr values to the radii.
   Such a bridge must be a separate lemma with its own assumptions.
5. **Optimization is not silently exact.** Any empirical approximate minimizer
   needs `epsilon_opt`; translated methods also need `epsilon_translation`.
6. **Complexity can be vacuous.** The bounds require finite per-domain
   Rademacher complexity and, for IPM use, bounded witness/loss-section norms.
   Unrestricted deep classes do not automatically satisfy this.
7. **Known-equivalence check.** The robust result is the standard support
   function plus uniform convergence argument; the conditional result is a
   standard change-of-measure decomposition. The project should claim a new
   result only if a later regularizer-to-radius lemma or tight comparison adds
   more than this known scaffold.

## Verdict

**NEEDS-TARGETED-VALIDATION.** The outer architecture and minimal proofs are
sound at their declared abstraction levels. The unresolved scientific result is
whether any derivative or marginal regularizer yields a non-vacuous conditional
radius under a defensible restricted model.
