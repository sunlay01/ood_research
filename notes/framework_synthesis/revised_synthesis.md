# Revised synthesis after attack

## Well-supported structure

1. **Risk/support bridge (P1):** exact for ERM, finite-group GroupDRO and the
   risk-vector part of V-REx. Its query is observed-group, declared-mixture or
   meta-domain risk, not arbitrary point-target risk.
2. **Witness/conditional bridge (P2):** exact for the selected IPM or moment
   witness and reusable with a conditional/joint residual. Marginal equality is
   not conditional equality.
3. **Mechanism bridge (P3):** exact for the stated causal/conditional model and
   useful for ICP, anchor and ideal IRM under coverage/identifiability assumptions.
4. **Complexity layer (P5):** exact for source estimation or posterior/stability
   control, but not a shift model.

## Restricted-model structure

- IRMv1 -> TV functional is a valid variational translation under Lai2024's
  regularity assumptions, not a general deep-training equivalence.
- Direct-sum invariant/spurious/variant decompositions are useful under the
  Wang2026TriSpace latent model, but do not encode generic DRO, Fishr or arbitrary
  SCMs.
- Anchor regression and Wasserstein/f-DRO are exact only for their declared shift
  geometry.

## Plausible but unproved

- A Fishr covariance certificate may be useful in a restricted model, but the
  covariance-to-target bridge is not in the current evidence.
- A V-REx variance penalty may improve a tail functional under a specified meta-law,
  but no general variance-to-tail theorem is assumed.
- A cross-method scalar “invariance” variable is not supported.

## Known non-unifiable differences

- Conditional mechanisms, marginal witnesses and optimizer derivative laws retain
  different information and have different failure semantics.
- GroupDRO's uncertainty support and causal intervention robustness answer distinct
  target queries, even if both are called robust.
- Fishr/IRMv1 operate on an objective/optimizer level; MMD/CORAL operate on
  pushforward distributions; V-REx/GroupDRO operate on risks.

## Surviving design principle

Build a modular bridge calculus with typed intermediate certificates and explicit
target-family semantics. The central research object is the local lemma

`Omega_j -> B_j`,

not a new universal statistic. A generic theorem can then map `B_j` to a target
query only when the certificate's semantics match the theorem.
