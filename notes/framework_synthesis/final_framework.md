# Final framework: typed bridge calculus for source-only OOD/DG

## Decision

The bottom-up round supports a **modular typed bridge calculus**, not one common
invariance statistic. The framework is proposed as a research architecture and
notation system; its generic backbone is conditional on the certificate theorem
chosen for a target family.

## Essential operations

1. Compute an exact source-visible method object `Omega_j` from source laws or
   samples.
2. Translate `Omega_j` to a typed population certificate `B_j`, keeping a named
   translation/optimization error.
3. Declare the admissible target family `U_T` or meta-law `Pi` and the target
   query `q_T`.
4. Apply the theorem native to the certificate type: support, discrepancy plus
   conditional residual, mechanism contraction, or information/change-of-measure.
5. Add finite-sample and computation terms separately; return a failure code if
   identification or target inclusion fails.

## Genuine shared primitives

- ERM, V-REx and GroupDRO share the source risk-vector object, but their support
  functionals and target queries differ.
- MMD, CORAL and DANN share distribution-witness algebra only after fixing the
  witness class; all require conditional/joint semantics for label shift.
- ICP, anchor and ideal IRM share a mechanism/intervention object only within a
  stated causal/linear model.
- Norm, stability, PAC-Bayes and information methods share a complexity layer,
  not a shift mechanism.

## Irreducible differences

IRMv1 and Fishr are derivative/optimizer objects. Their syntax does not make
them conditional mechanisms. Fishr has no accepted general local bridge in the
current evidence. Direct-sum latent decompositions and TV functional translations
are restricted representations, not universal replacements.

## Mathematical state

For a method `j`, the retained state is a tagged certificate, not a tuple of all
statistics:

`B_j ::= RiskSupport(r,U) | WitnessConditional(D,Delta_cond) | Mechanism(K,I) | ObjectiveFunctional(F,G) | Complexity(C)`.

The source observation map `O_S` and the exact training object `Omega_j` are
external inputs to the translation `Omega_j -> B_j`; they are not silently
concatenated into the certificate. Only fields required by the selected theorem
are instantiated. The world state supplies the full `P_e`, hidden `P_T`,
structural assumptions and target family; `O_S` is the only observed component.

## Backbone implications

- support: `R_T <= sigma_{U_T}(r_S)` when target inclusion holds;
- discrepancy: `R_T-R_S <= D_G(P_T^Phi,P_S^Phi)+Delta_cond`;
- mechanism: stable conditional kernel plus coverage contracts or removes
  `Delta_cond`;
- complexity: controls empirical-to-population deviation but does not remove
  shift or conditional terms.

Each line is a theorem only under its own assumptions. The hard research is the
method-specific `Omega_j -> B_j` lemma and target-family calibration.

## Inherited mathematics vs synthesis

Inherited: finite-group support and concentration, IPM/RKHS duality, causal
conditional invariance, DRO duality, TV contraction, PAC-Bayes/change of measure,
and the cited impossibility constructions.

Our synthesis: the typed interface, explicit fidelity labels, shared state schema,
and failure semantics that route each method to its native theorem without
forcing semantic equivalence.

## Central information-sufficiency issue

Source-only identification requires the target query to be constant on source-
indistinguishable admissible worlds. Finite source exposure does not ensure this.
Marginal alignment and risk-vector control fail the two-world label-reversal test;
derivative statistics fail unless a new injective bridge is proved.

## Status

`MULTIPLE-IRREDUCIBLE-PRIMITIVES-SURVIVE; MODULAR-FRAMEWORK-ACCEPTED-AS-RESEARCH-ARCHITECTURE`

This status is not a theorem or publication novelty claim. It records the
strongest abstraction justified by the current evidence.
