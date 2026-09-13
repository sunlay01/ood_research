# Wang2026TriSpace / Bridging Domain Invariance and Diversity

## Problem
Explain, in one risk bound, why invariant representation learning and domain augmentation can contribute complementary effects.

## Target
Target-domain risk over an admissible family.

## Representation
Direct sum `Z = Z_Γ ⊕ Z_Φ ⊕ Z_Ξ`: domain-invariant, spurious-invariant, and domain-variant components.

## Proof bottleneck
Existing bounds conflate invariance and diversity, hiding which latent component causes the target gap.

## Proof-enabling property
Unique decomposition isolates a localized discrepancy on the domain-variant subspace and weights it by hypothesis sensitivity.

## Main theorem
Theorem 1/Corollary 1 establish the direct-sum/unique decomposition; Theorem 2 bounds target risk with weighted source risk, target-to-family localized discrepancy, source-domain discrepancy, spurious-invariant and complexity terms; Theorem 3 identifies the invariant/variant interaction driven to zero by the corresponding method categories.

## Price of tractability
Deterministic labels from invariant `γ`, restricted latent/hypothesis classes, explicit admissible family and approximation terms.

## Algorithm relationship
EXACT within the stated latent model; algorithms are abstracted into representation learning and augmentation categories.

## Scope
Naturally covered: those two categories. Excluded: gradient-statistic methods, generic DRO and unencoded SCMs.

## Source-only status
YES conditional on the admissible target family and latent assumptions.

## Evidence
Theorems 1-3; Definition 4; JMLR volume 27 (2026).
