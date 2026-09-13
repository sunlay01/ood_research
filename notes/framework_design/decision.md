# Problem-first decision

## A. Is an existing framework sufficient?

**YES-WITH-LOCAL-THEOREM.** Existing discrepancy/IPM, conditional/causal, robust/DRO and meta-domain frameworks can state the relevant target quantities. The unresolved work is a local algorithm-specific bridge and, where necessary, a sharper decomposition or impossibility result. Current evidence does not justify a new universal representation.

## B. Representation to use

Use the representation native to each problem instance: loss-class/IPM for MMD, conditional/SCM objects for mechanism claims, uncertainty sets for DRO, and domain-risk functions for expected future-domain targets. Keep these semantics distinct. Add a conditional remainder or target-family term whenever the native object cannot identify labels or unseen domains.

## C. Remaining mathematical problem

The primary open relation is:

`empirical regularizer -> population bridge object -> target-risk theorem`

for one concrete regularizer under a restricted, explicit model. The highest-value candidates are a conditional witness penalty or a restricted IRMv1 derivative translation. Fishr remains an unresolved stress test rather than a premise.

## D. What to prove next

Primary theorem: bounded-loss marginal-plus-conditional witness bound with finite source complexity and an explicit penalty-to-radius lemma for one chosen regularizer. Fallback: a matching indistinguishability lower bound showing the conditional term cannot be removed, or a finite-mixture DRO comparison theorem under matched target-family assumptions.

## E. Falsification criterion

Reject a route if two admissible worlds have identical source information and identical retained bridge objects/regularizer values but target risks differ by a constant while satisfying the claimed target-family assumptions, or if every valid radius/conditional term is maximal and the bound is vacuous.

## F. Where novelty may lie

Only after the theorem survives can the contribution be classified. Plausible categories are `THEOREM`, `DECOMPOSITION`, or `IMPOSSIBILITY`; `REPRESENTATION`, `FRAMEWORK-EXTENSION`, `NEW-FRAMEWORK` and `COMBINATION` are not currently forced by the evidence.

Final status: **YES-WITH-LOCAL-THEOREM**.
