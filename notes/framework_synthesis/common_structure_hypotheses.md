# Common-structure hypotheses

All entries are `OUR-INTERPRETATION`; they are hypotheses to attack, not established
claims that the methods share one essence.

## H1. Certificate-interface hypothesis

Every usable source-only method can be analyzed through

`Omega_j(S) -> B_j(P_1,...,P_m,f) -> Q_T(f)`.

Here `B_j` is a semantically typed bridge object and `Q_T` is either target risk,
robust target risk, expected future-domain risk, or an explicit impossibility query.
The generic theorem is about `B_j`, not about the syntactic regularizer.

**Evidence:** exact support-function, IPM, conditional, contraction and
meta-domain theorems in the ledger. **Open point:** no proof that every optimizer
statistic admits a useful `B_j`.

## H2. Two-part shift-control hypothesis

For distributional methods, target error should decompose into an observable or
source-inferred witness term plus a conditional/joint residual:

`R_T(f) <= source_certificate(f) + witness_radius(P_T,P_S) + conditional_residual + complexity`.

The conditional residual cannot be removed under arbitrary label shift; the
binary channel-swap counterexample supplies a kill test.

**Status:** `PROVED-UNDER-RESTRICTIONS` for standard discrepancy decompositions;
not a universal equality for all losses.

## H3. Target-family-first hypothesis

Source-only control has content only relative to a declared admissible target
family `U_T` or meta-law `Pi`. A target-membership statement is external or
theorem-dependent, never learned from source statistics alone in general.

**Evidence:** DRO support theorems, Shui2022, Blanchard2021, Wang2024Lost and the
source-indistinguishability constructions.

## H4. Local translation hypothesis

IRMv1, Fishr and MLDG should enter through method-specific lemmas:

`Omega_j -> B_j`.

IRMv1 has a restricted derivative-to-TV translation; Fishr's covariance-to-risk
bridge is still an open probe; MLDG has Taylor/optimization-error translations.
This is stronger and more honest than calling all three “invariance”.

## H5. Complexity orthogonality hypothesis

Norm, stability, PAC-Bayes and information terms alter estimation or learner
sensitivity, while P1-P3 determine shift semantics. Combining them is a theorem
composition, not a new common OOD primitive.

## Candidate compression decision

H1-H5 support a modular hierarchy, not a scalar common primitive. Any framework
that identifies `B_j` across methods must preserve its semantic type and expose
translation, target-family, conditional and computation gaps.
