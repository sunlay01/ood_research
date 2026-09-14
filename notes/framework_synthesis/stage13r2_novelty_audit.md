# Stage 13R.2 Novelty Audit

## Primary sources directly inspected

1. Zhang, Zhao, Yu & Poupart, *Quantifying and Improving Transferability in
   Domain Generalization*, NeurIPS 2021, arXiv:2106.03632. The paper defines
   transferability and one-sided/symmetric/realizable transfer measures,
   proves equivalence of the transferability definition and transfer measures,
   gives target-error bounds, and discusses estimation and discrepancy
   relationships.
2. Hemati, Zhang, Estiri & Chen, *Understanding Hessian Alignment for Domain
   Generalization*, ICCV 2023, arXiv:2308.11778. The paper explicitly uses
   transfer measure, proves a classifier-head Hessian spectral-norm upper bound,
   and interprets CORAL, IRM, V-REx, Fish, IGA and Fishr as partial Hessian/
   gradient regularizers.
3. Chen, Si, Zhang & Zhao, *Moment Alignment: Unifying Gradient and Hessian
   Matching for Domain Generalization*, UAI 2025, arXiv:2506.07378. Definition 3
   extends the transfer measure to multiple sources; Theorems 1 and 3 bound it
   with parameter-side derivative moments under explicit target/convexity/IRM
   assumptions.
4. Mansour, Mohri & Rostamizadeh 2009 and Ben-David et al. 2010 provide the
   classical loss-class discrepancy and source-risk/discrepancy/joint-error
   bounds. These are component-level or special-case neighbors, not exact
   source-quotient transfer certificates.

## Classification of the present theorem package

| Claim | Classification | Explanation |
|---|---|---|
| transfer measure endpoint | `EXACT-ALREADY-DONE` | Zhang 2021; reused by Hemati 2023 and Chen 2025 |
| parameter derivative control of transfer | `EXACT-ALREADY-DONE` | Chen 2025 / Hemati 2023 cover this family |
| affine support reduction for a declared state | `SPECIAL-CASE` | Stage 12 support algebra; not claimed novel alone |
| source quotient `V*/S°` for transfer responses | `COMPONENT-LEVEL-OVERLAP` | quotient/partial-identification structure is classical, but not found stated at this transfer-measure interface |
| sharp minimum of source-only transfer certificates over compatible fibers | `NO-EXACT-OVERLAP-FOUND` | specific interface claim; absence is negative evidence only |
| blind-direction impossibility for source-only transfer certificates | `NO-EXACT-OVERLAP-FOUND` | target-information limitations are known, but this exact support/annihilator form was not found in checked primary sources |
| V-REx/GroupDRO/MMD/IRM corollaries | `SPECIAL-CASE` / `COMPONENT-LEVEL` | existing method-specific theory and standard duality are strong neighbors |

## Conservative conclusion

The contribution cannot be advertised as a new transfer measure, new discrepancy,
or new moment-alignment theory. The defensible claim is narrower:

> We develop a source-observability and sharp-certificate theory for an
> established transfer-measure endpoint, exposing the quotient ambiguity and
> proving when source-only regularizer statistics can or cannot control it.

This is a `DISTINCT-BUT-RISKY` theory direction. It requires a strict
source-certificate theorem and comparison against the cited transfer-measure
papers before any publication-level novelty claim.
