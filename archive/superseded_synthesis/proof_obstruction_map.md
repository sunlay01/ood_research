# Proof-obstruction map

Status labels follow the project protocol: `PROVED`, `COUNTEREXAMPLE-SUPPORTED`, `LITERATURE-SUPPORTED`, `INTERPRETATION`, and `OPEN`.

| Obstruction | Status | Evidence | Consequence for representation design |
|---|---|---|---|
| Arbitrary unseen target behavior is not identifiable from finitely many source domains. | COUNTEREXAMPLE-SUPPORTED / LITERATURE-SUPPORTED | `Rosenfeld2021` main lower bound; `Wang2024Lost` lower bound; `Gulrajani2021` discussion | State a target family or prove impossibility. No representation can recover missing information for arbitrary targets. |
| Classic `HDeltaH`/loss discrepancy bounds use target data and a joint-error oracle. | PROVED | `BenDavid2010` Thm. 2; `Mansour2009` Thm. 1-2 | Do not call DA discrepancy source-only. Replace target discrepancy with `Pi`, `U(S)`, or a causal family. |
| Source risk vectors can agree while conditional label mechanisms differ. | LITERATURE-SUPPORTED | `Zhao2019` Thm. 1 and alignment/label conflict construction | Risk variance or group risk alone cannot support conditional-shift claims. Add a conditional assumption, localized discrepancy, or irreducible term. |
| Marginal alignment does not identify predictive conditionals. | COUNTEREXAMPLE-SUPPORTED | `Zhao2019` Thm. 1; `Shui2022` conditional-invariance discussion | MMD/CORAL/DANN are insufficient as a universal DG representation. |
| A chosen uncertainty set makes robust risk tractable but can be misspecified. | PROVED | `Duchi2021` Thm. 1-2; `Esfahani2018` Thm. 4.2 | Treat radius, metric and support as first-class assumptions; report outside-set behavior as misspecification. |
| A fresh-domain risk has a domain-level fluctuation beyond expected future-domain risk. | INTERPRETATION | Bounded random variable `R_P(f)` under `P~Pi`; see minimal theorem attempt in `proofs/baselines/source_only_meta_domain_bound.md` | Use expected future-domain risk as the primary source-only target; add a quantile/robust remainder for one specific target domain. |
| Latent decomposition can localize invariant and variant contributions only under structural assumptions. | LITERATURE-SUPPORTED | `Wang2026TriSpace` Thm. 1-3; `Shui2022` Prop. 1 | Use decomposition as a conditional refinement, not a universal base representation. |
| IRMv1 gradient statistics do not automatically become a target-risk theorem. | LITERATURE-SUPPORTED | `Arjovsky2019` Eq. 2-3; `Kamath2021` Thm. 1-3; `Lai2024` Thm. 3.1-3.11 | Keep gradients at the translation layer unless a separate derivative-class and shift theorem is proved. |
| Algorithmic fidelity and source-only tractability conflict for deep optimizer statistics. | LITERATURE-SUPPORTED / OPEN | `Rame2022` is motivational-only in the ledger; `Liu2025InfoSGLD` is stochastic-process abstraction | Abstract optimization with an explicit error or pursue a separate restricted algorithm theorem. |
| Oracle and assumption-controlled terms can be hidden by a coarse discrepancy symbol. | INTERPRETATION | Cross-family comparison in `theorem_map.csv` | Every theorem must label source-estimable, assumption-controlled, target-dependent, oracle and irreducible terms separately. |

## Primary obstruction for this project

The missing object is not another feature representation. It is a defensible, source-only description of which future domains are being quantified over. Once that family is fixed, existing domain-level concentration or DRO representations already provide the proof operations needed for a meaningful theorem.
