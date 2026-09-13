# Framework design report

## Research decision

The project should not introduce a new cross-method representation at this stage. The existing literature already supplies the needed mathematical languages. The next theorem should use a two-level target specification:

1. **Primary source-only target:** expected future-domain risk
   `Q_Pi(f) = E_{P_T~Pi} R_{P_T}(f)`, where observed source domains are iid draws from the same domain meta-distribution `Pi`.
2. **Separate robust target:** source-defined worst-case risk
   `Q_U(f) = sup_{P in U(S)} R_P(f)`, where `U(S)` is explicitly declared as a Wasserstein, f-divergence, finite-group, or other uncertainty set.

These are different scientific claims. The first asks for average performance on an exchangeable future domain; the second asks for protection over a declared set. They must not be merged into one generic “OOD risk” theorem.

## 1. Exact target and information structure

For the primary target, training observes `m` source environments `P_1,...,P_m ~ Pi` and finite labeled samples from each environment. The target environment is unavailable during training and is an independent draw from `Pi`. The loss is bounded, the predictor belongs to `F`, and the relevant domain-level class is

`G_F = { P -> R_P(f) : f in F }`.

The training data can estimate within-domain risks and the mean of `G_F` across source environments. It cannot identify the risk of a particular unseen domain beyond the tail behavior of `R_P(f)` under `Pi`.

For the robust target, source data estimate an empirical objective over `U(S)`. The uncertainty set, metric, radius and support assumptions are assumption-controlled. A guarantee outside `U(S)` is not implied.

## 2. Why these targets are preferable

`Q_Pi` is a nontrivial source-only DG target with a clear sampling interpretation and a direct domain-level concentration proof. It avoids pretending that arbitrary target shifts are inferable. `Q_U` is the right target when the scientific claim is worst-case robustness, because the set of possible shifts is visible in the theorem. A single-target risk or transfer gap can be retained as a downstream quantity, but it needs a quantile, discrepancy, or oracle term and should not be the sole source-only target.

## 3. Admissible target families

The defensible family for `Q_Pi` is an exchangeable meta-distribution `Pi` over environments. The key assumption is that source and future domains are draws from the same law, with enough domain diversity for concentration. This is scientifically meaningful for repeated deployment across a domain population, but it excludes adversarial or out-of-support targets.

For `Q_U`, use a source-defined uncertainty set only when its geometry has scientific meaning:

- finite observed groups for GroupDRO;
- an f-divergence neighborhood for reweighting shifts;
- a Wasserstein ball when transport cost and radius are justified;
- a causal/interventional family when the structural graph and intervention class are justified.

The radius or family cannot be chosen only because it makes the bound small. Misspecification must be a named remainder.

## 4. Dominant proof obstructions

The main obstruction is target-family non-identifiability, supported by the lower-bound and alignment results in `Rosenfeld2021`, `Zhao2019`, and `Wang2024Lost`. Classic DA discrepancy bounds also expose a target-dependent discrepancy and joint-error oracle (`BenDavid2010`, `Mansour2009`). Risk vectors and marginal feature alignment lose conditional label information. Deep gradient or optimizer statistics have no general source-only target theorem in the current corpus. Latent decompositions and TV translations work only after their structural assumptions are declared.

## 5. Required properties

The primary route needs:

- domain-level concentration over `Pi`;
- within-domain concentration over samples;
- a domain-risk class `G_F` that makes the complexity term explicit;
- a separate tail or quantile term for one specific target domain;
- an assumption-controlled coverage statement connecting source domains to `Pi`.

The robust route needs convex/transport duality, a declared uncertainty set, and a misspecification term. Conditional-shift claims additionally need conditional invariance, a causal family, or a localized discrepancy. None of these properties requires a new base representation.

## 6. Existing representation stress test

The property matrix shows why no single family wins on every criterion. Domain-of-domains is the cheapest language for `Q_Pi`; DRO is the cleanest for `Q_U`; causal/conditional and latent decompositions provide stronger semantics at higher assumption cost; discrepancy/IPM is modular but target-dependent; norm/PAC-Bayes methods supply complexity terms rather than shift identification. IRM-TV remains a translation of one objective, not a unifier.

The project should therefore use a layered theorem design: domain-level target family first, then the least expensive conditional or latent refinement required by the scientific shift claim. Do not add gradient covariance, optimizer trajectories, or A/O/Pi objects unless a later theorem proves their necessity.

## 7. Representation cost ledger

| Route | Preserved | Discarded | Tractability gained | Main cost |
|---|---|---|---|---|
| `Pi` + `G_F` | domain population and function-level risk | arbitrary out-of-support targets; optimizer details | two-level concentration and source-only expected-risk bound | exchangeability/coverage |
| `U(S)` DRO | declared worst-case shift geometry | targets outside `U`; mechanism semantics | convex/transport duality and robust uniform convergence | metric/radius/support misspecification |
| causal conditional | stable conditional mechanism | shifts outside SCM/intervention class | identification and intervention robustness | strong structural assumptions |
| direct-sum latent | invariant/variant contribution split | arbitrary entanglement | localized discrepancy and interaction terms | deterministic labels and restricted latent class |
| TV translation | classifier-gradient objective at a functional level | exact deep optimizer dynamics | variational analysis of IRM-like penalty | regularity/coarea and no general finite-sample theorem |

## 8. Minimal validating theorem

The first proof should establish a two-level uniform convergence result for `Q_Pi(f)`. A concrete attempt is recorded in [`proofs/baselines/source_only_meta_domain_bound.md`](../../proofs/baselines/source_only_meta_domain_bound.md). It decomposes error into between-domain complexity, within-domain complexity, and concentration. The theorem is useful only if its domain-level class complexity is finite and the empirical source domains are representative of `Pi`.

The next extension should add a fresh-domain quantile or robust remainder. If the project cannot estimate or scientifically bound that remainder, it should report expected future-domain risk rather than claim a pointwise target-domain guarantee.

## 9. Falsification tests

The proposed route should be rejected or narrowed if any of these tests fail:

- Two source-domain populations with identical empirical domain-risk summaries but different `Pi` tails produce materially different target-domain risk.
- A proposed conditional-shift theorem has identical representation statistics in two worlds but different `P(Y|X)` and no irreducible term.
- The uncertainty-set theorem is claimed for a target outside its metric/radius support.
- The domain-level complexity term is infinite or cannot be estimated under the intended hypothesis class.
- A claimed IRM/Fishr/optimizer theorem requires an unproved bridge from gradient statistics to target risk.

## 10. Final research route

Advance with the existing domain-of-domains representation for an expected future-domain theorem, and keep DRO as a separate robust theorem family. Use conditional invariance or the Tri-Space decomposition only as explicitly justified refinements. The contribution should be a tight, source-only, two-level theorem with transparent family and tail terms, or a lower bound showing why one of those terms cannot be removed. Novel notation or a synthetic union of existing representations is not needed.

The genuinely open contribution is therefore theorem-level: sharpen the two-level complexity/coverage decomposition, characterize when its fresh-domain tail term is estimable, and compare it against a source-defined DRO guarantee under matched target-family assumptions. It is not a new representation claim.

Status: **ADVANCE existing representation; no new framework.**
