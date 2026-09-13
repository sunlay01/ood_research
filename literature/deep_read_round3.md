# Deep-read round 3: representation and formalization papers

This note is deliberately narrow. It records what was inspected in publisher HTML/PDF text and separates proof-level reading from metadata-only verification.

## 1. Shui, Wang & Gagné (2022): `INV` plus a representation regularizer

**Why the original problem is hard.** Enforcing marginal, feature-conditional, or label-conditional invariance on source environments can still leave a large unseen-domain prediction gap. The paper gives a concrete conditional-invariance counterexample and introduces a representation-smoothness term.

**Common representation chosen.** Equation (1) writes the family as `min_{phi,h} sum_t E_{S_t} L(h o phi) + lambda_0 INV(phi,S_1,...,S_T)`, where `INV` is intentionally a slot for marginal, feature-conditional, or label-conditional invariance. This is a genuine cross-method unifier: the common object is an invariance criterion on the learned representation, not a claim that the empirical optimizers are identical.

**Proof step enabled by the representation.** Proposition 1 assumes bounded loss, source feature-conditional TV distance at most `kappa`, and a target close to a nearest source in raw-space TV by `epsilon`. It obtains `BER_T <= average_t BER_{S_t} + kappa + alpha_TV(phi) epsilon`, where `alpha_TV(phi)` is the Dobrushin coefficient of the embedding. The representation makes the shift term factor into an unobservable source-target distance (`epsilon`) and a controllable contraction/smoothness factor (`alpha_TV`). The proof uses TV triangle inequalities and a strong data-processing inequality; the PDF proof begins by selecting the nearest source and then contracts TV through the representation kernel.

**Price of tractability.** Bounded loss, conditional invariance, a nearest-source target-family assumption, and a representation kernel with a finite contraction coefficient. Cross-entropy is noted as generally unbounded in the experiments, so the theorem's bounded-loss condition is nontrivial.

**Excluded methods.** The slot can host different invariance criteria, but gradient-variance penalties, optimizer trajectories, and arbitrary causal mechanisms are not represented by `INV` alone. The theorem is about the induced representation and family assumptions, not exact deep training dynamics.

## 2. Wang & Veitch (2022): causal shift unification

**Why the original problem is hard.** Marginal feature invariance and class-conditional invariance can be simultaneously insufficient; the correct invariance notion depends on the causal shift structure.

**Common representation chosen.** The paper's common object is a structural causal model (SCM) and its induced shift family. Data augmentation, distributional invariance, and risk-minimizer invariance are compared as strategies under the same causal language.

**Proof step enabled.** The SCM lets the authors state a population invariance condition and compare which latent mechanisms remain stable under interventions. This moves the theorem target from a generic discrepancy to a causal conditional; the proof can then use d-separation/intervention assumptions rather than treating all shifts as arbitrary IPM distances.

**Price and exclusions.** Validity depends on the SCM/shift graph and intervention assumptions. A method with no causal interpretation is outside the framework. The OpenReview PDF endpoint was not accessible in this audit environment, so this entry is metadata/abstract-level rather than a claim of line-by-line proof verification; the paper remains in the queue for a direct full-text pass.

## 3. Lai & Wang (2024): IRMv1 objective-to-functional translation

**Why the original problem is hard.** IRMv1 contains a gradient norm in the training objective, while standard DG bounds are usually written in terms of risks, discrepancies, or hypothesis classes.

**Common representation chosen.** This is not broad cross-method unification. The authors reinterpret the gradient penalty as a total-variation functional of the risk with respect to the classifier variable: the familiar TV-`ell_2` form corresponds to the IRMv1 gradient norm, and a TV-`ell_1` model is proposed as a variant.

**Proof step enabled.** The PDF states Theorems 3.1--3.11. The argument first treats the risk as a function of classifier variable `w`, identifies the gradient norm with TV variation, and then uses variational/measure-theoretic conditions to characterize global OOD objectives, minimax-TV variants, and admissible environment sets. This gives a mathematical home for the derivative without tracking every SGD iterate.

**Price of tractability.** Functional regularity, measure/coarea conditions, and an explicit environment-family condition. The result is not a generic finite-sample target-risk theorem for deep IRMv1; it is a translation and a set of OOD conditions.

**Excluded methods.** GroupDRO, Fishr, and generic Wasserstein DRO are not unified by this TV functional unless an additional mapping is supplied. This is why the paper belongs under “objective-to-theory translation,” not “cross-method unification.”

## 4. Wang, Bai, Yang, Xu & Liang (2026): Tri-Space latent representation

**Why the original problem is hard.** Existing latent-space bounds conflate domain invariance and domain diversity, so they cannot explain why representation alignment and domain augmentation can be complementary.

**Common representation chosen.** Theorem 1 proves a direct-sum decomposition `Z = Z_Gamma ⊕ Z_Phi ⊕ Z_Xi`; Corollary 1 gives the unique decomposition `z = gamma + phi + xi`, respectively domain-invariant, spurious-invariant, and domain-variant features.

**Proof step enabled.** Under Assumptions 1--2 (linear upper bound on loss and deterministic labels from invariant `gamma`), Definition 4 introduces a discrepancy localized to the domain-variant subspace, weighting distribution shift by the hypothesis sensitivity `|h(xi)|`. Theorem 2 then bounds target risk by a weighted source-risk term, a target-to-admissible-family localized discrepancy, a source-domain discrepancy, a spurious-invariant term, and a complexity/approximation term. Theorem 3 states that domain-invariant representation learning drives the corresponding invariant/variant interaction term to zero. The decomposition makes “invariance” and “diversity” appear as separate terms in one bound.

**Price of tractability.** The label must be deterministically recoverable from `gamma`; domain-specific predictive features are explicitly excluded. The latent representation, hypothesis class, localized discrepancy, and admissible target family are restricted. Optimizers are abstracted into two algorithm categories (domain-invariant representation learning and domain augmentation), so the result is not an exact theorem for a particular deep training loop.

**What it captures/excludes.** It captures two broad categories at the level of their induced latent representations and source-domain diversity. It does not directly capture gradient-statistic methods, arbitrary DRO uncertainty sets, or causal mechanisms not encoded by the latent assumptions.

## 5. Rivasplata et al. (2018): PAC-Bayes + stability, not DA

This paper is included as a hygiene/control case because it was previously misclassified. The NeurIPS PDF's Theorem 2 assumes a Hilbert-space-valued algorithm that outputs a Gaussian-randomized classifier and bounds randomized population risk using a stability coefficient plus a PAC-Bayes binary-divergence term. The proof combines a standard PAC-Bayes change-of-measure theorem (Theorem 4) with McDiarmid concentration for the stable algorithm output. There is no source/target disagreement term and no domain-adaptation setting. Therefore it belongs in the PAC-Bayes/stability intersection and must not be used as a second PAC-Bayes DA representative.

## Cross-paper lesson

The papers implement two different moves:

* **Cross-method unification:** choose invariance (`INV`), an SCM, or a direct-sum latent space; accept structural assumptions and exclude optimizer-level fidelity.
* **Objective-to-theory translation:** map a gradient penalty to TV variation; retain a trace of the original objective while changing the proof language.

The practical implication is methodological, not a new theorem: when a DG objective is too heterogeneous, inspect whether a population statistic, latent decomposition, causal family, or variational functional makes the desired bound possible, and record exactly which algorithms and shifts disappear in the translation.
