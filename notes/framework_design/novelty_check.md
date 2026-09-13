# Targeted novelty verification

Date: 2026-09-13. This is a focused check of the unresolved arrows, not a
complete 2024--2026 census. Crossref metadata queries and direct publisher or
repository records were used where available; rate-limited search services and
CAPTCHA-protected pages were not treated as evidence. Absence from a query is
therefore never classified as proof of novelty.

The check targets the exact missing arrows rather than repeating a broad survey.

| Missing statement | Search/evidence | Classification |
|---|---|---|
| `MMD/CORAL -> conditional label invariance` | Zhao2019 counterexample; existing IPM theory controls only the witness class | KNOWN FALSE WITHOUT EXTRA ASSUMPTIONS |
| `V-REx variance -> arbitrary unseen-domain target risk` | Krueger risk-dispersion formulation; domain-of-domains evidence retains meta-law/tail term | KNOWN-IN-RESTRICTED-SETTING; otherwise false |
| `IRMv1 empirical gradient -> ideal IRM/causal mechanism` | Arjovsky/Kamath restricted theory; Lai functional translation; no unrestricted deep bridge in ledger | KNOWN-IN-RESTRICTED-SETTING |
| `Fishr covariance -> invariant conditional target risk` | Rame2022 remains motivational/surrogate in the repository; focused 2024--2026 metadata searches returned no directly matching theorem record | UNCERTAIN; not a novelty claim |
| `GroupDRO/f-DRO/Wasserstein objective -> worst-case risk in declared U` | Duchi2021, Esfahani2018 duality and source-group proof baseline | KNOWN |
| `source-only finite samples -> arbitrary target risk` | Rosenfeld2021, Wang2024Lost and channel-swap construction | KNOWN IMPOSSIBLE |
| `conditional witness penalty -> conditional target bound` | standard conditional change-of-measure pattern; requires overlap and bounded witness class | KNOWN-IN-RESTRICTED-SETTING |

## 2024--2026 records that affect the decision

| Work | Year | Relevance to the missing arrow | Classification |
|---|---:|---|---|
| Liu, Yu, Wang & Liao, *An Information-Theoretic Framework for OOD Generalization* (ISIT), DOI `10.1109/ISIT57864.2024.10619471` | 2024 | Information-radius and shift assumptions give a different source-to-OOD bound; it does not translate Fishr or V-REx statistics into conditional invariance | CLOSE-BUT-DIFFERENT |
| Liu, Yu, Wang & Liao, *An Information-Theoretic Framework for OOD Generalization With Applications to SGLD* (IEEE TIT), DOI `10.1109/TIT.2025.3598722` | 2025 | Extends information-theoretic/OOD analysis to stochastic optimization; target-risk control remains conditional on the information and shift model | CLOSE-BUT-DIFFERENT |
| Wu, Liu, Cui & Wu, *Bridging Multicalibration and OOD Generalization Beyond Covariate Shift* (NeurIPS 2024), DOI `10.52202/079017-2325` | 2024 | Uses calibration and density-ratio coverage to handle beyond-covariate shift; it supplies a conditional coverage route, not a marginal-MMD or Fishr bridge | CLOSE-BUT-DIFFERENT |
| Wang, Wu & Zhang, *Lost Domain Generalization Is a Natural Consequence of Lack of Training Domains* (AAAI 2024), DOI `10.1609/aaai.v38i14.29497` | 2024 | Gives an impossibility/lower-bound direction for insufficient source-domain coverage | KNOWN IMPOSSIBILITY / REUSED |
| Wang et al., *Tri-Space Representation for Domain Generalization* (repository record `Wang2026TriSpace`) | 2026 record in current ledger | Adds a restricted direct-sum latent decomposition with deterministic-label and latent-class assumptions; it does not establish a universal cross-method representation | KNOWN-IN-RESTRICTED-SETTING |

These records reinforce the route comparison: recent work solves narrower
information, calibration, latent-decomposition, or impossibility problems. It
does not remove the conditional remainder, target-family coverage term, or
algorithm-specific translation obligation in the present problem.

Search metadata and selected adjacent works are recorded in `notes/evidence/framework_construction/extraction/external_search_log.md`. No absence from the repository is treated as a novelty claim.
