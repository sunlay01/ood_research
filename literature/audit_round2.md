# Literature audit round 2: network-verified corrections and additions

This file records live checks performed on 2026-09-13, after review of commit `2ac90c7`. The first round did not use an online search API; this round does.

## Metadata checks

| Record | Live endpoint | Result |
|---|---|---|
| Ben-David et al. | Crossref `https://api.crossref.org/works/10.1007/s10994-009-5152-4` | Title, six authors, Machine Learning 79 (2010), DOI verified. The old `...0130-7` DOI was wrong. |
| Blanchard et al. 2011 | NeurIPS proceedings | Title is *Generalizing from Several Related Classification Tasks to a New Unlabeled Sample*; authors Blanchard, Lee, Scott. The old entry had conflated this with the 2021 JMLR extension. |
| Blanchard et al. 2021 | JMLR v22 | Separate *Domain Generalization by Marginal Transfer Learning* entry added. |
| Germain et al. | ScienceDirect PII `S0925231219315486`; Crossref title query | Final article is Neurocomputing 379 (2020), authors Germain, Habrard, Laviolette, Morvant, DOI `10.1016/j.neucom.2019.10.105`. |
| Rivasplata et al. 2019 | NeurIPS proceedings | Record is *PAC-Bayes Bounds for Stable Algorithms with Instance-Dependent Priors*, not PAC-Bayes domain adaptation. Label corrected. |
| Duchi & Namkoong | Crossref `10.1214/20-AOS2004` | AoS 49(3), 2021, DOI verified; old `10.1214/17-AOS1561` was unrelated. |
| Lai & Wang 2024 | PMLR v235 page `lai24c` | Title, authors, pages 25913--25935, abstract, and claim “IRM is TV-l2 in classifier variable” verified from the publisher page. |
| Wang, Wu & Zhang 2024 | Crossref `10.1609/aaai.v38i14.29497` | AAAI 38(14) title/authors/year/DOI verified. |
| Cao & Chen 2024 | Crossref `10.1609/aaai.v38i10.28994` | AAAI title/authors/year/DOI verified. |
| Shui, Wang & Gagné | Crossref `10.1007/s10994-021-06080-w` | Machine Learning 111 (2022), title/authors/year/DOI verified. |

## Scientific corrections

* MMD and DANN are now separate taxonomy rows: RKHS mean embedding versus discriminator-induced discrepancy.
* V-REx and Fishr are now separate rows: environment-risk variance versus per-example gradient variance.
* Spectral normalization remains a capacity-control row, not evidence for a DG-specific spectrum theory; a future pass must search representation/operator spectrum directly.
* No claim of a complete 2024--2026 census is made. The provenance file now lists this as unresolved and the queue contains verified 2024 primary papers for the next deep read.

## Publisher-text checks used for theorem anatomy

The REx PMLR page (`v139/krueger21a`) was fetched and its author/date/page metadata verified; the PDF text was searched for Theorems 1--2 and the V-REx/MM-REx sections. The Fishr PMLR page (`v162/rame22a`) was fetched and its author/date/page metadata verified. The Shui et al. Springer PDF was fetched and searched for Proposition 1/Theorem 1 and the TV/Jacobian terms. The Lai--Wang PMLR PDF was fetched and searched for Theorems 3.1--3.11, including the TV-l2 interpretation and OOD conditions. These are the source checks behind the added proof-anatomy paragraphs; they are not claims that every proof line has been independently rederived.
