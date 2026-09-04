# Papers

This folder tracks the paper corpus extracted from the shared ChatGPT discussion.

The PDF files are downloaded locally under `papers/pdfs/`, but that directory is
not committed to Git because the GitHub repository is public. The tracked files
below preserve the official source URLs, PDF URLs, local filenames, page counts,
file sizes, and SHA-256 hashes so the corpus can be rebuilt reproducibly.

Run:

```bash
bash scripts/download_papers.sh
```

## Downloaded Local Corpus

| ID | Paper | Venue / status | Local PDF | Pages | Status |
| --- | --- | --- | --- | ---: | --- |
| P-001 | Representation Bayesian Risk Decompositions and Multi-Source Domain Adaptation | arXiv, 2020 | `papers/pdfs/representation_bayesian_risk_decompositions_2020_wu.pdf` | 21 | downloaded |
| P-002 | Failure Modes of Domain Generalization Algorithms | CVPR, 2022 | `papers/pdfs/failure_modes_domain_generalization_algorithms_2022_galstyan.pdf` | 10 | downloaded |
| P-003 | On the benefits of representation regularization in invariance based domain generalization | Machine Learning, 2022 | `papers/pdfs/representation_regularization_invariance_dg_2022_shui.pdf` | 21 | downloaded |
| P-004 | Domain Generalization without Excess Empirical Risk | NeurIPS, 2022 | `papers/pdfs/domain_generalization_without_excess_empirical_risk_2022_sener.pdf` | 12 | downloaded |
| P-005 | Understanding Hessian Alignment for Domain Generalization | ICCV, 2023 | `papers/pdfs/understanding_hessian_alignment_dg_2023_hemati.pdf` | 11 | downloaded |
| P-006 | Invariant Risk Minimization Is A Total Variation Model | ICML / PMLR, 2024 | `papers/pdfs/invariant_risk_minimization_total_variation_model_2024_lai.pdf` | 23 | downloaded |
| P-007 | Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization | UAI / PMLR, 2025 | `papers/pdfs/moment_alignment_gradient_hessian_matching_dg_2025_chen.pdf` | 32 | downloaded |
| P-008 | Bridging Domain Invariance and Diversity: A Fine-Grained Risk Bound for Domain Generalization | JMLR, 2026 | `papers/pdfs/bridging_domain_invariance_diversity_2026_wang.pdf` | 54 | downloaded |
| P-009 | From Objectives to What Models Learn: A Landau Theory of Invariant Learning | arXiv, 2026 | `papers/pdfs/landau_theory_invariant_learning_2026_wang.pdf` | 33 | downloaded |
| P-010 | Implicit differentiation of Lasso-type models for hyperparameter optimization | ICML / PMLR, 2020 | `papers/pdfs/implicit_differentiation_lasso_hyperparameter_2020_bertrand.pdf` | 12 | downloaded |

## Scope Notes

- These are the unique paper-level sources linked in the shared discussion.
- Duplicate links to the same paper, such as arXiv and conference pages for
  Galstyan et al., were collapsed to one local PDF.
- Author pages and search pages were not downloaded as papers.
- Download status confirms file availability and metadata only; it does not mean
  the paper's technical claims have been fully read or adopted by this project.
