# 证据登记册

> 规则：仅 `VERIFIED` 项可支持正式引用或新颖性判断。先登记链接，再写解释；不要从聊天讨论补造书目信息。

| ID | 主张/待核验问题 | 原始来源链接或 DOI | 来源类型 | 状态 | 核验范围 | 记录人/日期 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SEED-01 | 表示层 target-risk / hybrid decomposition 的近邻工作 | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Wu et al. (2020)，需确认精确题名、作者和结论。 |
| SEED-02 | nested oracle 形式的 DG failure decomposition | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Galstyan et al. (2022)，需核验。 |
| SEED-03 | gradient/Hessian 对齐与 feature moments 的关系 | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Moment Alignment (2025)，需核验。 |
| P-001 | representation-level exact risk decomposition | https://arxiv.org/abs/2004.10390 | arXiv preprint | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/representation_bayesian_risk_decompositions_2020_wu.pdf`。 |
| P-002 | nested-oracle / counterfactual error decomposition for DG failure modes | https://openaccess.thecvf.com/content/CVPR2022/html/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.html | CVPR 2022 | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/failure_modes_domain_generalization_algorithms_2022_galstyan.pdf`。 |
| P-003 | invariance-based DG 中 representation smoothness / regularization 的理论动机 | https://link.springer.com/article/10.1007/s10994-021-06080-w | Machine Learning, open access | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/representation_regularization_invariance_dg_2022_shui.pdf`。 |
| P-004 | DG penalty 的 excess empirical risk 问题 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/57568e093cbe0a222de0334b36e83cf5-Abstract-Conference.html | NeurIPS 2022 | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/domain_generalization_without_excess_empirical_risk_2022_sener.pdf`。 |
| P-005 | Hessian/gradient alignment as transfer-measure control in DG | https://openaccess.thecvf.com/content/ICCV2023/html/Hemati_Understanding_Hessian_Alignment_for_Domain_Generalization_ICCV_2023_paper.html | ICCV 2023 | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/understanding_hessian_alignment_dg_2023_hemati.pdf`。 |
| P-006 | IRM 的 TV-`l2` / TV-`l1` 数学重写 | https://proceedings.mlr.press/v235/lai24c.html | ICML 2024 / PMLR | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/invariant_risk_minimization_total_variation_model_2024_lai.pdf`。 |
| P-007 | gradient/Hessian matching 与 feature moments 的统一 | https://proceedings.mlr.press/v286/chen25f.html | UAI 2025 / PMLR | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/moment_alignment_gradient_hessian_matching_dg_2025_chen.pdf`。 |
| P-008 | Tri-space latent representation 与细粒度 DG risk bound | https://www.jmlr.org/beta/papers/v27/25-0399.html | JMLR 2026 | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/bridging_domain_invariance_diversity_2026_wang.pdf`。 |
| P-009 | invariant-learning objectives 的 Landau-style regularization path 分析 | https://arxiv.org/abs/2608.09396 | arXiv preprint | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/landau_theory_invariant_learning_2026_wang.pdf`；近期预印本，后续须严格审查。 |
| P-010 | regularization path / hyperparameter implicit differentiation 背景 | https://proceedings.mlr.press/v119/bertrand20a.html | ICML 2020 / PMLR | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/implicit_differentiation_lasso_hyperparameter_2020_bertrand.pdf`。 |

## 新增证据的最小字段

- 原始来源（DOI、PMLR/CVF/期刊官网或 arXiv）；不要只留搜索页。
- 论文存在性、版本、同行评审状态。
- 与本课题直接相关的一段命题/定义及其定位（章节或页码）。
- 证据支持什么、不支持什么，以及可能的反例或限制。
