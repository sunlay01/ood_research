# 证据登记册

> 规则：仅 `VERIFIED` 项可支持正式引用或新颖性判断。先登记链接，再写解释；不要从聊天讨论补造书目信息。
>
> 活跃问题已改为 regularizer-induced operator、controlled/blind components 与 OOD error accounting。下表中关于 latent decomposition、certificate-first audit 与 cross-decomposition attribution 的条目仍是历史证据，不自动支持新主线。

| ID | 主张/待核验问题 | 原始来源链接或 DOI | 来源类型 | 状态 | 核验范围 | 记录人/日期 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SEED-01 | 表示层 target-risk / hybrid decomposition 的近邻工作 | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Wu et al. (2020)，需确认精确题名、作者和结论。 |
| SEED-02 | nested oracle 形式的 DG failure decomposition | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Galstyan et al. (2022)，需核验。 |
| SEED-03 | gradient/Hessian 对齐与 feature moments 的关系 | `UNVERIFIED` | 聊天线索 | UNVERIFIED | 未检索 | — | 讨论中提到 Moment Alignment (2025)，需核验。 |
| P-001 | representation-level exact risk decomposition | https://arxiv.org/abs/2004.10390 | arXiv preprint | VERIFIED | 官方 PDF 摘要、§1、§3.1、§4.1 与其 risk-decomposition comparison 定向精读 | Codex / 2026-09-04 | 精确分解 source risk、representation conditional-label divergence、covariate shift；再把 covariate shift 拆为 absolute-continuous/singular risk。映射 DANN 到 covariate shift、IRM 到 conditional alignment，并比较多个 risk decompositions；未核验到同一算法路径在两个 construction 下的 crosswalk comparison。 |
| P-002 | nested-oracle / counterfactual error decomposition for DG failure modes | https://openaccess.thecvf.com/content/CVPR2022/html/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.html | CVPR 2022 | VERIFIED | 官方 PDF 摘要、§3–5、Fig. 3 与算法列表定向精读 | Codex / 2026-09-04 | 单一 nested-oracle sequential error decomposition；比较 ERM、HSIC、CORAL、IRM、DANN 等，且 Fig. 3 直接比较 regularization strength 与训练时长下的分量。它是当前 "单一分解 + 算法路径" 路线的最接近重叠。 |
| P-003 | invariance-based DG 中 representation smoothness / regularization 的理论动机 | https://link.springer.com/article/10.1007/s10994-021-06080-w | Machine Learning, open access | VERIFIED | 官方页；本地 PDF 摘要、§3 命题和 §4 定向精读 | Codex / 2026-09-04 | 已给出 target-risk bound、representation smoothness/Dobrushin 项与 collapse 警告。 |
| P-004 | DG penalty 的 excess empirical risk 问题 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/57568e093cbe0a222de0334b36e83cf5-Abstract-Conference.html | NeurIPS 2022 | VERIFIED | 官方页；本地 PDF 摘要、§1–2 定向精读 | Codex / 2026-09-04 | 支持“penalty 下降不等于 target risk 下降”的优化层替代解释。 |
| P-005 | Hessian/gradient alignment as transfer-measure control in DG | https://openaccess.thecvf.com/content/ICCV2023/html/Hemati_Understanding_Hessian_Alignment_for_Domain_Generalization_ICCV_2023_paper.html | ICCV 2023 | VERIFIED | 官方页；本地 PDF 摘要与 §3.1 Theorem 3 定向精读 | Codex / 2026-09-04 | 在强凸、同最优 head 等条件下，Hessian 距离上界 transfer measure；本项目主线的直接近邻。 |
| P-006 | IRM 的 TV-`l2` / TV-`l1` 数学重写 | https://proceedings.mlr.press/v235/lai24c.html | ICML 2024 / PMLR | VERIFIED | 官方 PMLR 页；本地 PDF 摘要与理论部分定向检索 | Codex / 2026-09-04 | IRM 已有 objective-level 数学重解释；不能声称本项目首次解释 IRM penalty。 |
| P-007 | gradient/Hessian matching 与 feature moments 的统一 | https://proceedings.mlr.press/v286/chen25f.html | UAI 2025 / PMLR | VERIFIED | 官方 PMLR 页；本地 PDF 摘要、§2–3 定向精读 | Codex / 2026-09-04 | 已把 IRM、gradient、Hessian 作为 moment alignment 特例，并给出 multi-source target-error/transfer 分析；是主线最强重叠。 |
| P-008 | Tri-space latent representation 与细粒度 DG risk bound | https://www.jmlr.org/beta/papers/v27/25-0399.html | JMLR 2026 | VERIFIED | 官方 JMLR 页；本地 PDF 摘要与 §1 定向精读 | Codex / 2026-09-04 | 已有 latent direct-sum decomposition 和 fine-grained target-risk bound；压缩“新 latent risk decomposition”空间。 |
| P-009 | invariant-learning objectives 的 Landau-style regularization path 分析 | https://arxiv.org/abs/2608.09396 | arXiv preprint | VERIFIED | 官方 arXiv 元数据；本地 PDF 摘要与 §1 定向精读 | Codex / 2026-09-04 | 近期预印本已直指 objective-to-representation regularization path；须作为高风险近邻，不能以其缺乏同行评审忽略。 |
| P-010 | regularization path / hyperparameter implicit differentiation 背景 | https://proceedings.mlr.press/v119/bertrand20a.html | ICML 2020 / PMLR | VERIFIED | 元数据、PDF 下载、页数核验；未逐页精读 | Codex / 2026-09-04 | 本地 PDF：`papers/pdfs/implicit_differentiation_lasso_hyperparameter_2020_bertrand.pdf`。 |
| P-011 | IRMv1 practical linear relaxation can fail on simple population problems | https://proceedings.mlr.press/v130/kamath21a.html | AISTATS 2021 / PMLR | VERIFIED | 官方 PDF 摘要、§2.2、§3、§3.1 与结论定向精读 | Codex / 2026-09-05 | 已核对原始 scalar-dummy-classifier penalty `grad_w R_e(w*Phi)|_{w=1}`；文章证明 population practical linear IRM/IRMv1 可在 Colored-MNIST-like two-bit environments 失败、并讨论 representative environments。未核验到当前连续部分观测 Gaussian 的 exact source-ERM/zero-penalty/correlation-ball 等式；该差异不足以支持 standalone novelty。 |
| P-012 | class-conditional invariance insufficient for DG | https://proceedings.mlr.press/v139/mahajan21b.html | ICML 2021 / PMLR | VERIFIED | 官方 PMLR 摘要核验 | Codex / 2026-09-04 | 已有 invariance 满足而 unseen-domain failure 的结构反例。 |
| P-013 | distribution similarity alone cannot guarantee adaptation | https://proceedings.mlr.press/v9/david10a.html | AISTATS 2010 / PMLR | VERIFIED | 官方 PMLR 摘要核验 | Codex / 2026-09-04 | `Ω-only` 分布无关风险结论属于已知 impossibility 家族；本项目只能提出更窄、非平凡的版本。 |
| P-014 | joint shift separates domain `P(x)` and task `P(y|x)` | https://chongkaigao.com/files/Invariant_Learning_on_Domain_Generalization_with_Variable_Tasks.pdf | 作者公开 PDF，发表信息待核验 | PARTIALLY_VERIFIED | PDF 摘要与 §1 定向精读；未核验正式版本 | Codex / 2026-09-04 | 仅作为跨任务历史近邻保存；第一阶段不把它作为跨域证书路线的证据或差异点。 |
| P-015 | Le Cam deficiency 给出 feature representation 相对原始 observation 的 decision-risk information loss | https://arxiv.org/abs/1402.4884 | arXiv preprint | VERIFIED | 官方 PDF 摘要、§3--5；feature gap、deficiency 与 Bayes-risk relation | Codex / 2026-09-04 | 支持将 representation insufficiency 定义为统计决策/风险对象；讨论 generic feature learning 和 reconstruction，不提供本项目的 OOD regularizer-path theorem。 |
| P-016 | proper loss、Bayes risk、Bregman divergence 与 statistical information 的标准对应 | https://jmlr.org/papers/v12/reid11a.html | JMLR 12 (2011), 731--817 | VERIFIED | 官方 JMLR 页面与 PDF 摘要、§3--4 定向精读 | Codex / 2026-09-04 | 支持 proper-loss Bayes/Bregman language；二元实验为主要表述，不直接给 domain-generalization regularizer attribution。 |
| P-017 | Anchor Regression 的结构干预鲁棒性 | https://doi.org/10.1111/rssb.12398 | JRSSB 2021 | VERIFIED | DOI 元数据、开放摘要与 DRIG §2.3 对原始 SCM/objective 的复述交叉核验 | Codex / 2026-09-05 | 观察到 exogenous anchor 的加性结构干预；对 anchor-induced perturbation class 给 distributional robustness。当前 conditional relation ball 不自动属于该类。 |
| P-018 | DRIG 对一般噪声干预的最坏风险保证 | https://arxiv.org/abs/2307.10299 | arXiv v2 (2025); JASA 2026 DOI `10.1080/01621459.2025.2544365` | VERIFIED | 原始 PDF 摘要、§2.1--2.3、Theorem 3--4 定向精读 | Codex / 2026-09-05 | DRIG 以 gradient-invariance regularizer 实现 source-derived PSD noise-moment uncertainty set 的 worst-case squared-risk minimization；Anchor 是 additive-mean special case。与当前 `R`-ball/独立 moment ball 相关但不相等。 |

## 新增证据的最小字段

- 原始来源（DOI、PMLR/CVF/期刊官网或 arXiv）；不要只留搜索页。
- 论文存在性、版本、同行评审状态。
- 与本课题直接相关的一段命题/定义及其定位（章节或页码）。
- 证据支持什么、不支持什么，以及可能的反例或限制。

## 当前闸门的证据使用

当前需要按方法逐个核验四类近邻，而不是先检验一个 `Omega-only certificate`：

1. 该正则的原始 objective 是否已被重写为相同或等价的 operator；
2. 该 operator 的 range/nullspace、source observability 与 target harmful directions 是否已有相同刻画；
3. controlled/blind/interaction error accounting 是否已在相同任务保持干预接口中完成；
4. 是否已有严格相同的 positive theorem 与 blind/failure pair。

P-005/P-007 是 derivative/moment operator 路线的高风险近邻；P-011/P-012 是 invariance/IRM blind-boundary 的高风险近邻；P-017/P-018 是 uncertainty-set-matched positive-control 的高风险近邻；P-001/P-002/P-008/P-009 限制把新的 latent/error decomposition 当作贡献。上述结论不能直接推出“完整覆盖”或“存在新颖性”。

`C010` 新增的内部事实是：在一个 centered scalar relation family，IRMv1 radial response 对 relation 是二次多项式，full-rank quadratic source design 将 penalty 桥接到 effective nuisance coefficient。该代数事实尚未完成与 P-011、P-006、P-007 的 theorem-level collision comparison，故状态为 `LITERATURE_AUDIT_OPEN`，不支持新颖性判断。

2026-09-04 的 cross-decomposition retrieval 仅服务于历史路线，状态仍为 `RETRIEVAL_INCOMPLETE`。新的 literature audit 必须对 C006 及每个 C007/C008 Claim 按 operator、target family、observability、error accounting、negative boundary 和 empirical prediction 六轴更新矩阵。
