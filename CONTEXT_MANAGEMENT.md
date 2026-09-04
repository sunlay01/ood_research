# 上下文管理与研究状态

## 当前状态

- 项目阶段：0.5 — 已完成一次探索性正则扫描；尚未完成独立的新颖性检索，不能进入论文级验证。
- 工作性母题：**正则化目标如何诱导表示空间的优化，并影响 OOD 风险？**
- 当前最小对象：多环境、core/spurious 分解、线性或浅层表示模型。
- 当前候选方法：ERM、IRMv1、weight decay/L1、谱或低秩正则；它们是比较对象，非承诺都要研究。
- 当前候选路线：representation risk decomposition、nested oracle error、局部 sensitivity / regularization path。尚未选择主路线。

2026-09-04 探索更新：`EXPL-001` 已在统一合成模型上比较 ERM、L1、L2、IRMv1、MMD、CORAL、gradient alignment 与 Hessian alignment。完整复跑逐字节一致；结果只标记为 `EXPLORATORY_SIGNAL`。当前最具体的理论候选是线性平方损失下的 gradient/Hessian risk-difference identity，以及它向 unseen target 扩展时所需的 coverage、task discrepancy 与 sufficiency residual。详见 `docs/experiments/02_exploratory_regularizer_results.md` 与理论 ledger。

## 已知证据状态

讨论链接中提到 Wu et al. (2020)、Galstyan et al. (2022)、Shui et al.、Moment Alignment (2025)、Lai & Wang (2024) 等线索。这些只是**待核验种子**：不要直接引用、不要依赖其中的年份/结论或“尚无人做过”的判断。

2026-09-04 更新：分享链接中可定位到的 10 篇论文已下载到本地 `papers/pdfs/`，并在 `papers/download_manifest.tsv` 和 `docs/research/01_evidence_register.md` 登记。`VERIFIED` 仅表示论文存在性、元数据、PDF 下载和页数已核验；技术命题、定理和证明质量仍需在后续精读阶段逐页确认。

## 术语约定

| 名称 | 约定 |
| --- | --- |
| representation / feature / latent space | 对判别网络中的 `z=Φ_θ(x)`，三者默认指学习到的表示；若使用生成模型含义须另行声明。 |
| environment | 训练或测试分布索引 `e`；必须写清它改变的是何种条件分布。 |
| core / spurious | 生成机制上的定义，不能仅以“与标签相关”代替；需给出环境变化和可辨识性条件。 |
| induced representation regularizer | `Ψ(Z,w)=inf_{θ: Φ_θ(X)=Z} Ω(θ,w)` 类型对象；是否可达、有限、可解须证明或列为假设。 |
| regularization response | `S_{j,k}(λ)=E_k(Z_{λ,j})-E_k(Z_0)`；它是比较量，不自动意味着因果机制。 |

## 证据标签

- `VERIFIED`：已由原始论文/正式出版页面/可靠元数据核验。
- `PARTIALLY_VERIFIED`：论文存在已核验，但具体定理或叙述未逐页核验。
- `UNVERIFIED`：来自聊天、笔记或二手线索，不能作为依据。
- `CONJECTURE`：本项目提出的可证伪主张。
- `NEGATIVE_RESULT`：预注册检验未支持主张；保留而不删除。

## 上下文压缩规则

每次完成阶段时更新对应 ledger，而不是只依赖聊天记录。后续会话读取：主提示词、本文件、课题简报、最新决策记录和当前阶段 ledger。若这些文件与聊天记忆矛盾，以带证据、带日期、带来源的仓库记录为准。
