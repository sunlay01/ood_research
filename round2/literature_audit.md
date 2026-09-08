# 第二轮文献审计

本文件登记查重对象的数学同构，而不是把“没有搜到”写成新颖性结论。

| 方向 | 需要核对的同构结构 | 当前定位 |
|---|---|---|
| Infimal projection / representation cost | \(\inf_{\theta:q(\theta)=g}\Omega(\theta)\) | T6 是基础工具，不单独主张创新 |
| Matrix factorization / path / nuclear / variation norm | 参数分解诱导函数或矩阵成本 | 用于判断 L2/CORAL fiber cost 是否已有覆盖 |
| Blackwell / statistical experiments | 风险等价、充分性、实验比较 | T5 属于有限维风险 functional 的 quotient 化，需要核对同构 |
| Partial identification / identified sets | source observations 下的 target ambiguity | 与 \(\mathcal A_{S,\tau}\) 比较 |
| DRO / robust Bayes | uncertainty-set support function 与 regularization | 用于核对 T5c 和 L2 generic bound |
| Johansson 2019 / IRMv1 / Anchor / DRIG | source support、IRM failure、干预集合与 target bound | 作为近邻和碰撞审计，不把已有反例包装成新结果 |

当前结论：第二轮的潜在差异不在“存在一个 fiber infimum”这一形式事实，而在能否把 source-unidentified 与 regularizer-blind 严格分离，并在同一 target-visible quotient 上比较三种异质作用。正式论文主张必须在完成原始论文定理、假设和页码核对后决定。

## 已核验的原始页面

- Johansson, Sontag, Ranganath, *Support and Invertibility in Domain-Invariant Representations*, AISTATS 2019, PMLR 89:527--536: <https://proceedings.mlr.press/v89/johansson19a.html>。摘要明确讨论 fixed representation、non-invertibility cost 和 source support coverage；因此与本轮 target-visible/source-unidentified 问题是高相关近邻，不能把 support/representation failure 作为新颖性单独主张。
- Kamath, Tangella, Sutherland, Srebro, *Does Invariant Risk Minimization Capture Invariance?*, AISTATS 2021, PMLR 130:4069--4077: <https://proceedings.mlr.press/v130/kamath21a.html>。摘要明确给出 practical linear IRM/IRMv1 的 population failure、比 ERM 更差的新环境泛化和 sampling fragility；本轮 IRMv1 blind branch 只能作为已知负对照和 quotient 化审计。
- Arjovsky et al., *Invariant Risk Minimization*, arXiv:1907.02893: <https://arxiv.org/abs/1907.02893>。用于核对原始 IRM 目标与 practical scalar penalty 的对象差异。
- Sun and Saenko, *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*, ECCV 2016, arXiv:1607.01719: <https://arxiv.org/abs/1607.01719>。CORAL 的原始对象是深层表示的二阶相关/协方差对齐；本轮 T8 的 rescaling 退化是对该参数化对象的额外审计，不把 CORAL 原论文误读为 target-risk quotient theorem。

## 查重结论

上述来源足以确认：IRMv1 failure、representation information loss、support coverage 和 covariance alignment 都已有直接近邻。当前未完成的是对 induced-cost/infimal-projection、Blackwell/partial-identification 和 Anchor/DRIG 原始定理的逐页核验；因此本轮报告只能给出 `REVISE`，不能给出 `CLEAR_NOVELTY_GAP`。
