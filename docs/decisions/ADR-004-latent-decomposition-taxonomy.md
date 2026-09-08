# ADR-004：先建立隐空间分解图谱，再做 ERM 配对归因

## 状态

`ACCEPTED_FOR_STAGE_1 / 2026-09-04`

## 背景

此前的探索实验只实现了 representation oracle 与 fitted-head mismatch 的相对 ERM 记账。这两项可以精确回答一部分问题，但不能覆盖隐空间结构、nested failure、transport regularity 或正则化 mode path。若直接把它们当作唯一分解，容易把 latent information loss、head mismatch、conditional shift 和 collapse 混在一起。

## 决策

先建立候选 decomposition taxonomy，再选择可审计的最小实验矩阵。taxonomy 采用八类 construction：

- 条件期望/Bayes projection；
- representation-conditioned risk transport；
- nested-oracle failure decomposition；
- latent direct-sum/factor decomposition；
- latent smoothness/coverage transport；
- approximation-estimation-optimization-shift 外层脚手架；
- feature deficiency/decision-risk comparison；
- objective-to-mode/spectral regularization path。

第一轮实现优先级为 A/C/D/E。所有算法都与固定 source ERM 配对；同一算法的 lambda 路径另行报告。不同 construction 之间只比较预先登记的 semantic factor，不做分量求和或统一排序。

## 理由

1. A 提供 representation information 与 head fit 的 exact risk identity。
2. C 提供训练/测试 representation 与 classifier oracle 的可操作反事实。
3. D 提供 invariant、spurious-invariant 和 variant latent structure 的候选几何。
4. E 检查 latent representation 的 conditional transport、smoothness 与 coverage，防止把 marginal alignment 当成泛化保证。
5. F/G/H 是重要的外层解释，但分别属于学习误差、统计决策和路径动力学，不能冒充 A/C/D/E 的 risk component。

## 后果

- 当前 latent oracle 实验降级为 A construction 的有限样本 proxy。
- 正则原始标量 \(\Omega\) 不再直接解释为任何隐空间误差；必须通过响应签名与诊断量观察。
- `AUDIT-001` 在 prior-art gate 完成前仍不授权正式实验。
- 若 A/C/D/E 的 crosswalk 只产生已有工作的重述，则停止“新分解框架”路线，不通过增加 benchmark 或正则强度规避闸门。

## 相关记录

- `docs/research/04_latent_space_decomposition_taxonomy.md`
- `docs/theory/00_theory_ledger.md` 的 `TAX-001` 与 `CMP-001`
- `docs/theory/01_dual_track_decomposition.md`
