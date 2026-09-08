# ADR-003：表示层分解先行与跨分解归因闸门

- 日期：2026-09-04
- 状态：`ACCEPTED / RESEARCH_GATE_OPEN_ONLY_FOR_RETRIEVAL`
- 决策者：项目研究记录

## 背景

用户已明确否决以 source-only certificate 为第一步的路线。主问题应先问：给定
`(theta_lambda,j, w_lambda,j) in argmin R_S + lambda Omega_j`，现有算法在严格的
representation-level decomposition 中改变什么；target risk 只用来检验已定义分量的外部含义。

Wu et al. (2020, P-001) 已给 representation risk decomposition；Galstyan et al.
(2022, P-002) 已用 nested-oracle error components 比较 DG 算法、regularization strength
和训练时长。故不能以新的单一分解或 regularization sweep 作为论文差异。

## 决策

1. 主链固定为 `objective -> Z_lambda -> E_k^(d)(Z_lambda) -> risk check`。
2. 算法先使用 `TAX-001` 的机制标签审计：marginal/moment alignment、conditional/head-response
   alignment、sensitivity/smoothness、compression/collapse。标签可重叠，且不构成因果结论。
3. 用 `S_{j,k}^{(d)}(lambda)` 表示 construction `d` 内的响应。不同 construction 的项不可相加。
4. 跨分解比较只有在共同模型类中预注册 semantic crosswalk 和正尺度后才允许；否则输出
   `UNDEFINED`，不能用标准化、排名或图形叙事强行比较。
5. 当前唯一可继续查重的窄候选是：同一算法、同一路径在多个严格 construction 下的 attribution
   agreement/disagreement，以及可证明的 crosswalk 或 impossibility。该候选不是新颖性结论。
6. `AUDIT-001` 和任何正式实验保持阻塞。target-risk certificate、relative degradation 与 coverage
   remainder 被降级为第四步风险检验，不主导分解选择。

## 后果

- 研究工作立即转为针对 P-001/P-002/P-008/P-009 的引文链和六轴比较；不运行新的 benchmark、lambda
  sweep 或算法实现。
- 若跨分解归因已被完整覆盖，裁决为 `STOP`，不以 dashboard 或更多可视化包装为论文。
- 若未被覆盖，仍须先在可解模型中证明 nontrivial crosswalk 或不可比性，再考虑一个区分机制的最小 probe。

## 证据与不确定性

- `VERIFIED`：P-001、P-002 已是单一 decomposition / algorithm comparison 的直接近邻。
- `VERIFIED`：P-008、P-009 压缩新的 latent decomposition 和 objective-to-mode path 的空间。
- `UNRESOLVED`：截至本次精确关键词定向检索，没有检出以 cross-decomposition attribution agreement 为主张的
  直接命中；这只是检索不足，不能支持 `CLEAR_NOVELTY_GAP`。
