# ADR-002：主 derivative 路线停止，转向跨域证书审计

- 日期：2026-09-04
- 阶段：Stage 1 prior-art gate
- 状态：`PIVOT_REQUIRED / NO_EXPERIMENT_AUTHORIZED`

## 所依赖证据

- P-005（Hemati et al., ICCV 2023）：head Hessian/gradient 与 transfer measure。
- P-007（Chen et al., UAI 2025）：gradient/Hessian/moment alignment 的统一 multi-source 理论。
- P-011（Kamath et al., AISTATS 2021）：practical IRMv1 的失败。
- P-003、P-013：invariance/相似性不足和 impossibility 的必要附加项。
- P-001、P-002、P-003、P-008、P-009：representation decomposition、DG failure modes、target-risk bound、latent risk bound 和 regularization path 的相邻空间。

## 决定

1. 不把 ID-004/BND-005 作为论文主张，也不为其继续做正式算法实验。
2. 保留 DERIVATION-002/003 作为后续 theorem 和实验的负对照：任何新上界都必须显式排除其塌缩/投影失败机制。
3. 不把 `Omega-only` 反例作为可投稿贡献；它只约束未来命题必须包含 source risk、coverage、充分性或 task 条件。
4. 将课题收敛为跨域算法误差证书审计：固定 source ERM 基准，比较 `Omega-only`、`Omega + source risk` 与完整 source-only observable view 对正向退化 `D_{T,j}` 的解释力。当前这是审计框架候选，不是新算法或已证明上界。
5. 跨任务路线在第一阶段明确搁置，不作为当前研究问题、实验授权依据或新颖性差异点。

## 被拒绝的替代解释

- 用更大 benchmark、更多正则或更多 lambda 扫描来区分已覆盖的 A/B/C；这只会增加实验量，不产生机制级差异。
- 因 P-009 尚未同行评审而忽略其 regularization-path 覆盖；预印本仍是必须面对的近邻。
- 以 EXPL-001 中 gradient alignment 的经验方向替代 theorem/novelty evidence；该实验仍只具有 `EXPLORATORY_SIGNAL`。

## 可撤销条件与下一闸门

若 Stage 2 的六轴核验确认现有工作没有完整覆盖“以算法训练指标为输入、以 source ERM 相对正向跨域退化为标签、并显式分离 source-only terms 与 target coverage remainder”的对象，同时给出可证明的 tightness 或 impossibility 边界，才可将 `AUDIT-001` 改为 `PROBE_AUTHORIZED`。否则保留为审计工具并停止论文主线，不以更多 benchmark 或 lambda 扫描替代差异。
