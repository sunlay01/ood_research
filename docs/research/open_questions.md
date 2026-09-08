# 开放问题

每项必须先给出精确 state space、目标干预族和可证伪标准；不能直接进入大规模 DG benchmark。

| ID | 问题 | 当前优先级 | 关闭条件 |
| --- | --- | --- | --- |
| OQ-001 | 对 C001 的二次 risk transport，如何从一个实际 `L_{j,S}` 导出不任意的 projector `P_j` 与 `E_ctrl/E_blind/E_interaction`？ | P0 | C010 已给标量 `P=I` 特例；C006 需完成 vector projector、operator 失配反例与可观察条件。 |
| OQ-002 | 什么定义同时包含 penalty nullspace 与 source-unobservable directions，而不把 target information 泄漏到 source diagnostic？ | P0 | 明确 `H_{S,T}`、source response operator 及可观测/不可观测分解。 |
| OQ-003 | L2 在部分观测模型中控制的是全局系数、有效 nuisance sensitivity 还是两者的混合？source fit 代价能否 sharp characterization？ | P1 | 完成 C007-L2 的 theorem/counterexample 对。 |
| OQ-004 | full-gradient matching 的 zero set 何时非空且与 source-optimal fit 相容？它相对 IRMv1 的 controlled/blind spaces 是否严格更大？ | P1 | 完成 C007-G，分别处理 scalar 与 vector case。 |
| OQ-005 | CORAL/MMD 的 moment/IPM operator 如何与 conditional nuisance shift 的风险二次型连接？有哪些 adversarial conditional shifts 留在其 nullspace？ | P1 | 建立具体 bridge 或严格 blind construction，并完成文献碰撞审计。 |
| OQ-006 | 当前 `I_corr/I_mom` 能否在加入 observed anchor 后与 Anchor Regression/DRIG 的原始 uncertainty set 精确对齐？ | P2 | 证明集合相等/严格包含/不可比之一。 |
| OQ-007 | 线性表示的 `WB_A` 如何替代不具坐标不变性的 `B_A`，以及它的 operator/projector 是什么？ | P2 | C009 给出 reparameterization-invariant statement。 |
| OQ-008 | 有限样本中如何估计受控与盲区贡献，同时把 estimator error 和 coverage residual 分开？ | P2 | 先有 population C006/C007，再给 estimator assumptions 与 error terms。 |
| OQ-009 | 该 operator/blind/accounting interface 是否已被 DRIG、moment alignment 或 representation risk decomposition 完整覆盖？ | P0 | 每个候选 Claim 完成 six-axis literature comparison；未检出不算关闭。 |
| OQ-010 | scalar IRMv1 的 quadratic relation-design bridge 能否在 vector `R` family 中由 polynomial lift 或 restricted singular value 推广，且不退化成已有 IRM representative-environment 结果？ | P0 | 给出明确 positive/negative theorem，并完成 Kamath、Lai--Wang、Chen 等的逐定理碰撞审计。 |
