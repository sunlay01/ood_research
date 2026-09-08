# 研究状态

更新：2026-09-05。第一阶段只限 cross-domain。

## 活跃问题

`LATENT-001` 已停止：第二次 `CODE_GATE` 裁决为 `VETO`，按冻结协议不得运行十 seed 主实验。阻断项是 source-only lambda selection 未实现、未测试；当前 runner 只枚举强度网格。若继续该问题，必须新建重新注册的实验，而不能修改本轮结果后第三次重试。

## 已确认结果

| Claim | 状态 | 结论与范围 |
| --- | --- | --- |
| C001 | `PROVED` | 部分可观测 Gaussian SCM 下 `R_T-R_S=b^T(M_T-M_S)b` 是 exact equality；correlation/moment balls 有 exact worst-case formula。 |
| C004a | `PROVED` | source response operator 可有有害的零奇异方向；环境数量不保证 nuisance observability。 |
| C002-IRM | `DISPROVED` | standard scalar-scale IRMv1 可同时达到 source observational optimum 与 zero penalty，却对允许 correlation sign-flip 有正 robust causal excess。它是 blind-component baseline。 |
| C005 | `PARTIAL` | Anchor/DRIG 与当前自由 relation/moment balls related but not identical；不作为未经嵌入证明的 positive control。 |
| C010 | `PARTIAL` | scalar centered relation family 中，full-rank quadratic source design 把 IRMv1 penalty 显式桥接到 `w_A^2`；two-source design 可有 blind branch。target transport 有条件总体界。 |
| LATENT-001-MVP | `DISPROVED AS IDENTIFIED DECOMPOSITION` | projector 对称、幂等、正交、完备且风险记账闭合，但合法语义投影顺序的最大偏差约 `0.4116 > 0.25`；当前 SCM/estimator 下五分量语义 direct sum 不可辨识。该结果不支持七类方法的机制比较。 |

实现与数值核验位于 `src/ood_repr_reg/intervention_linear.py` 与相应 tests。旧的机制实验已按研究决策移除；当前正在执行 source-supervised semantic latent decomposition 实验 `LATENT-001`。

## 当前 Claim 队列

| 优先级 | Claim | 状态 | 成功标准 |
| --- | --- | --- | --- |
| 1 | C006: operator-induced controlled/blind accounting | `PARTIAL` | C010 已给 scalar `P=I` bridge；仍需 vector operator/projector 的非任意构造。 |
| 2 | C007-L2: nonselective shrinkage | `PARTIAL` | 仍需在 LATENT-001 中核验 source-fit/nuisance shrinkage tradeoff 与 sharp population frontier。 |
| 3 | C007-G: full-gradient | `PARTIAL` | shared-head diagnostic 显示 covariance response 可下降，但仍需分类 zero set、可行前沿和 representation nullspace。 |
| 4 | C008: CORAL/MMD operators | `PARTIAL` | 仍需从 objective 导出 controlled moment/IPM term、语义子空间效应与文献 collision。 |
| 5 | C009: representation interface | `PARTIAL` | LATENT-001 将检验 source-supervised semantic projectors 的可辨识性、正交性与重参数化稳定性。 |

## 文献判定

- Kamath et al. (AISTATS 2021) 已覆盖 practical IRMv1 population failure 的核心现象，C002-IRM 不可单独主张新颖。
- Anchor Regression 与 DRIG 是 uncertainty-set-matched robustness 的重要正对照；当前尚无 exact set equivalence。
- Wu、Galstyan、feature deficiency、moment/gradient alignment 等文献压缩“新 error decomposition”与“首次解释正则”的空间。文献登记见 `01_evidence_register.md`。

## 当前决策

`LATENT-001` 冻结合同仍作为审计记录，但执行已在第二次 `CODE_GATE` 的 `VETO` 处终止。C001/C010 保留为数学工具和负对照。PCA、单一 predictor operator 或 IRMv1 负例都不能替代监督式语义 latent decomposition；同样，代数正交而顺序不稳定的 projector 也不能被命名为已识别语义子空间。

## 禁止的结论

- 不从 `Omega_j` 小直接推出 target risk 小。
- 不把 target-dependent blind/coverage term 写为 source-only metric。
- 不把 local nullspace 写成全局 deep-network theorem。
- 不把“没有检出文献”写成新颖性。
