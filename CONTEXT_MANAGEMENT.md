# 上下文管理

## 唯一事实来源

长期研究状态只写在仓库内，不以聊天记录为准：

```text
README.md                                  # 当前中心问题与形式化
PROJECT_PROMPT_CN.md                       # agent 工作约束
docs/research/research_state.md            # 短状态与 Claim 队列
docs/research/open_questions.md            # 未决问题
docs/theory/00_theory_ledger.md            # 定义、恒等式、条件结果和反例
docs/research/01_evidence_register.md      # 文献证据
docs/experiments/00_experiment_ledger.md   # 已授权实验及限制
docs/decisions/                             # 可追溯路线决定
```

开始任何工作前先读 `README.md`、`research_state.md`、`open_questions.md` 和最新 ADR。聊天和附件可提供建议，但不能覆盖这些文件中的已记录事实，除非由 Lead 修改相应记录。

## 活跃范式

活跃主线是：

```text
regularizer Omega_j
  -> learned source-supervised semantic latent subspaces
  -> controlled task-preserving shifts
  -> blind/mixed component causing failure
  -> component main effects + interactions + remainder error accounting
```

不是 `Omega_j -> universal certificate`，也不是以 operator/nullspace 或 intervention-risk ANOVA 替代 latent decomposition。Certificate 是盲区与 remainder 被额外假设消除或控制时的子结果。PCA/cluster 不能命名语义；target 不能参与分解或选择。

`LATENT-001` 是当前唯一实验主线，冻结合同位于 `docs/experiments/LATENT-001_contract.md`。执行受 `DESIGN_GATE`、`MVP_GATE`、`CODE_GATE`、`RESULT_GATE` 约束；Supervisor 是独立 ephemeral read-only Codex，Lead 不得覆盖其 `VETO`。

## Claim 协作协议

一个 Claim 至少有：

```text
Statement; status; assumptions; state space and operator;
controlled/blind definition; theory evidence; counterexample evidence;
literature collision; executable check; audit decision.
```

状态只可为 `UNVERIFIED`、`PARTIAL`、`PROVED`、`DISPROVED`、`COLLIDED` 或 `REVISE`。任何 agent 的单次输出只是证据，不直接改变状态。

- Theory：给出精确定义、证明、常数与适用范围。
- Counterexample：攻击 nullspace、可行前沿、source invisibility、抵消、重参数化和 out-of-family shift。
- Literature：核对原始结果的 intervention set、观测变量、假设与 theorem scope。
- Experiment：只实现已有 Claim 的可反驳预测或代数核验。
- Auditor：检查 claim inflation、符号、oracle 泄漏、未申明假设和实验混淆。

## 文件写入规则

1. 每个结论标为 `definition`、`assumption`、`exact equality`、`conditional theorem`、`bound`、`diagnostic`、`counterexample` 或 `conjecture`。
2. 一般保留旧记录；用户已明确要求永久删除的错误 `MECH-001/C011` 是例外，不能恢复或引用。
3. source-only training/selection 不能使用 target risk、target moments 或 target coverage residual；它们可做离线标签或理论 remainder。
4. `Omega` 小、representation compact、PCA factor 或 cluster 都不是 OOD 机制结论。必须先给出实际 operator 和 risk bridge。
5. 对每个正向 theorem，同时写明 operator nullspace 或其没有危害的额外条件。
