# OOD Regularization → Representation → Error

一个围绕 OOD / Domain Generalization 理论问题的研究仓库：**不同正则化目标如何改变表示空间，并由此影响跨环境误差？**

本仓库不是把“某个正则在几个 benchmark 上更好”当作结论，而是把研究拆成可审计的链条：

```text
正则化目标 Ω  →  参数响应  →  表示空间几何/统计量  →  OOD 误差分量
```

当前的研究假设、而非已验证结论，是：可以把算法自身的正则项视为一个可下降的指标，研究它下降时隐空间到底发生了什么，并进一步判断这些变化能否进入跨域/跨任务误差的上界分析。

## 从这里开始

1. 阅读 [项目总提示词](PROJECT_PROMPT_CN.md)、[上下文管理](CONTEXT_MANAGEMENT.md) 和 [课题简报](docs/research/00_project_brief.md)。
2. 先运行 [文献与新颖性闸门](prompts/01_novelty_and_literature_gate.md)，不要先写“新方法”。
3. 只有取得 `DISTINCT_BUT_RISKY`、`CLEAR_NOVELTY_GAP` 或受限的 `PROBE_AUTHORIZED` 结论，才进入 [理论路线](prompts/02_theory_decomposition.md) 或 [局部实验](prompts/04_local_experiment.md)。
4. 每个阶段都把可复查证据登记到相应 ledger；没有证据时标记 `UNVERIFIED`，不补造引用。

## 当前探索状态

`EXPL-001` 已用统一合成模型比较 ERM、L1、L2、IRMv1、MMD、CORAL、gradient alignment 和 Hessian alignment。执行结果可逐值复现，但科学解释仍是探索性的。主要结果、统计限制和候选风险上界见 `docs/experiments/02_exploratory_regularizer_results.md`。

## 当前研究计划

导师给出的核心问题被固定为：**针对具体算法，分析其正则项下降时 latent representation 的变化，并把这种变化和跨域/跨任务误差联系起来；若可能，给出误差上界或反例。** 因此本项目不把正则项只当作训练技巧，而是把它当作待解释的机制指标。

### 1. 研究问题

给定算法 `j` 的训练目标

```text
min_{theta,w} R_S(w(Phi_theta(x))) + lambda * Omega_j(theta,w),
z = Phi_theta(x),
```

我们要回答三个问题：

1. 当 `Omega_j` 在训练中下降时，`z` 中的 core 信息、spurious 信息、domain 信息、秩、方差、环境矩差异和 head sensitivity 分别如何变化？
2. 这些 latent changes 中，哪些只是塌缩或整体 shrinkage，哪些真正与跨域误差、跨任务误差或 joint domain-task error 有关？
3. 能否把 `Omega_j` 或它诱导的 representation quantity 写进形式化结论：exact identity、local expansion、conditional upper bound，或者证明某类 `Omega-only bound` 不成立？

### 2. 当前证据与暂定判断

`EXPL-001` 的作用是生成猜想，不是验证论文结论。当前最重要的经验信号是：

- 正则项下降本身不保证目标误差下降。CORAL 和 Hessian alignment 能明显压低 covariance discrepancy，但 cross-domain error 没有随之改善。
- 只让不同域的边际表示更接近也不够。MMD 降低 domain probe accuracy，但 joint error 方向不稳定。
- 含标签/风险梯度信息的量更值得推进。Gradient alignment 和 IRMv1 对 cross-domain error 出现更稳定的改善信号，但不能自动推出 cross-task 改善。
- 强 L1/L2 更像信息压缩或表示塌缩，不适合直接解释为有益 OOD 机制。
- 跨域项和跨任务项必须分开定义；一个 source-domain regularizer 不能直接替代 task discrepancy。

### 3. 主理论路线

当前优先路线是平方损失、线性 head 下的 gradient/Hessian risk-difference identity。固定表示 `z` 和 head `w`，令

```text
R_e(w) = E_e[(w^T z - y)^2],
g_e = grad_w R_e(w),
H_e = grad_w^2 R_e(w).
```

若两个环境的 `E[y^2]` 相同，则可得候选恒等式：

```text
R_e(w) - R_e'(w)
= w^T(g_e - g_e') - 1/2 * w^T(H_e - H_e')w.
```

由此得到条件上界：

```text
|R_e(w) - R_e'(w)|
<= ||w|| * ||g_e - g_e'|| + 1/2 * ||w||^2 * ||H_e - H_e'||_op.
```

这条路线的意义是：gradient alignment 和 Hessian alignment 不只是“看起来合理的指标”，而是可能对应风险差中的两个明确项。下一步必须查重该恒等式和近邻上界是否已被已有 DG、MTL、moment alignment 或 transferability 文献覆盖。

### 4. 下一轮实验计划

下一轮不再只问“哪个算法好”，而是围绕可证伪机制跑实验：

| 实验方向 | 要检验的内容 | 失败/转向信号 |
| --- | --- | --- |
| combined gradient + Hessian | 两个项一起是否比单独项更稳定地控制 source-source risk gap 和 target error | 下降但误差无改善，或只由 source risk/shrinkage 解释 |
| coverage sweep | 目标域位于源域 convex hull 内、有限外推、spurious sign flip 时，上界残差如何变化 | 只在某个合成设置有效，无法形成条件定理 |
| IRMv1 projection counterexample | 构造 IRMv1 scalar penalty 小但完整 gradient discrepancy 大的例子 | 反例失败，说明需要补充 rank/angle 条件 |
| CORAL/Hessian/MMD negative controls | 检查 marginal/covariance alignment 为零但 label-risk gap 仍大的情形 | 若反例不存在，需要重新审视候选上界 |
| cross-task protocol | 明确定义 target task 的 head adaptation 和 task discrepancy | domain discrepancy 能解释一切，或 task term 不可估计 |

这些实验必须先写预注册：生成模型、正则定义、固定项、种子、指标、成功/失败/无结论标准。只有通过 prior-art gate 或受限的 `PROBE_AUTHORIZED`，才升级为正式本地验证。

### 5. 当前可写成数学命题的候选

- `CONJECTURE`：在平方损失、线性 head、source/target moment coverage 与 bounded head norm 条件下，source risk 加上 gradient/Hessian discrepancy 和 coverage residual 可以上界 target risk。
- `CONJECTURE`：IRMv1 的 scalar gradient penalty 通常只能控制 `w^T g_e` 投影，不能控制完整环境风险差；需要额外 rank、angle 或一维表示条件。
- `CONJECTURE`：跨任务误差上界至少需要 task discrepancy、representation sufficiency residual 和 head adaptation protocol，不能只靠 domain regularizer。
- `NEGATIVE_RESULT`：分布无关的 `target risk <= C * Omega_j` 通常不成立，因为正则可被重标度，且表示塌缩可让 alignment penalty 变小但任务误差不小。

### 6. 工作闸门

1. **文献闸门**：优先查重 gradient/Hessian risk identity、IRMv1 projection gap、moment alignment risk bound。若发现 exact equivalent，停止把恒等式本身作为贡献。
2. **反例闸门**：先构造最小反例，证明哪些正则项不能单独成为上界。
3. **验证闸门**：把通过反例筛选的命题转成预注册实验，复跑、统计解释和独立审查后才允许写成 `PASS_LOCAL_SIGNAL`。
4. **论文闸门**：只有当新颖性、理论义务和实验信号同时站住，才进入论文大纲和正式写作。

## 目录

```text
.
├── PROJECT_PROMPT_CN.md       # 每次项目工作时的主提示词
├── CONTEXT_MANAGEMENT.md      # 研究状态、术语和证据纪律
├── prompts/                   # 可直接复制给 Codex 的阶段性 prompts
├── docs/
│   ├── research/               # 课题、检索、文献矩阵与新颖性记录
│   ├── theory/                 # 定义、假设、定理和证明账本
│   ├── experiments/            # 预注册、运行和结果验证
│   └── decisions/              # 可追溯的阶段决策
├── src/ood_repr_reg/           # 可复现实验代码
├── configs/                    # 实验配置，提交到 Git
├── tests/                      # 对度量、数据生成和复现性的测试
├── notebooks/                  # 仅用于探索；可复现结论必须迁移到 src/ 或 docs/
└── artifacts/                  # 本地输出，不提交到 Git
```

## 研究顺序与停止条件

| 阶段 | 目标 | 允许推进的条件 | 主要产物 |
| --- | --- | --- | --- |
| 0 | 把母题收敛成可证伪问题 | 能给出对象、机制、可观测量与反例边界 | `research/00_project_brief.md` |
| 1–2 | 文献与新颖性闸门 | 非 `EXACT_ALREADY_DONE` / `CLOSE_EQUIVALENT` | 证据包、文献矩阵、重叠表 |
| 3–4 | 理论建模 | 明确可证明的简化模型与假设 | 定义/假设/定理 ledger |
| 5 | 小型机制实验 | 已通过新颖性闸门或有 `PROBE_AUTHORIZED` | 预注册、代码、结果表 |
| 6 | 独立审稿闸门 | 证据足以反驳最强替代解释 | `PASS` / `REVISE` / `PIVOT` / `STOP` |

**不满足推进条件就停止、修改问题或转向；不以“再多跑几组”替代机制和新颖性。**

## 开发约定

- 文献结论、定理前提、实验假设、结果解释必须分开记录。
- `notebooks/` 不承载唯一的研究结论；可复现代码放入 `src/`，配置放入 `configs/`，结果写回实验 ledger。
- 使用固定随机种子、记录环境/数据版本，并预先声明失败或无结论标准。
- `artifacts/` 和原始数据不进 Git；仅提交可复现的代码、配置、文档和轻量汇总结果。

详见 [Prompt 索引](prompts/README.md)。
