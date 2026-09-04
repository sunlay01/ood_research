# OOD Regularization → Representation → Error

一个围绕 OOD / Domain Generalization 理论问题的研究仓库：**不同正则化目标如何改变表示空间，并由此影响跨环境误差？**

本仓库不是把“某个正则在几个 benchmark 上更好”当作结论，而是把研究拆成可审计的链条：

```text
正则化目标 Ω  →  参数响应  →  表示空间几何/统计量  →  OOD 误差分量
```

当前的研究假设、而非已验证结论，是：可以构造 representation-level 的误差分解或响应量，使 IRMv1、CORAL、Fishr、L1/weight decay、谱正则等方法在同一语言中可比较。

## 从这里开始

1. 阅读 [项目总提示词](PROJECT_PROMPT_CN.md)、[上下文管理](CONTEXT_MANAGEMENT.md) 和 [课题简报](docs/research/00_project_brief.md)。
2. 先运行 [文献与新颖性闸门](prompts/01_novelty_and_literature_gate.md)，不要先写“新方法”。
3. 只有取得 `DISTINCT_BUT_RISKY`、`CLEAR_NOVELTY_GAP` 或受限的 `PROBE_AUTHORIZED` 结论，才进入 [理论路线](prompts/02_theory_decomposition.md) 或 [局部实验](prompts/04_local_experiment.md)。
4. 每个阶段都把可复查证据登记到相应 ledger；没有证据时标记 `UNVERIFIED`，不补造引用。

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
├── src/ood_repr_reg/           # 后续可复现实验代码（目前不预设方法）
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
