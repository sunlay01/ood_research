# 实验账本与预注册

> 只有得到新颖性闸门通过或 `PROBE_AUTHORIZED` 的机制实验可以填入本表。探索性可视化须明确标记，不能事后改写为验证性实验。

| ID | 状态 | 对应 RQ/定理 | 假设 | baseline / candidate | 主要指标 | 机制诊断 | 预注册失败标准 | 预算 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | PLANNED | TBD | TBD | TBD | TBD | core/spurious loading、环境 moment、有效秩等（待选） | TBD | ≤ 1 小时本地 |

## 探索性实验（不占用验证编号）

| ID | 状态 | 目的 | 方法 | 输出限制 | 预算 |
| --- | --- | --- | --- | --- | --- |
| EXPL-001 | COMPLETE / EXPLORATORY_SIGNAL | 比较七类正则下降时的 latent response 与 cross-domain / cross-task error，生成可证伪猜想 | ERM、L1、L2、IRMv1、MMD、CORAL、gradient alignment、Hessian alignment | 已形成 3 个 `CONJECTURE` 和 1 个 `NEGATIVE_RESULT`；执行可复现，但统计推断为 `CAUTION`，不得给出验证或新颖性结论 | 首次主运行 95.1 秒；完整复跑一致 |

完整预注册见 `docs/experiments/01_exploratory_regularizer_sweep.md`。

结果、限制与下一证明义务见 `docs/experiments/02_exploratory_regularizer_results.md`。

## 每个实验的记录模板

```markdown
## EXP-XXX — 标题

### 目的与可反驳假设

### 闸门依据
- Prior-art verdict：
- 或 Probe artifact：

### 生成模型、数据与环境变化

### 方法与公平对照
- 固定项：
- 唯一改变项：
- 种子：

### 指标
- 任务表现：
- 机制量：
- 反事实/对照：

### 成功、失败、无结论标准

### 命令、环境与配置版本

### 结果（不可覆写）

### 验证与替代解释

### 判定
`PASS_LOCAL_SIGNAL` / `REVISE_AND_RERUN_ONCE` / `PIVOT` / `STOP` / `INCONCLUSIVE_STOP`
```
