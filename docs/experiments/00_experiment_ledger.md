# 实验账本：正则作用、盲区与目标误差

> 实验的目的不是为已有方法排名，也不是从 target performance 倒推机制。每个验证实验必须对应一个已定义的 operator/risk Claim；target quantities 只作离线结果或预先声明的理论对象。

## 必须记录的最小字段

| 类别 | 字段 |
| --- | --- |
| 任务与 shift | task mechanism、source/target intervention family、shift geometry、in-family/covered/unseen/out-of-family 标签 |
| 模型与训练 | predictor/representation state space、objective、`lambda`、优化误差、source data/seed |
| 正则机制 | `Omega_j`、actual `L_{j,S}`、zero set/response value、由 source 定义的 observability diagnostic |
| 误差记账 | `R_S-R_S^{X,*}`、`R_S^{X,*}-R_C^*`、`R_T-R_S`、controlled/blind/interaction terms（若已定义）、estimation/coverage/observation remainder |
| 对照 | ERM、matched positive control（仅在 uncertainty set 严格匹配时）、regularizer-specific nullspace counterexample |
| 判定 | 预注册预测、失败标准、替代解释、文献 collision 与是否允许进入下一阶段 |

`Omega`、accuracy 或 PCA factor 的相关性不能独自被登记为机制验证。

## 已完成

| ID | 状态 | 对应 Claim | 作用与限制 |
| --- | --- | --- | --- |
| CAL-001 | `COMPLETE / THEOREM_SANITY_CHECK` | C001, C004a, C002-IRM | population 部分可观测 Gaussian SCM。核对 exact risk identity、worst-case formula、source unobservability 与 IRMv1 zero-penalty counterexample。它不是 benchmark，不能支持新颖性或一般网络结论。 |
| CAL-002 | `COMPLETE / THEOREM_SANITY_CHECK` | C010 | population scalar relation family。核对 ERM mixture cancellation、symmetric ERM positive control、full-rank IRMv1 `w_A^2` bridge、relation/mean/covariance transport 与表示重参数化不变性。它不是 finite-sample 或 neural-network 验证。 |
| EXPL-001 | `HISTORICAL / EXPLORATORY` | 无当前验证 Claim | 历史 regularizer sweep。结果只能提出猜想；不作为 operator、因果变量或风险分量的证据。 |
| EXPL-002 | `HISTORICAL / EXPLORATORY` | 无当前验证 Claim | 历史 PCA/Ward response study。PCA 仅描述响应结构，不能命名真实 latent/causal factors。 |
| LATENT-001-MVP | `STOPPED / CODE_GATE VETO` | C006--C009 候选接口 | MVP 的 projector 代数检查和风险闭合通过，但顺序敏感度约 `0.4116` 导致 `SEMANTIC_DECOMPOSITION_NOT_IDENTIFIED`。第二次 CODE_GATE 又因缺少 source-only lambda-selection 实现和测试而 `VETO`；未运行十 seed 主实验，不能回答七类方法各自能 OOD 什么。 |
| CMNIST-VIS-001 | `PARTIAL_SIGNAL / EXPLORATORY` | feature-response observation | 3 seeds、21 models 的 paired color counterfactual probe。CORAL strength 1 显示 head-used color response 降至 ERM 的 `0.918`，但 latent color response 升至 `1.069`，即 head rejection 而非 feature removal；IRMv1 strength 10 出现 task collapse；L2 基本无选择性效果。不能支持一般 OOD 控制或 Omega-only bound。完整报告见 `docs/experiments/reports/CMNIST-VIS-001_report.md`。 |

已有 CAL-001 报告在 `docs/experiments/reports/C002-IRM_population_sanity.md`；实现与单元测试在 `src/ood_repr_reg/intervention_linear.py` 和 `tests/test_intervention_linear.py`。

## 后续授权顺序

| ID | 前置 Claim | 预注册问题 | 通过标准 | 禁止的解释 |
| --- | --- | --- | --- |
| EXP-C006 | C006 | projector/operator 的 exact split 是否与风险二次型逐项一致？ | 手工 closed form 与 implementation 在 scalar/vector cases 一致；projector 不使用 target risk 选择 | “任意 PCA/聚类方向都是 controlled space” |
| EXP-C010-V | OQ-010 | vector relation polynomial lift 是否保持 C010 的 controlled/blind distinction？ | full-rank design 给出 nonvacuous bridge，低秩 design 给出严格 blind direction | “三个环境自动足够” |
| EXP-L2 | C007-L2 | L2 是否只非选择性缩小 task 与 nuisance dependence？ | source-fit tradeoff 与 predicted controlled/blind contribution 随 `lambda` 的定量曲线匹配 | “L2 学得因果特征” |
| EXP-G | C007-G | full-gradient 的 controlled directions 与 source-fit frontier 是否符合 theorem？ | zero set、nullspace 与 non-vacuity 条件被可反驳核验 | 与 scalar-scale IRMv1 混称 |
| EXP-CORAL/MMD | C008 | moment/IPM reduction 是否只缓解相应受控 shift term？ | 对 matched shift 下降、对构造的 conditional blind shift 失效 | marginal alignment 等于 task invariance |
| EXP-REP | C009 | `WB_A`/`J_Af` 是否给重参数化不变的有效 sensitivity？ | 等价 reparameterization 下风险与机制量稳定 | 仅报告 encoder coordinate norm |

## 实验模板

```markdown
## EXP-XXX — 标题

### 对应 Claim 与结论类型
- Claim:
- statement: exact equality / conditional theorem / counterexample / diagnostic

### 固定假设与干预几何
- task mechanism:
- source observability:
- target family and budget:
- coverage/recoverability:

### 方法的实际 operator
- objective and penalty:
- state space:
- L_{j,S}, zero set, bridge to Omega:
- predicted controlled and blind components:

### 预注册预测与失败标准

### 指标与不可泄漏约束

### 结果、替代解释、审计结论
```
