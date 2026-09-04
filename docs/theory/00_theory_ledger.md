# 理论账本

> 不把“看起来合理”的叙述写成 theorem。每个编号项都要有准确的假设、量纲/可达性检查和反例义务。

## 0. 问题设置

- **环境集合**：首个探索模型含两个源环境与一个不可见目标环境；正式 theorem 的 coverage 条件仍为 `TBD`。
- **损失与风险定义**：首个候选对象为平方风险 `R_e(w)=E_e[(w^Tz-y)^2]`。
- **表示/头部参数化**：`z=Bx`，共享线性表示与 task-specific linear heads。
- **环境变化机制**：core label mechanism 保持，spurious-label correlation 与 domain mean 改变；目标域包含 sign flip。
- **可观测信息与不可观测变量**：训练时可观测 source risk/penalty/source moments；target moments 与 target risk 不可用于训练或选参。

## 1. 定义台账

| ID | 定义 | 依赖 | 是否可估计 | 状态 |
| --- | --- | --- | --- | --- |
| DEF-001 | `Ψ(Z,w)=inf_{θ: Φ_θ(X)=Z} Ω(θ,w)`（候选诱导表示正则） | 可实现集合、固定样本/总体定义 | TBD | DRAFT |
| DEF-002 | `S_{j,k}(λ)=E_k(Z_{λ,j})-E_k(Z_0)`（正则响应） | 特定优化选择规则、`E_k` | TBD | DRAFT |

## 2. 假设台账

| ID | 假设 | 作用 | 违反时可能发生什么 | 可否弱化 |
| --- | --- | --- | --- | --- |
| ASM-001 | 多环境间 core 机制稳定、spurious 机制变化 | 定义 OOD 目标 | `core/spurious` 不可分或不可辨识 | TBD |
| ASM-002 | 优化解/正则路径可选择且局部可微（仅用于 sensitivity 路线） | 隐函数定理 | 非唯一、bifurcation 或不可微 | TBD |

## 3. 命题与证明义务

| ID | 类型 | 主张 | 精确前提 | 证明义务 | 反例/边界 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| THM-001 | 候选 | 某正则诱导的 `Ψ` 具有可解释的谱/矩性质 | TBD | 给出可达性、等价变换与最小化解 | 构造同 `Z` 的参数对 | NOT_STARTED |
| THM-002 | 候选 | `E_k(Z)` 与 target risk 的 exact / local / bound 关系 | TBD | 明确是等式、局部展开还是上界 | support shift、head mismatch | NOT_STARTED |
| THM-003 | 候选 | 特定正则沿 `λ` 的响应选择性压制 spurious mode | TBD | 求解或符号分析路径 | uniform shrinkage / collapse | NOT_STARTED |
| ID-004 | exact identity draft | 固定表示与 head、平方损失且环境间 `E[y²]` 相同时，`R_e-R_e' = wᵀ(g_e-g_e') - 1/2 wᵀ(H_e-H_e')w` | 线性 head；二阶矩有限；相同 `E[y²]` | 展开平方风险并核验梯度/Hessian 定义与常数 | 损失非平方、head 随环境改变、`E[y²]` 不同 | ALGEBRA_CHECKED / LITERATURE_UNCHECKED |
| BND-005 | conditional bound draft | 风险差由 `||w||·||Δg|| + 1/2||w||²·||ΔH||op` 控制 | ID-004 的前提 | Cauchy-Schwarz 与算子范数；把实现中的均方 penalty 常数显式换算 | 正则重标度、只控制 source-source、target coverage 缺失 | LOCAL_DERIVATION / TARGET_STEP_OPEN |
| NEG-006 | counterexample obligation | 单独 CORAL/Hessian、marginal MMD 或 IRMv1 scalar projection 一般不足以给出分布无关 target-risk bound | 待分别构造最小二维例子 | 给出 penalty 为零/很小而 risk gap 非零/大的显式分布 | 附加充分性、coverage、秩/角度条件可能恢复 bound | EXPERIMENTAL_SIGNAL / PROOF_OPEN |
| BND-007 | cross-task bound draft | joint domain-task risk 需要 domain discrepancy、task discrepancy、representation sufficiency 与 coverage residual | task-specific heads；目标任务定义待固定 | 明确 task metric、head adaptation protocol 与分解顺序 | unseen task 不在 source task span；label mechanism 任意改变 | NOT_STARTED |

### DERIVATION-001 — 平方风险的 gradient/Hessian identity

- 日期：2026-09-04
- 目标：把算法使用的 gradient/Hessian alignment 指标连接到环境风险差。
- 使用定义/假设：固定 `z` 与 `w`；平方损失；二阶矩有限；比较环境的 `E[y²]` 相同。
- 推导：写 `M_e=E_e[zzᵀ]`、`c_e=E_e[zy]`，则 `R_e=wᵀM_ew-2wᵀc_e+E_e[y²]`、`g_e=2(M_ew-c_e)`、`H_e=2M_e`。代入即得 ID-004；再用 Cauchy-Schwarz 和算子范数得 BND-005。
- 结论类型：source-environment `exact equality` 与 `conditional bound`；不是 target-domain theorem。
- 失败点或未闭合步骤：source penalties 如何控制不可见 target 的 `Δg/ΔH`；正则归一化；task shift 时 head 与 `E[y²]` 的变化。
- 需要检索或数值检验的地方：同一恒等式/上界是否已被 gradient matching、Hessian alignment、moment alignment 或 transferability 文献明确提出。

## 4. 推导日志格式

每次推导追加以下区块，而不是覆写失败尝试：

```markdown
### DERIVATION-XXX — 标题
- 日期：
- 目标：
- 使用定义/假设：
- 推导：
- 结论类型：exact equality / conditional theorem / local result / bound / conjecture
- 失败点或未闭合步骤：
- 需要检索或数值检验的地方：
```
