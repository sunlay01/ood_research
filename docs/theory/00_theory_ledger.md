# 理论账本

> 不把“看起来合理”的叙述写成 theorem。每个编号项都要有准确的假设、量纲/可达性检查和反例义务。

## 0. 问题设置

- **环境集合**：`TBD`
- **损失与风险定义**：`TBD`
- **表示/头部参数化**：`TBD`
- **环境变化机制**：`TBD`
- **可观测信息与不可观测变量**：`TBD`

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
