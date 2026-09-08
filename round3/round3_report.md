# 第三轮报告：生成机制驱动的 OOD 风险分解

> **补充实验说明（2026-09-06）**：本文件前半部分保留第一次第三轮实验的历史记录。第一次实验使用手工 predictor library、(d_C=d_A=1) 和 model-side discovery，因此其中的 `effective rank=3` 与 `NO-STABLE-STRUCTURE` 不能作为机制分解结论。补充重做见 [`empirical_discovery/retry/round3_retry_report.md`](empirical_discovery/retry/round3_retry_report.md)。

## 补充重做裁决

补充实验使用 660 个真实 population-trained predictors、300 个 shift probes、(d_C=d_A=2)，并以 shift columns 为主要 discovery 对象。最终裁决为：

```text
LOW-RANK-BUT-NONSEMANTIC
```

响应矩阵的 raw effective rank 为 17，shift-column bootstrap stability 为 0.9694；但加权机制 cluster purity 只有 0.2133，raw 与 shift-normalized 聚类的 pairwise 一致率为 0.5260。稳定结构主要反映风险响应幅度和模型表型，不支持将 cluster 命名为 core、nuisance 或 relation 机制。该结果不是“没有风险结构”，而是“目前没有通过 blind discovery 和事后干预验证得到语义机制分解”。

五种方法的响应也显示出不同的风险行为：ERM、IRMv1 和 V-REx 在当前 source design 下几乎重合；L2 的 source risk 和全局 response amplitude 更大；CORAL 具有独立的 covariance-representation response，并有 45 个 gauge 优化使用同一目标的 Powell fallback。上述差异是方法表型诊断，尚不足以构成 mechanism bridge。

## 第一次实验历史裁决：`PARTIAL-DECOMPOSITION`

本轮在固定的线性 structural family 中得到严格的 risk-response 配对，但没有得到从 observed risk quotient 唯一恢复生成机制标签的 canonical decomposition。

## 逐项裁决

| 问题 | 裁决 | 依据 |
|---|---|---|
| risk-visible direction 是否有机制语义 | `PARTIAL` | 在声明 SCM/tangent family 内可解释；脱离 family 不唯一 |
| 最自然 parameterization | `structural coordinates` | moment coordinates 只描述统计变化；structural coordinates 指定生成模块 |
| CC/CA/AA 是否 canonical | `NOT-CANONICAL` | 固定 causal roles 时是 exact accounting；任意可逆混合会改变 block 含义 |
| model state × shift mechanism 公式 | `PASS / theorem` | \(dR_f[\delta\eta]=\langle Q_f,DM[\delta\eta]\rangle_F\) |
| tangent decomposition | `PARTIAL` | 可按预声明模块定义；模块 overlap 时不能强写 direct sum |
| global 还是 local | `both, with boundary` | tangent 是 local first-order；finite shift 用 exact transport accounting |
| mechanism identifiability | `IDENTIFIABILITY-LIMITED` | \(b\) 与 innovation mean 可互换而观测分布不变 |
| regularizer bridge | `diagnostic only` | L2 是 global envelope，IRMv1 需 relation family，CORAL 需 gauge；本轮未声称普适 coercivity |
| 与已有 decomposition 的关系 | `not a latent decomposition` | 本轮分解的是 mechanism tangent 与 risk response，不是 hidden coordinate 或 condition taxonomy |
| blind discovery | `NO-STABLE-STRUCTURE` | effective rank=3，但 bootstrap stability=0.432 |
| 新 decomposition candidate | `NONE VERIFIED` | 当前 factor 未通过稳定性与干预升格门槛 |

## 理论核验

- tangent finite-difference 最大误差：`8.274e-12`。
- mechanism tangent operator rank：`3`。
- finite-shift interaction closure error：`1.388e-17`。
- `mean_mechanism_nonidentifiability` 是严格 counterexample：相同观测矩对应不同 structural labels。
- `CC/CA/AA` 是 fixed structural roles 下的 exact equality，不是任意坐标下的语义正交分解。

## Blind discovery

- response records：`20`；features：`12`。
- effective rank：`3`。
- bootstrap stability：`0.432`。
- discovery features 不使用机制标签、正则标签或 target 标签。
- cluster 只作 diagnostic；低秩不等于发现机制。

## 正则 probe 边界

ERM/L2/IRMv1/CORAL 在本轮只作为 response probe。任何 penalty 到 mechanism sensitivity 的结论都必须另外给出 operator bridge；当前报告不把硬编码 predictor library 的差异称为训练实验。

## 最终结论

当前最强结果是 `model state × mechanism tangent -> risk response` 的严格局部公式，以及有限 shift 下的 interaction 闭合。机制语义需要外部 structural assumptions；仅从 observed risk quotient 无法 canonicalize。

详细证明与实验协议见本目录下的各专题文件。
