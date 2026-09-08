# 第二轮报告：异质正则与公共 OOD 风险商

## 最终裁决：`REVISE`

第二轮确认了一个比第一轮更准确、但更弱的统一形式：IRMv1、CORAL 和 L2 不共享同一作用算子；它们可以被放到同一个 target-risk quotient 中比较，但 induced cost 的数学性质不同。

最重要的结果不是 T6，而是：source kernel 不能直接叫作 OOD ambiguity。真正的 ambiguity 必须同时 target-visible。当前 population 构造中，symmetric risk-state 的 ambient dimension 是 10，target-visible quotient dimension 是 4，source absolute observation rank 是 3，因此真正的 source-unidentified/target-visible ambiguity dimension 是 1；raw source kernel dimension 是 7，其中 6 个方向对声明的 target family 也不可见。

## 已证明的命题

| 编号 | 命题 | 状态 |
|---|---|---|
| T5 | target annihilator、source kernel 与 target-visible ambiguity 的 quotient/rank characterization | `theorem` |
| T5b | target moment 在 source span 内时可由 source absolute risks 恢复 | `theorem` |
| T5c | Hilbert/Frobenius shift ball 的 support function | `exact equality`；PSD/参数化约束下为 `conditional` |
| T6 | fiber infimum 与 quotient objective 的交换 | `theorem`，但属于基础 infimal projection |
| T7 | rank-one 直接 predictor 上 L2 induced cost 闭式 | `exact characterization` |
| T8 | 无 gauge 的 CORAL induced cost 为 0 | `theorem / degeneracy counterexample` |
| T9 | IRMv1 relation-response 的三点满秩 coercivity | `conditional theorem` |
| T9b | 两 relation 点的 IRMv1 target-visible blind branch | `counterexample` |

## 三种正则的统一比较

| 方法 | quotient 上的状态 | induced cost | 可以控制 | 不能控制 |
|---|---|---|---|---|
| L2 | residual vector 的 rank-one state | 有限，直接 predictor 上可闭式计算 | 已知全局 moment budget 下的整体幅度 | 特定 nuisance/shift 方向；task-nuisance 选择性 |
| CORAL | 需要额外的 representation realization fiber | 无 gauge 时退化为 0；固定 gauge 后为 conditional representation cost | 表示协方差差异 | mean、conditional label response、表示信息丢失 |
| IRMv1 | relation-response quotient，通常还需 realization 信息 | 固定 predictor 上 exact；直接 predictor 的 rank-one sign fiber 可取 exact infimum；三点 relation design 下有条件 coercivity | source relation 足够丰富时的 relation-response/nuisance use | 低秩 source design、mean/covariance、径向投影的 tangent、out-of-family target |

因此“统一”只能表示：三者最终都通过某个风险状态影响 target risk；不能表示它们直接优化同一个 latent quantity。

## Failure separation

### Source-unidentified

source risk 观察看不到，但 target risk 可以看到的方向属于 ambiguity。它说明数据和 source environment design 本身不足，不能把失败全部归罪于正则。

### Regularizer-blind

source quotient 已经包含该方向，但具体 induced cost 沿该方向 flat 或曲率不足，才叫正则没有利用已有信息。IRMv1 两点 relation 的 response nullspace 是这种分析的候选，但若同时 source absolute observations 也无法区分，则只能标为 source-unidentified。

### Destructive

正则 cost 偏好 target-risk 更差的状态，或惩罚 target-good direction。这需要比较 induced cost 与 target risk 的排序，不能从 penalty 单调下降直接推出。

### Quotient-external

CORAL 的 representation rescaling 是参数化/gauge 退化：它不能被误写成 risk quotient 内的 OOD 失败，而应单列为 quotient-external degeneracy。

## 误差记账

在线性平方损失下仍使用 exact identity：

\[
R_T-R_S=\langle Q_f,M_T-M_S\rangle_F.
\]

若目标 shift 属于 Hilbert 子空间 \(\mathcal H\) 且预算为 \(\rho\)，则未识别风险差的 sharp support function 是

\[
\rho\|\Pi_\mathcal H Q_f\|_F.
\]

IRMv1 满秩 relation design 还可以把 \(w_A\) 的一部分替换为 penalty 和 \(\sigma_{min}(V_S)\) 的条件界；L2 只能通过 \(\|v\|\) 给 generic bound；CORAL 无 gauge 时没有有效的 penalty-to-risk bridge。

## 数值核验

结果见 `round2/results/round2_results.json`：

- target-visible dimension：4；
- source rank in quotient：3；
- true ambiguity dimension：1；
- raw source kernel dimension：7；
- target outside source span 的恢复残差：约 0.7071；
- 两点 IRMv1 penalty：约 \(2.6\times10^{-32}\)；
- 对应 source risk：0.51；target risk：约 1.1569；
- 三点 relation operator rank：3；
- CORAL 无 gauge induced cost：0。

这些数字只验证定理的实例化和反例，不构成一般性证明。

## 最大数学漏洞

第一，T6 是已知的 infimal projection 事实，不能作为创新。第二，CORAL 和表示正则的 induced cost 需要 gauge；无 gauge 时问题退化。第三，IRMv1 的 coercivity 只针对标量 relation-response quotient，不能自动推广到完整 representation 或任意 target shift。第四，IRMv1 的径向响应一般不由 rank-one 风险状态唯一决定，公共 quotient 需要保留 realization 或取 fiber 聚合。第五，本轮没有建立有限样本置信界，也没有完成系统的原始文献页码级查重。

## 研究判断

公共 OOD risk quotient 是有数学内容的分析坐标，但“fiber infimum 统一异质正则”本身过于形式化。当前应保留并继续修正的贡献候选是：在同一 target-visible quotient 上，严格区分 source-unidentified、regularizer-blind 与 parameterization degeneracy，并把各自的风险作用写成可检验的 conditional accounting。若后续文献核验表明这一 failure separation 已被完整覆盖，则终止该论文路线。
