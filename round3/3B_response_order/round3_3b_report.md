# 3B 报告：OOD Risk-Response Differential / Order Filtration

## 裁决：`SECOND-ORDER-CLOSURE`

本轨只分析 3A risk-response space 的 differential order。没有 clustering、机制命名、正则控制或新的 ambient space。语义问题：`DEFER-TO-3C`；正则问题：`DEFER-TO-LATER`。

## 数据与方法

- 使用 3A 已保存的 `660` 个模型权重；没有重新手写 predictor。
- structural path 为 `eta(t)=eta0+t v`，协方差路径不做 eigenvalue clamp；所有参与有限差分的点均保持正定。
- 使用 `h=[0.025, 0.05, 0.1, 0.2, 0.4]` 的正负点，计算一阶、二阶、三阶中心差分；解析导数来自同一 structural moment polynomial。

## 全局 filtration

| quantity | value |
|---|---:|
| rank D1 | 16 |
| rank D2 | 7 |
| rank D3 | 0 |
| cumulative <=1 | 16 |
| cumulative <=2 | 17 |
| cumulative <=3 | 17 |
| new @2 | 1 |
| new @3 | 0 |
| finite pure span rank | 17 |
| rho1 | 6.019e-01 |
| rho2 | 1.308e-15 |
| rho3 | 1.308e-15 |

### Pure-family ranks (nonzero anisotropic base)

| family | finite rank | rank D1 | cumulative <=2 | cumulative <=3 | new@2 | new@3 | moment degree range |
|---|---:|---:|---:|---:|---:|---:|---|
| core_mean | 5 | 2 | 5 | 5 | 3 | 0 | [2, 2] |
| core_covariance | 3 | 3 | 3 | 3 | 0 | 0 | [1, 1] |
| relation | 7 | 4 | 7 | 7 | 3 | 0 | [2, 2] |
| b | 5 | 2 | 5 | 5 | 3 | 0 | [2, 2] |
| mu_xi | 5 | 2 | 5 | 5 | 3 | 0 | [2, 2] |
| sigma_xi | 3 | 3 | 3 | 3 | 0 | 0 | [1, 1] |
| task | 3 | 2 | 3 | 3 | 1 | 0 | [2, 2] |

### Base-point comparison

| base | global cumulative <=1 | global cumulative <=2 | global cumulative <=3 |
|---|---:|---:|---:|
| nonzero-anisotropic | 16 | 17 | 17 |
| symmetric-zero-mean | 16 | 17 | 17 |
| generic-2 | 16 | 17 | 17 |

## 解析与有限差分

三阶 stencil 采用 ` [f(2h)-2f(h)+2f(-h)-f(-2h)]/(2h^3) `，对 `t^3` 返回 `6`。每个 family 的逐步收敛和相对误差保存在 JSON；rank 只有在解析/有限差分一致且 tolerance profile 稳定时才解释。

一阶有限差分 rank 在所有五个步长均为 16，二阶均为 7，与解析 rank 一致。解析三阶导数严格为 0；由于三阶 stencil 具有 `h^-3` 放大，零导数上的浮点残差在原始 SVD 中产生伪 rank。因此三阶 finite-difference rank 被标记为 `invalid-zero-derivative`，不参与阶数裁决；三阶结论使用解析矩阵及其零范数。

## Mixed paths

Mixed paths 仅报告是否离开 pure-order span，不赋予任何机制语义。它们是 interaction diagnostic，不能替代 pure-family filtration。全量结果显示 `core_mean+relation` 和 `core_mean+task` 的 mixed affine path 可达到四阶；因此本报告的 `SECOND-ORDER-CLOSURE` 只对 pure-family finite span 成立，不是对任意 mixed structural path 的全局二阶结论。机器结果中的 `incremental_over_pure` 是 mixed cumulative span 相对 pure cumulative span 的实际增量；它不等于机制 interaction 的因果解释。

本轮每个 pure family 使用 100 个确定性随机方向，避免旧坐标轴实验的方向欠采样。随机方向下 `core_mean`、`relation`、`b`、`mu_xi` 的 finite ranks 分别为 5、7、5、5；旧坐标轴结果的 4、6、4、4 不能作为充分覆盖下的 family rank。

## 结论边界

- `R^(k)=Phi(Psi^(k))^T` 是 quadratic population risk 下的精确因子化；当 model lifting rank 在相关子空间上无 kernel 时，order rank 与 shift-lifting rank 相等。
- `V^(<=1) subseteq V^(<=2) subseteq V^(<=3)` 是由 span 定义产生的 filtration；新增秩是代数结论，不是机制结论。
- 二阶或三阶新增方向即使存在，也不能命名为 core、nuisance 或 relation：`DEFER-TO-3C`。
- 本报告不把响应幅度或 penalty 下降解释为控制：`DEFER-TO-LATER`。

机器结果见 `results/round3_3b_results.json`，阶数矩阵和汇总 CSV 见同目录。
