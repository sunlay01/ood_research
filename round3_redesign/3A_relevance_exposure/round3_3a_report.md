# 3A 报告：OOD Relevance 与 Source Exposure

## 裁决：`RELEVANCE-EXPOSURE-GEOMETRY-PASS`

本轨把旧 3A ambient algebra 保留为底层可见性工具，核心对象改为 source-good neighborhood 中的 model-discriminating vulnerability。没有进行聚类、机制命名、响应阶数或正则控制。

## 数学核验

- residual-coupling identity 最大误差：`5.439e-16`。
- local relevance 使用 `g_s = 2 Delta M_s w*_S - 2 Delta m_s = -2 E_T[X r*]`。
- quadratic source risk 的 near-optimal set 是 Hessian 椭球；finite-epsilon vulnerability 使用 trust-region 求解。
- trust-region 与独立 SLSQP 核验相对误差：`4.357e-11`。

## 关键边界

`source usage != OOD relevance`。在 `rho_S=0, rho_T!=0` 时，source shortcut weight 可以为零，但 target residual coupling 仍可非零；这不是失败，而是 target-emergent relevance 的正式反例。

独立噪声的 source 权重为零：`True`。在 `rho=0.8` 时 shortcut source usage 为：`True`。`rho=0` 的 target-emergent relevance 为 `0.422476`。

## Noise explosion attack

| K | ambient rank | relevance effective rank | max noise relevance | shortcut relevance |
|---:|---:|---:|---:|---:|
| 0 | 2 | 2 | 0.000e+00 | 2.1086 |
| 4 | 6 | 2 | 0.000e+00 | 2.1086 |
| 16 | 18 | 2 | 0.000e+00 | 2.1086 |
| 64 | 66 | 2 | 0.000e+00 | 2.1086 |
| 256 | 258 | 2 | 0.000e+00 | 2.1086 |

Ambient moment visibility grows with irrelevant coordinates, while the
whitened relevance rank remains 2 and the independent-noise coupling remains
zero. This is a benchmark result, not a universal theorem for arbitrary model
classes.

## Positive and invariance controls

The genuine-shortcut response ranks are `1→1, 2→2`; adding a second
genuine shortcut adds a response direction. Redundant-copy collective probes
are recorded separately and are not interpreted as one-axis semantic factors.
Across `20` orthogonal noise rotations, shortcut relevance ranges
from `2.10858` to
`2.10858`, while maximum noise
relevance is `0.000e+00`.

## 结论边界

本实现提供的是 population quadratic-risk theorem、conditional trust-region calculation 和 exposure diagnostics。四象限标签只描述 relevance/exposure 数值，不是生成机制语义。后续机制模块与 regularizer control 不在本轨中。
