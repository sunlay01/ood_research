# ADR-001：优先检验 gradient/Hessian 风险差路线

- 日期：2026-09-04
- 阶段：探索后、文献新颖性闸门前
- 状态：`DEFERRED_PENDING_PRIOR_ART`

## 背景

EXPL-001 同时运行七类正则。CORAL/Hessian 能稳定压低 covariance discrepancy 但没有降低跨域误差；gradient alignment 与 IRMv1 对跨域误差出现更稳定的方向。平方风险可以写出 gradient/Hessian 风险差 exact identity。

## 决定

下一轮优先检索并审查“gradient difference + Hessian correction → risk-difference bound”路线，同时保留以下竞争解释：

1. 结果只由当前 core/spurious 生成模型造成；
2. gradient alignment 的收益来自隐式 norm/shrinkage，而非 label-aware discrepancy；
3. source-source alignment 无法控制 convex hull 外的 target shift；
4. 跨任务误差需要完全不同的 task discrepancy。

## 未选择的路线

- 不把 marginal MMD、CORAL 或 Hessian penalty 单独作为通用 target-error bound；当前已有反例信号。
- 不以 L1/L2 强正则结果提出 OOD 机制；它们被 source error 与 rank collapse 混淆。
- 不因本地信号宣称新颖性或论文潜力。

## 下一闸门

必须完成精确定理层面的近邻检索。如果已有 exact equivalent，转向 target coverage、task interaction、bound tightness 或明确的负面定理。
