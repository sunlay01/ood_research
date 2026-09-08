# 第一轮报告：Risk-Sufficient State + Exposure + Regularizer Action

## 结论先行

当前最稳妥的裁决是：`REVISE`，不是 `CONTINUE` 作为统一 OOD 理论，也不是 `KILL`。

原因是：线性 moment family 中确实存在严格的风险状态 reduction 和非循环的 source ambiguity 定理；但三种正则不共享一个 operator，IRMv1 只有在 relation design 满秩时存在条件 bridge，CORAL 只观察 covariance response，L2 只有 generic norm bound。因此研究对象应是“同一风险状态上的异质 action 与失败边界”，不能声称统一机制。

## 已证明

1. 共享条件机制下，\(L_f(c,a)\) 通过积分给出环境风险；其最小一般对象是对允许环境族取零积分方向后的 risk-equivalence quotient。
2. 线性平方损失下，\(Q_f=vv^\top\) 与环境矩 \(M_e\) 通过 Frobenius 配对给出 exact risk and transport identity。
3. 对线性 moment family，投影到 \(\operatorname{span}\{M_e\}\) 保持所有允许环境的风险；正交补是 risk-blind quotient。
4. source observation kernel 导致 source-equivalent/target-different 的一般 ambiguity；target 落在 source span 内是可恢复的充分条件。
5. IRMv1 在标量 relation family 中的 response 是二次多项式；三点满秩 design 给出 \(w_A^2\) 的条件 bridge。
6. L2 给出 \(|R_T-R_S|\leq\|\Delta M\|_{op}\|v\|^2\) 的 generic conditional bound。

## 正则实际控制什么

| 方法 | 直接观察 | 可控制范围 | 主要盲区 |
|---|---|---|---|
| IRMv1 | 每环境 radial risk response | 充分 relation exposure 下的 relation-induced nuisance response | tangent response、mean/covariance、低秩 source design、out-of-family target |
| CORAL | 表示 centered covariance difference | 已暴露且经表示保留的 marginal covariance mode | mean、conditional label response、representation information loss |
| L2 | 全局参数/系数 norm | 已知全局 shift budget 下的风险放大规模 | 无法区分 task/nuisance 或 shift 类型 |

## 误差记账

在线性模型中，target degradation 是

\[
R_T-R_S=\langle Q_f,M_T-M_S\rangle_F.
\]

按任何预先给定的正交 projectors，可精确拆成 diagonal component、blind residual 和 interaction；但 projectors 的存在不等于语义识别。若某方法能证明 \(\|P_jv\|\leq b_j(\Omega_j)\)，才可把对应 diagonal term 写成 penalty-dependent bound；否则只能保留 blind/interaction remainder。

## 本轮数值核验

`round1/results/round1_results.json` 显示：原始二次状态维度为 16，当前 source/target moment span 的 rank 为 4，quotient projection residual 约 \(2.2\times10^{-16}\)；三 source risk observation rank 为 3，source kernel dimension 为 13。它们核验定理，不替代定理。

两 relation IRMv1 blind predictor 的 penalty 约 \(2.6\times10^{-32}\)，source risk 为 \(0.51\)，target risk 为 \(1.1000\)。这与“低 penalty 不必然控制 target risk”一致。

## 最大数学漏洞

第一，\(L_f\) 的一般定义仍可能过强，必须使用 quotient 才能避免把完整条件风险换名为 state。第二，source span 的矩投影是风险几何，不是因果语义分解。第三，CORAL 和 L2 的正结果依赖额外 representation、parameterization 与 shift-budget 假设；目前没有从它们原始 penalty 到任务特定 risk component 的普遍 coercive bridge。

## 最终裁决

`REVISE`：保留“风险状态 + exposure + regularizer action”作为分析框架，但把论文主张收缩为：不同正则作用于风险状态的不同观测/响应层，OOD 成功由 target shift 是否落在受控子空间决定，失效由 source kernel、operator blind direction、representation loss 和 interaction 造成。后续任何更强结论都必须先给出对应正则的独立 bridge theorem。
