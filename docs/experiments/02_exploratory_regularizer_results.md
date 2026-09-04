# EXPL-001 结果：正则项—隐空间—跨域/跨任务误差

## Material Passport

- Artifact ID：`EXPL-001-RESULT`
- 实验类型：`EXPLORATORY`
- 执行状态：`COMPLETE`
- 执行复现：`VERIFIED`（完整复跑的 `trajectory.csv` SHA-256 一致）
- 科学推断：`CAUTION`（五个种子、多重比较、合成数据、事后机制筛选）
- 配置：`configs/regularizer_sweep_main.json`
- 原始轨迹 SHA-256：`81d826937d5e7e1078b76093596e423d1b3e59abc311fea3908daacec7fbb326`
- 运行环境：Python 3.13.13、PyTorch 2.12.1、NumPy 2.4.6、CPU
- 规模：145 个最终运行，1,015 条 checkpoint 记录；首次主运行 95.1 秒

## 结果摘要

下表固定在 `lambda=1.0`，数值为相同种子、相同初始化下相对 ERM 的配对平均差。误差列为负表示优于 ERM；诊断量的正负只表示变化方向。`Ω ratio` 是最终正则值与初始化正则值之比。

| 方法 | Ω ratio | Δ cross-domain MSE | Δ cross-task MSE | Δ joint MSE | Δ spurious probe R² | Δ covariance gap | Δ effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gradient alignment | 0.033 | -0.1221 | +0.0378 | -0.0940 | -0.0699 | -0.1784 | -0.3366 |
| IRMv1 | 0.155 | -0.1221 | +0.0122 | -0.0223 | -0.0333 | +0.0942 | -0.1310 |
| CORAL | 0.014 | +0.0214 | +0.0078 | +0.0191 | -0.0108 | -0.2864 | -0.1613 |
| Hessian alignment | 0.011 | +0.0173 | +0.0012 | +0.0525 | +0.0139 | -0.2831 | -0.3965 |
| MMD | 0.060 | -0.0177 | -0.0061 | +0.0807 | +0.0431 | -0.0973 | -0.7417 |
| L1 | 0.308 | +0.3943 | +0.1312 | +0.8232 | +0.0314 | -0.2621 | -1.1560 |
| L2 | 0.368 | +0.0713 | +0.2933 | +1.0348 | -0.0133 | -0.2165 | -1.3879 |

ERM 在四个误差指标上的五种子均值分别为：source MSE 0.2662、cross-domain MSE 1.1766、cross-task MSE 0.3068、joint MSE 0.7680。

完整轻量汇总位于：

- `docs/experiments/results/EXPL-001_final_summary.csv`
- `docs/experiments/results/EXPL-001_paired_effects.csv`
- `docs/experiments/results/EXPL-001_regularizer_reduction.csv`
- `docs/experiments/results/EXPL-001_bound_diagnostics.csv`

## EVIDENCE：当前数值直接显示什么

1. **正则下降不等于目标误差下降。** CORAL 和 Hessian alignment 在 `lambda=1` 时把自己的正则压到初始化的约 1%，latent covariance gap 在 5/5 种子中下降，但 cross-domain MSE 也在 5/5 种子中上升。
2. **边际域不可分也不足以保证任务风险。** MMD 在 `lambda=1` 时令 domain probe accuracy 相对 ERM 平均下降 0.1408，5/5 种子同向，但 cross-domain 和 joint error 的方向不稳定。
3. **标签相关的梯度量更接近本生成机制中的风险变化。** Gradient alignment 在 `lambda=1` 时于 5/5 种子降低 cross-domain MSE，并于 5/5 种子降低 spurious probe `R²`；source MSE 只平均增加 0.0027。
4. **IRMv1 与全梯度对齐不等价。** IRMv1 同样在 5/5 种子降低 cross-domain MSE，但 covariance gap 反而在 5/5 种子增加，且 joint error 没有稳定方向。
5. **L1/L2 的强正则结果主要受信息损失混淆。** 两者在 `lambda=1` 时均大幅降低 effective rank 并提高 source/cross-task error；不能把 covariance gap 或 domain probe 的下降解释成有效域不变性。
6. **跨域改善不能外推为跨任务改善。** Gradient alignment 和 IRMv1 的 cross-domain 信号较稳定，但 cross-task error 多数上升，说明一个域正则项不能自动覆盖 task shift。

## INFERENCE：最值得推进的数学对象

在线性 head、平方损失和固定表示下，定义

\[
R_e(w)=\mathbb E_e[(w^\top z-y)^2],\qquad
g_e=\nabla_w R_e(w),\qquad
H_e=\nabla_w^2 R_e(w).
\]

若两个环境的 `E[y²]` 相同，则直接展开得到候选 exact identity：

\[
R_e(w)-R_{e'}(w)
=w^\top(g_e-g_{e'})
-\frac12w^\top(H_e-H_{e'})w.
\]

因此有

\[
|R_e(w)-R_{e'}(w)|
\le
\|w\|_2\,\|g_e-g_{e'}\|_2
+\frac12\|w\|_2^2\,\|H_e-H_{e'}\|_{\mathrm{op}}.
\]

如果算法的 gradient/Hessian penalties 分别是上述差异范数的平方，则它们的平方根可以进入环境风险差上界。这个恒等式同时解释了为什么：

- 只控制 Hessian/CORAL 会遗漏 label-feature cross moment；
- IRMv1 只观察 `wᵀg_e` 的标量投影，通常不足以控制完整梯度差；
- gradient 与 Hessian 的组合比任意一个单项更接近完整风险差。

这只是 source-environment identity。要变成不可见目标域的 OOD bound，还必须加入 target coverage 假设或不可观测残差，例如目标 moments 位于源域 moments 的凸包/有界邻域。当前目标域采用 spurious correlation 翻转，故意违反简单凸包覆盖，不能用本实验声称 target theorem 已成立。

## 候选猜想

### CONJ-EXPL-001：双项风险差控制

在线性平方损失、多源环境且目标 moments 满足明确覆盖条件时，source risk、gradient alignment penalty 的平方根、Hessian alignment penalty 的平方根及 coverage residual 可以共同上界 target risk。单独 Hessian/CORAL 项一般不能完成该上界。

### CONJ-EXPL-002：IRMv1 的投影缺口

除非 latent/head 受到一维、满秩或角度条件约束，IRMv1 的 scalar gradient penalty 不能控制完整风险差；存在 IRMv1 penalty 很小但 orthogonal gradient discrepancy 很大的反例。

### CONJ-EXPL-003：域项与任务项必须分开

跨域—跨任务联合风险需要至少一个 domain discrepancy、一个 task discrepancy、一个 representation sufficiency residual 和一个 coverage residual。只在源域间定义的正则项不能普遍上界 unseen-task error。

### NEGATIVE-EXPL-001：Ω-only bound 不成立

形如 `target risk <= C * Ω` 的分布无关上界一般不成立。表示塌缩可以令若干 alignment penalties 接近零而保持较高任务误差；原始正则还可以被任意常数重标度。上界至少需要 source/task risk、head norm 或 sufficiency/coverage 项，并需固定正则归一化。

## 统计与方法学验证

- 完整复跑与原始轨迹逐字节一致，执行复现状态为 `VERIFIED`。
- 145 个最终运行全部完成，无缺失、无非有限值、无幸存者筛选。
- 五个种子的双侧 exact sign-flip test 最小可能 `p=0.0625`；本次最小观测值也为 0.0625。
- 共报告 308 个方法—强度—指标配对检验；Benjamini-Hochberg 校正后的最小 `q=0.1385`。没有结果应被称为统计显著。
- `final_correlations.csv` 中的 nominal p-values 没有按独立样本解释：不同 lambda 共享 seed/data，且 `lambda` 同时影响 Ω、表示和误差。

### 11 类统计谬误扫描（11/11 checked）

| 项目 | 判定 | 说明 |
| --- | --- | --- |
| Simpson's paradox | CAUTION | 若跨 lambda 聚合会掩盖非单调路径；核心结果同时报告固定 lambda 与逐种子方向。 |
| Ecological fallacy | 不适用 | 推断限定为合成运行层面，不外推个体。 |
| Berkson's paradox | 未发现 | 没有按结果筛选运行。 |
| Collider bias | CAUTION | 最终 Ω 与误差都由 lambda 和优化轨迹决定，二者相关不能解释成因果。 |
| Base-rate neglect | 不适用 | 未使用诊断灵敏度/阳性预测值。 |
| Regression to the mean | 未发现 | 没有按极端初始表现选组，且使用配对 ERM。 |
| Survivorship bias | 未发现 | 145/145 运行完成。 |
| Look-elsewhere effect | RED_FLAG（若作验证性解释） | 多方法、多强度、多指标；BH 后无显著结果。当前明确标记探索性。 |
| Garden of forking paths | CAUTION | 配置与判据运行前固定，但候选机制来自事后观察，必须重新预注册。 |
| Correlation != causation | CAUTION | 算法干预是受控的，但 latent proxy 与误差之间的因果中介尚未干预验证。 |
| Reverse causality | CAUTION | 轨迹具有时间顺序，但 Ω、latent 与风险共同受优化更新驱动。 |

## 结论与下一闸门

状态：`EXPLORATORY_SIGNAL`，不是 `PASS_LOCAL_SIGNAL`。

建议把下一轮范围收敛为：

1. 先查重“风险差 = gradient difference − Hessian correction”及其 DG/MTL 上界近邻；
2. 构造 IRMv1 scalar projection 失败的最小反例；
3. 预注册 combined gradient + Hessian penalty，并加入 joint/class-conditional MMD 对照；
4. 系统改变 target coverage：源域凸包内、有限外推、spurious sign flip；
5. 为跨任务部分显式定义 task discrepancy，不再用 domain penalty 代替 task term。

若第 1 步发现 exact equivalent work，则停止把恒等式本身作为贡献，转向更严格的 target coverage、task interaction 或 tightness 分析。
