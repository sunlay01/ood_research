# EXPL-001：算法相对 ERM 的隐空间误差审计

## 状态

`EXPLORATORY / CANDIDATE_STRUCTURE`。本记录只报告有限样本探索结果，不把
latent oracle 诊断称为 population theorem、因果归因或 target certificate。

本轮使用 conda 环境 `crypto_research`，配置为
`configs/regularizer_sweep_main.json`。共有 5 个 seed、8 种方法、4 个非零
lambda，共 145 个最终运行和 1,015 条 checkpoint 记录。独立产物位于
`artifacts/EXPL-001-latent-rerun/`。

## 比较对象与诊断探针

主比较对象始终是算法 `j` 相对固定 source ERM 的目标域风险差：

\[
\Delta R_T^{j,\mathrm{ERM}}
=R_T(f_j)-R_T(f_{\mathrm{ERM}}).
\]

对每个已训练表示 `Z_j=phi_j(X)`，另外在固定评估数据上做两折 cross-fitted
linear ridge probe，得到隐空间最优线性 head 的诊断风险 `L_T(Z_j)`。令当前
训练 head 的目标风险为 `R_T(f_j)`，则：

\[
H_T(f_j)=R_T(f_j)-L_T(Z_j).
\]

因此在相同 target 评估集、相同 cross-fitting construction 下，直接加减 ERM
项得到：

\[
\Delta R_T^{j,\mathrm{ERM}}
=\underbrace{[L_T(Z_j)-L_T(Z_{\mathrm{ERM}})]}_{\text{latent-oracle change}}
+\underbrace{[H_T(f_j)-H_T(f_{\mathrm{ERM}})]}_{\text{head-mismatch change}}.
\]

这是选定有限样本 probe 后的 `exact accounting identity`，不是总体条件期望
恒等式。oracle 的作用是测量“相对 ERM 的表示可达风险变化”和“相对 ERM 的
head 失配变化”，不是把研究对象从 ERM 换成 oracle。

`raw oracle` 仅作为辅助 reference，用于观察 `X` 到 `Z` 的可恢复性损失；它
不能被解释为真实 Bayes representation insufficiency。当前 ridge probe 只检验
线性 head 可恢复性，并且 target label 只用于离线评估。

## 代数闭合检查

输出文件 `erm_paired_latent_decomposition.csv` 中，target accounting residual
的最大绝对值为 `4.16e-16`，source residual 的最大绝对值为 `1.68e-16`。
这确认实现的比较量确实是算法相对 ERM 的风险差，而不是把几个未对齐的误差
指标事后相加。

## 主要结果：lambda = 1.0

下表为同 seed 配对 ERM 后的均值。`latent delta` 和 `head delta` 两列相加即
`target gap`。

| 方法 | target gap | latent delta | head delta | source excess |
| --- | ---: | ---: | ---: | ---: |
| CORAL | +0.0214 | +0.0497 | -0.0284 | +0.0004 |
| gradient alignment | -0.1221 | +0.1822 | -0.3043 | +0.0027 |
| Hessian alignment | +0.0173 | +0.0533 | -0.0359 | +0.0004 |
| IRMv1 | -0.1221 | +0.0288 | -0.1509 | +0.0014 |
| L1 | +0.3943 | -0.2090 | +0.6033 | +0.2262 |
| L2 | +0.0713 | +0.3482 | -0.2769 | +0.1328 |
| MMD | -0.0177 | -0.0836 | +0.0660 | +0.0012 |

## 当前可保留的经验观察

1. **gradient alignment 和 IRMv1 的改善主要来自 head mismatch 下降。**
   在本合成机制和本训练设置下，target latent-oracle risk 并没有下降：
   gradient alignment 的 latent delta 为 `+0.1822`，IRMv1 为 `+0.0288`；但
   head delta 分别为 `-0.3043` 和 `-0.1509`。因此不能把 target 改善直接写成
   “正则学到了更好的隐空间”。更准确的候选解释是：正则改变了优化所得的
   表示/head 配置，其中当前 head 的跨域使用方式改善占主要部分。

2. **L1 暴露了表示可达性与当前 head 适配的分离。** target latent-oracle
   delta 为 `-0.2090`，但 head delta 为 `+0.6033`，最终 target gap 为
   `+0.3943`。也就是说，仅观察一个隐空间 probe 可能会把严重的 head 失配
   漏掉；这是表示压缩类方法的明确负对照。

3. **L2 更接近表示层退化。** latent delta 为 `+0.3482`，head delta 为
   `-0.2769`，二者部分抵消后仍有 `+0.0713` 的 target degradation。它与 L1
   的响应不同，说明“正则化强度”不能直接映射成单一的 representation quality
   轴。

4. **CORAL/Hessian alignment 的几何指标下降不等于 target gap 下降。** 两者
   在 lambda=1 时 target gap 分别为 `+0.0214` 和 `+0.0173`，而 latent
   delta 为正。当前实验只能把这作为 negative-control signal：边际/二阶几何
   对齐与任务相关的隐空间可达风险不是同一个量。

5. **MMD 在最大强度下的改善由 latent-oracle delta 主导，但不是普遍规律。**
   lambda=1 时 latent delta 为 `-0.0836`、head delta 为 `+0.0660`，净 gap
   为 `-0.0177`。这说明即使表示层 probe 变好，训练 head 也可能变差，仍须
   保留两条轴。

## 跨所有强度的审慎统计

在 140 个非 ERM 的 method-lambda-seed 配对点上：

- target gap 与 latent delta 的 Spearman `rho=-0.014`；
- target gap 与 head delta 的 Spearman `rho=0.221`；
- target gap 与 source excess 的 Spearman `rho=0.306`。

这些点共享数据生成机制、seed 和训练轨迹，不能当作独立样本或显著性证据。
它们支持的最小结论是：单一 latent-oracle 变化在本轮没有稳定解释 target gap，
而 head mismatch 至少是不可省略的第二轴。

## 当前候选结构与边界

当前值得继续验证的不是“oracle 是否是证书”，而是下面这个相对 ERM 的双轴
审计接口：

```text
algorithm-vs-ERM target degradation
    = latent reachable-risk change
    + fitted-head mismatch change
```

其中第一项和第二项目前都是 cross-fitted linear-probe diagnostics。它们还不
等于 representation insufficiency、conditional shift 或 optimization error 的
population 分解。尤其 `target_latent_oracle_mse - source_latent_oracle_mse`
仍可能混合 covariate shift、conditional label shift 和有限样本 probe 误差。

下一步应固定训练算法和模型，只改变 target evaluation 的 mechanism shift 与
marginal/domain shift，检查上述两项能否在不同 shift 类型下稳定地区分响应。
在此之前不应把它们写成上界，也不应据此宣称新颖性。

## 可复核产物

- 原始轨迹：`artifacts/EXPL-001-latent-rerun/trajectory.csv`
- 相对 ERM 的隐空间分解：`artifacts/EXPL-001-latent-rerun/erm_paired_latent_decomposition.csv`
- 汇总：`artifacts/EXPL-001-latent-rerun/final_summary.csv`
- 配对效果：`artifacts/EXPL-001-latent-rerun/paired_effects.csv`
- 相关性：`artifacts/EXPL-001-latent-rerun/final_correlations.csv`
- 运行环境：`artifacts/EXPL-001-latent-rerun/environment.json`
