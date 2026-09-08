# EXPL-002：从实验表现发现隐空间响应因子

## 定位

本实验遵循“先发现、后解释”的范式。它不把 Bayes projection、invariant/spurious、nested oracle 或 approximation/estimation/optimization/shift 预设为观测分解，而是先从算法相对 ERM 的隐空间响应中寻找稳定结构，再把已有理论作为后验解释候选。

本轮结果是探索性结果，不是 exact decomposition、因果归因、target certificate 或论文新颖性结论。

## 研究对象

对每个非 ERM 运行 `(method, lambda, seed)`，选取最终 checkpoint，并与相同 seed 的 source ERM 最终 checkpoint 配对。对每个观测量 `q` 构造：

\[
\Delta q_{j,\lambda,s}=q_{j,\lambda,s}-q_{\mathrm{ERM},s}.
\]

ERM 是固定跨算法参照；`lambda=0` 不是这里的主要参照。target signed gap 和 positive degradation 只用于聚类后的外部结果分析：

\[
G_T=R_T(f_{j,\lambda})-R_T(f_{\mathrm{ERM}}),
\qquad D_T=[G_T]_+.
\]

## 发现矩阵

PCA 和 Ward 聚类只使用以下 13 个 ERM-paired、target-free 观测：

- `source_mse`；
- `core_probe_r2`、`spurious_probe_r2`、`domain_probe_accuracy`；
- `latent_mean_gap`、`latent_covariance_gap`；
- `effective_rank`、`latent_variance`；
- `encoder_core_norm`、`encoder_spurious_norm`、`encoder_domain_norm`、`encoder_nuisance_norm`；
- `source_head_norm`。

以下字段明确不进入发现：`cross_domain_mse`、`cross_task_mse`、`joint_mse`、所有 target oracle、target gap、`own_regularizer` 和方法名。算法名只在发现完成后用于描述 cluster composition。

## 方法

1. 对发现矩阵按列标准化。
2. 用 SVD/PCA 提取最多四个主成分；符号由最大绝对载荷固定，仅用于复现显示。
3. 在完整标准化观测矩阵上做 Ward clustering，比较 `k=2,...,6` 的平均 silhouette，选择最大者，平局选较小 `k`。
4. 用最大绝对载荷生成临时描述名；名称只描述响应方向，不等同于信息损失、压缩、invariance 或 shortcut suppression。
5. 用 latent-only、source/probe-only、去掉 source risk、去掉 encoder norm 的特征子集，以及 leave-one-seed-out 计算 PCA score matching 和 cluster ARI。
6. 聚类完成后，才计算因子与 target signed gap / positive degradation 的 Spearman 关联，以及 cluster 的 target gap 汇总。

## 本轮结果

运行命令：

```bash
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/sunlay/miniconda3/envs/crypto-research/bin/python -m ood_repr_reg.discover_latent_factors \
  --trajectory artifacts/EXPL-001-latent-rerun/trajectory.csv \
  --output-dir artifacts/EXPL-001-latent-rerun/data_driven_factors
```

输入包含 140 个非 ERM 最终运行。PC1--PC4 的解释方差分别为 `0.366, 0.242, 0.141, 0.094`，前四个方向合计约 `0.843`。

由载荷后验产生的临时描述为：

- PC1：latent variance、encoder core norm、latent covariance gap 共同变化的整体幅度方向；
- PC2：encoder domain norm、latent mean gap、domain probe accuracy 共同变化的域位移/域可分性方向；
- PC3：core probe、source risk、spurious probe 共同变化的任务拟合/预测结构方向；
- PC4：spurious probe、effective rank、source risk 共同变化的选择性 latent loading 方向。

`k=3` silhouette 为 `0.308`，但 `k=2` 为 `0.286`、`k=4` 为 `0.279`，因此离散三簇不是强发现。三个簇的外部 target signed gap 均值分别为 `-0.040`、`+0.216`、`+0.011`；第二簇只有 9 个运行，主要表现为 source risk 上升、core probe 下降、effective rank 下降，应称为“强退化型响应”，不能直接宣称 collapse 机制。

表示坐标存在 gauge 问题：encoder norm 和 latent variance 会随 encoder/head 的互相缩放而改变，而最终预测可以不变。因此 PC1 中的“整体幅度”是本实验坐标系下的经验方向，不是坐标不变的信息量或压缩量；后续分析必须加入缩放对照，或改用预测风险、谱/子空间和 probe 风险等更稳定对象。

PC1 与 target signed gap 的 Spearman 为 `-0.633`，与 positive degradation 为 `-0.687`；PC2 分别为 `-0.304`、`-0.259`；PC3 接近零；PC4 为 `+0.332`、`+0.343`。这些是外部探索关联，不是独立样本显著性结论，也不构成上界。

## 稳定性与当前判断

因子方向相对稳定：完整视图与 latent-only、去 source risk、去 encoder norm 的平均绝对 factor-score correlation 分别为 `0.944`、`0.932`、`0.957`。但 cluster ARI 分别只有 `0.502`、`0.498`、`0.619`；leave-one-seed-out ARI 为 `0.338--0.928`。因此当前应保留连续因子候选，暂不把三簇当作固定机制 taxonomy。

当前最小可保留结论是：在这轮合成轨迹中，正则化相对 ERM 的隐空间响应可以由少数连续方向近似描述；这些方向中，整体表示幅度和域位移方向与 target gap 有探索性关联，而任务拟合方向在本轮对 target gap 的线性关联弱。强压缩/退化运行形成了一个小而可辨认的异常群体。

## 解释闸门

下一步理论 crosswalk 必须以这些数据因子为输入，逐项检查它们是否对应已有的 representation deficiency、transport、nested-oracle、latent subspace 或 outer error decomposition。若某个理论分量无法稳定映射到多个 held-out source/target shift 设置，不得把它升级为正式隐空间分解。

下一轮必须加入至少两个独立的 source-domain 留出协议或 shift regime；在不改变发现矩阵和聚类选择规则的情况下，检验 PC loading、cluster profile 和 target 外部关联是否复现。target 标签不得回流到因子发现、聚类或选参。

## 产物

- 可复现脚本：`src/ood_repr_reg/discover_latent_factors.py`；
- 因子分数：`artifacts/EXPL-001-latent-rerun/data_driven_factors/discovered_factor_scores.csv`；
- 因子载荷：`.../discovered_factor_loadings.csv`；
- cluster 运行与汇总：`.../discovered_clusters.csv`、`.../discovered_cluster_summary.csv`；
- cluster 标准化剖面：`.../discovered_cluster_profiles.csv`；
- 稳定性：`.../discovered_stability.csv`；
- 自动报告：`.../discovered_factor_report.md`。
