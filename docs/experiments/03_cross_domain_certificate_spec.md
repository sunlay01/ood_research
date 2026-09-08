# Cross-domain error certificate specification

> 状态：`SPECIFICATION_ONLY / NO_FORMAL_EXPERIMENT_AUTHORIZED`

## 1. 目标

本阶段审计现有 OOD/DG 算法的训练指标，而不是提出新算法。对算法 `j` 的最终模型 `f_j`，固定 source ERM 模型 `f_ERM` 为比较基准，并定义：

```text
D_T,j = max(R_T(f_j) - R_T(f_ERM), 0)
```

`D_T,j` 是 target evaluation label，不参与训练、正则校准或模型选择。

## 2. 证书输入层级

### Omega-only

```text
{ Omega_j, sqrt(Omega_j) }
```

检验原始算法指标单独是否具有跨域退化解释力。已有塌缩反例要求不能默认存在通用上界。

### Omega plus source risk

```text
{ Omega_j, sqrt(Omega_j), R_S(f_j),
  max(R_S(f_j) - R_S(f_ERM), 0) }
```

检验加入 source fit cost 后是否减少 `Omega-only` 的失败。

### Source-only observable

在上述输入上加入 source-domain latent mean/covariance spread、head sensitivity 或复杂度量。它们是可观测诊断量，不自动等于 decomposition 中的理论项。

`target_coverage_residual` 只允许作为理论 remainder 或事后审计字段，不能进入 source-only view。

当前 exploratory sweep 的字段映射必须按 proxy 解释：`latent_mean_gap` 和 `latent_covariance_gap` 来自 source domains，只是 source coverage/representation-spread proxy，不是 target shift；`source_head_norm` 是 complexity proxy；`diagnostic_{method}` 被暂放入 `head_mismatch` 位置只是 method-diagnostic proxy，不是已经定义或估计的 head sensitivity。它们不能在没有额外定理的情况下被相加或称为 upper-bound terms。

## 3. 误差分解接口

对固定表示 `phi`，定义 hybrid representation distribution：

```text
Q_phi(z, y) = P_T^phi(z) P_S(y | z)
```

然后通过加减 `R_Q(f)` 得到：

```text
R_T(f) - R_S(f)
= [R_T(f) - R_Q(f)] + [R_Q(f) - R_S(f)]
```

第一项表示 conditional response/label mismatch，第二项表示 representation marginal/covariate shift。它是 exact telescoping identity；把两项换成 source-observable upper bounds 需要额外假设。

必要时再用 nested-oracle 定义 representation insufficiency 与 classifier/head mismatch，但不得把 oracle gap 误写成 source-only observable。

## 4. 算法与指标

保留现有实现作为审计对象：ERM、CORAL、MMD、IRMv1、gradient alignment、Hessian alignment、L1 和 L2。报告：

```text
lambda -> (Omega_j, certificate inputs, D_T,j)
```

现有 `EXPL-001` 只作为探索性证据；不改变其 `EXPLORATORY_SIGNAL` 状态。

## 5. 允许的验证设计

只有 prior-art gate 产生 `PROBE_AUTHORIZED` 后才运行：

1. leave-one-source-domain-out，把一个源域作为 pseudo-target；
2. source coverage 内、有限外推和明显外推三类 shift；
3. 比较三种 certificate input view 与真实 `D_T,j`；
4. 使用 CORAL/MMD 作为 marginal-alignment negative controls；
5. 报告 bound violation、coverage 条件和替代解释，不只报告 accuracy。

## 6. 通过与停止标准

通过条件是：存在可证明的 exact/conditional 关系、明确的 `Omega-only` 失败边界、source-only 的 held-out-domain 可反驳预测，以及与最近邻工作的机制级差异。

若只能得到事后 correlation、需要 target tuning，或六轴比较显示已有工作覆盖完整主张，则停止论文主线，将结果保留为算法审计工具。
