# EXPL-001：正则项—隐空间—跨域/跨任务误差探索

> 类型：`EXPLORATORY`。本实验没有通过论文级新颖性闸门，结果只允许生成猜想、反例和后续预注册，不得回写成验证性证据。

## 目的

在统一的低维可解释模型中比较 ERM、L1、L2、IRMv1、MMD、CORAL、gradient alignment 与 Hessian alignment，观察算法自身正则项下降时：

1. 隐空间中的 core、spurious 和 domain-only 信息如何变化；
2. 跨域误差、跨任务误差以及域与任务同时变化时的误差如何变化；
3. 哪些经验关系值得进一步写成 exact identity、conditional bound 或反例。

本实验不检验“哪个算法总体最好”，也不使用目标域指标选择超参数。

## 生成模型

- 两个源域和一个未参与训练的目标域。
- 两个源任务共享线性 encoder，各有一个线性 head。
- 一个未参与 encoder 训练的目标任务；在源域 support set 上为冻结表示拟合 ridge head。
- 输入由两维 core、一维与标签相关但随域翻转的 spurious 特征、一维 domain mean 特征和一维噪声组成。
- 目标域同时改变 spurious-label 相关性与 domain mean。

## 比较方法

| 方法 | 训练正则项 |
| --- | --- |
| ERM | 无 |
| L1 | encoder 与 source heads 的平均绝对值 |
| L2 | encoder 与 source heads 的平均平方值 |
| IRMv1 | 各 task-environment 风险对统一 logit scale 的梯度平方 |
| MMD | 同一任务在两个源域的 latent marginal 多核 RBF MMD |
| CORAL | 同一任务在两个源域的 latent covariance 差异 |
| gradient alignment | 同一任务的两个环境对 task head 的风险梯度差异 |
| Hessian alignment | 同一任务的两个环境对 task head 的风险 Hessian 差异 |

为降低不同正则数值尺度造成的不公平，每次运行在同一初始化处，以“正则梯度范数与 ERM 梯度范数之比”校准正则乘子；原始正则值与校准乘子都保留。

## 自变量、因变量与控制

- 自变量：正则方法、正则强度、训练步数、随机种子。
- 主要因变量：source MSE、cross-domain MSE、cross-task MSE、joint domain-task MSE。
- 机制诊断：core/spurious 线性 probe `R²`、domain probe accuracy、latent mean/covariance gap、effective rank、latent variance、encoder 分块范数。
- 固定项：数据生成机制、样本规模、初始化规则、优化器、学习率、训练步数、评估代码与 ridge probe。

## 探索性判据

以下阈值只用于筛选后续猜想，不构成显著性检验：

- `candidate signal`：正则自身至少下降 50%，并且某个 latent 诊断量在多数种子和至少两个非零正则强度下同向变化；该诊断量与对应目标误差的 Spearman 相关绝对值至少为 0.5。
- `negative signal`：正则明显下降，但 latent 诊断量和目标误差没有稳定方向，或相同正则值对应相反误差变化。
- `confounded`：目标误差变化主要伴随 source error 大幅恶化、effective rank 接近 1，或 core 与 spurious 信息同时丢失。
- `invalid`：出现非有限损失、正则未被优化、数据/评估泄漏或方法间预算不一致。

## 预算与命令

- smoke：1 个种子、每种正则 1 个强度、10 步，目标为 5 分钟内完成。
- main：5 个种子、4 个非零强度、300 步，目标为 45 分钟内完成。
- 原始输出：`artifacts/EXPL-001*/`，不提交 Git。

```bash
bash scripts/run_regularizer_smoke.sh
bash scripts/run_regularizer_sweep.sh
```

## 解释限制

1. 合成数据只用于暴露机制，不能证明真实数据上也成立。
2. MMD、CORAL 等存在多个实现版本，本实验只代表表中精确定义。
3. 跨任务误差使用冻结 encoder 后拟合新 head，研究的是 representation transfer，不是 zero-shot task generalization。
4. 目标域只用于离线评估，不参与训练、正则校准或模型选择。
5. 从本实验观察到的关系必须经过反例搜索、文献查重和独立预注册后，才可进入数学证明或正式验证。
