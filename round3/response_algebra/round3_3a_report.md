# 3A 报告：OOD Risk-Response Algebra

## 范围

本报告只分析 raw population risk-response matrix 的 lifting factorization、rank、quotient 和线性子空间几何。没有运行 clustering、机制命名、response-order 分解或 regularizer control。

## 1. 精确因子化

对 (X=(1,C,A)) 使用带 \(\sqrt{2}\) off-diagonal 权重的 Frobenius-isometric `svec`，定义 \(\phi(w)\) 和 \(\psi(T)\)。结果：

- \(N_{model}=660\)，\(N_{shift}=300\)，lifting feature dimension=`21`；
- \(\operatorname{rank}(\Phi)=21\)；
- \(\operatorname{rank}(\Psi)=17\)；
- \(\operatorname{rank}(\mathsf R)=17\)；
- factorization max residual=`2.665e-15`，Frobenius residual=`2.115e-13`；
- exact within tolerance: `True`。

因此当前数据上 \(\mathsf R=\Phi\Psi^\top\) 数值成立。该结论是 quadratic-risk algebra 的 exact equality，不是机制解释。

## 2. Rank 与 quotient

response rank 的瓶颈判断：`shift ensemble`。

- model-side blind dimension：`4`；
- shift-side blind dimension：`0`；
- shift-side risk-visible quotient dimension：`17`；
- response rank equals shift lifting rank：`True`；
- response rank equals model lifting rank：`False`。

这里的 blind dimension 只表示 lifting space 上的线性代数 kernel，不表示生成机制盲区。

## 3. 四个 shift-unexcited risk-statistic combinations

当前 shift lifting 的右 null space 为 4 维。下表使用 coordinate-anchored basis，列出的每一行都是一个线性泛函；对当前 300 个 shifts，其值均为零。系数坐标依次对应 \(\Delta M_{XX}\) 的 `svec`、\(\Delta m_{XY}\) 和 \(\Delta m_{Y^2}\)。这是可读代表，不是唯一 basis。

| anchor coordinate | null functional (largest coefficients) |
|---|---|
| `dM[0,0]` | `dM[0,0]=+1.0000` |
| `dmXY[2]` | `dmXY[0]=+2.2682; sqrt2*dM[0,1]=-1.6039; sqrt2*dM[0,2]=+1.1548; sqrt2*dM[1,2]=-1.0470; dmXY[2]=+1.0000; dM[2,2]=+0.7200; dM[1,1]=+0.6677` |
| `dmXY[3]` | `dmXY[3]=+1.0000; dM[1,1]=+0.8969; dmXY[1]=-0.8969; sqrt2*dM[1,3]=-0.7071; sqrt2*dM[2,3]=+0.5091; sqrt2*dM[1,2]=-0.4566; dmXY[0]=+0.1790` |
| `dmXY[4]` | `dmXY[0]=+1.5856; sqrt2*dM[0,1]=-1.1212; dmXY[4]=+1.0000; sqrt2*dM[0,2]=+0.8072; sqrt2*dM[1,4]=-0.7071; sqrt2*dM[2,4]=+0.5091; dM[1,1]=+0.0773` |

其中，`dM[0,0]` 这一行是严格的结构不变量：\(X_0=1\)，所以 \(\Delta M_{00}=\Delta\mathbb E[1^2]=0\) 对任何 environment 都成立。其余三行不是由 intercept 恒等式单独推出的普遍约束：在允许所有 structural parameters（包括 \(\beta\)）变化的随机有效 structural family 中，数值 span rank 为 `20`，只剩 `1` 维 null；当前 4 维 null 与它的交为 `1` 维。因此当前结果应解释为：`1` 个 universal structural invariant + `3` 个当前 finite shift design 未激活的 combinations，而不是 4 个都被 SCM 普遍禁止。

固定 task mechanism（\(\beta\) 不变）的 structural reachability 数值 rank 为 `14`；这说明是否把 task shifts 纳入声明 family 会改变 quotient dimension，属于 environment-family 选择，而不是同一个 rank 结论。

## 4. Pure structural-family 子空间

以下 probes 每次只改变一个 structural parameter family，并同时加入正负 finite shifts；这不是 3B 的阶数分解。`lifting rank` 在 21 维 risk-statistic coordinate space 中计算，`response rank` 在 660 维 model-response space 中计算。

| family | lifting rank | response rank |
|---|---:|---:|
| `core_mean` | 4 | 4 |
| `core_covariance` | 3 | 3 |
| `relation` | 6 | 6 |
| `b` | 4 | 4 |
| `mu_xi` | 4 | 4 |
| `sigma_xi` | 3 | 3 |
| `task` | 3 | 3 |

| pair | intersection dimension | relation | first in second residual | second in first residual |
|---|---:|---|---:|---:|
| `core_mean__core_covariance` | 2 | `partial-overlap` | 6.396e-01 | 4.606e-01 |
| `core_mean__relation` | 0 | `zero-intersection` | 8.320e-01 | 8.915e-01 |
| `core_mean__b` | 0 | `zero-intersection` | 9.004e-01 | 9.004e-01 |
| `core_mean__mu_xi` | 0 | `zero-intersection` | 9.004e-01 | 9.004e-01 |
| `core_mean__sigma_xi` | 0 | `zero-intersection` | 9.690e-01 | 9.584e-01 |
| `core_mean__task` | 0 | `zero-intersection` | 8.794e-01 | 8.353e-01 |
| `core_covariance__relation` | 0 | `zero-intersection` | 7.710e-01 | 8.929e-01 |
| `core_covariance__b` | 0 | `zero-intersection` | 9.335e-01 | 9.505e-01 |
| `core_covariance__mu_xi` | 0 | `zero-intersection` | 9.335e-01 | 9.505e-01 |
| `core_covariance__sigma_xi` | 0 | `zero-intersection` | 9.621e-01 | 9.621e-01 |
| `core_covariance__task` | 0 | `zero-intersection` | 8.304e-01 | 8.304e-01 |
| `relation__b` | 2 | `partial-overlap` | 6.411e-01 | 3.414e-01 |
| `relation__mu_xi` | 2 | `partial-overlap` | 6.411e-01 | 3.414e-01 |
| `relation__sigma_xi` | 2 | `partial-overlap` | 7.669e-01 | 4.199e-01 |
| `relation__task` | 0 | `zero-intersection` | 9.806e-01 | 9.608e-01 |
| `b__mu_xi` | 4 | `identical` | 1.291e-14 | 1.291e-14 |
| `b__sigma_xi` | 2 | `partial-overlap` | 6.533e-01 | 4.855e-01 |
| `b__task` | 0 | `zero-intersection` | 9.892e-01 | 9.856e-01 |
| `mu_xi__sigma_xi` | 2 | `partial-overlap` | 6.533e-01 | 4.855e-01 |
| `mu_xi__task` | 0 | `zero-intersection` | 9.892e-01 | 9.856e-01 |
| `sigma_xi__task` | 0 | `zero-intersection` | 1.000e+00 | 1.000e+00 |

按当前固定顺序累积 pure family 的 lifting incremental rank：`core_mean`: +4 (rank 0 -> 4); `core_covariance`: +1 (rank 4 -> 5); `relation`: +6 (rank 5 -> 11); `b`: +2 (rank 11 -> 13); `mu_xi`: +0 (rank 13 -> 13); `sigma_xi`: +1 (rank 13 -> 14); `task`: +3 (rank 14 -> 17)。

`b` 与 `mu_xi` 的两行在 lifting 与 response geometry 中均相同（intersection 等于各自 rank、两个 inclusion residual 为零），因为在线性观测方程中它们只通过 `b + mu_xi` 进入 A。其余 family 多数是部分重叠或零交，而不是两两正交 direct sum。当前 pure lifting 与 response 的 group/joint ranks 是否一致：`{'group_rank_mismatches': {'core_mean': False, 'core_covariance': False, 'relation': False, 'b': False, 'mu_xi': False, 'sigma_xi': False, 'task': False}, 'joint_rank_mismatch': False}`；主角本身仍依赖各自 ambient inner product。

## 5. 结论边界

当前可确认：

1. response matrix 存在由 model lifting 与 shift lifting 共同决定的精确双侧因子化；
2. response rank 给出 source model ensemble 实际能看到的 shift lifting quotient 维度；
3. 两侧 rank 差分别量化 model-side 与 shift-side algebraic blind dimensions。

当前不能确认：

- 低秩是否对应 core/nuisance/relation 等语义；`DEFER-TO-3C`；
- 一阶/二阶或更高阶响应如何组成 filtration；`DEFER-TO-3B`；
- 任意正则项是否控制某个 quotient factor；延期，不在 3A 结论中讨论。

详细结果见 `results/`。
