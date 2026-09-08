# 隐空间分解候选图谱与跨分解 crosswalk

> 版本：`v0.1 / STAGE_1_REPRESENTATION_TAXONOMY`。本文不是新颖性结论，也不把不同 construction 的分量相加。目的只有一个：在固定 ERM 参照下，明确“隐空间优化到底改变了什么”，再决定哪些量值得进入正式实验或风险上界。

> **范式修正。** 下列 A--H 不是实验输入的预设分解，而是用于解释数据驱动因子的候选语义库。正式分析必须先从 ERM-paired 的 source/latent 轨迹中发现稳定的连续方向或响应簇，再进行后验 mapping；target error 只能作为发现后的外部结果，不能参与因子发现、聚类或命名。

## 1. 研究对象先固定

对环境 `e`、输入 `X`、标签 `Y`，令

\[
Z_{j,\lambda}=\phi_{j,\lambda}(X),
\qquad
f_{j,\lambda}=h_{j,\lambda}\circ\phi_{j,\lambda}.
\]

其中 `j` 是现有算法，`lambda` 是正则强度。固定 source ERM 结果

\[
f_{\mathrm{ERM}}=f_{\mathrm{ERM},0}
\]

作为跨算法参照。对任意 decomposition construction `d`，只定义同一 construction 内的相对响应：

\[
\Delta^{(d)}_{j,k,e}(\lambda)
 =E^{(d)}_{k,e}(f_{j,\lambda})
  -E^{(d)}_{k,e}(f_{\mathrm{ERM}}).
\]

若要研究正则路径本身，再报告

\[
S^{(d)}_{j,k,e}(\lambda)
 =E^{(d)}_{k,e}(f_{j,\lambda})
  -E^{(d)}_{k,e}(f_{j,0}).
\]

`ERM` 是跨算法的固定 anchor，`lambda=0` 是同一算法的路径 anchor；二者不能混写。target risk gap 另行记为

\[
G_{T,j}(\lambda)=R_T(f_{j,\lambda})-R_T(f_{\mathrm{ERM}}),
\qquad
D_{T,j}(\lambda)=[G_{T,j}(\lambda)]_+.
\]

## 2. 八类隐空间分解/构造

### A. 条件期望与 Bayes projection：信息损失 + head mismatch

**隐空间对象。** 嵌套 sigma-field
`sigma(Z) subseteq sigma(X)`，或者等价的 `L2` 条件期望投影。

**平方损失。**

\[
R_e(h\circ\phi)=N_e+I_e(\phi)+H_e(h,\phi),
\]
\[
N_e=\mathbb E_e[(Y-m_e(X))^2],
\quad
I_e=\mathbb E_e[(m_e(X)-m_e^\phi(Z))^2],
\quad
H_e=\mathbb E_e[(m_e^\phi(Z)-h(Z))^2].
\]

这里 `I` 是表示没有保留的任务信息，`H` 是在当前表示上 head 没有达到 representation oracle 的部分。该式是 `exact equality`，但分量是 population/oracle 量。

**proper loss 推广。** 用 conditional Bregman divergence 定义 `I_e^ell` 与 `H_e^ell`，得到 Bayes risk + information loss + head regret 的 `exact equality`。log loss 下 `I_e^ell = I_e(Y;X|Z)`。

**能回答什么。** 正则是否压低了任务信息损失、是否只是改变了 head 可达性、以及表示 collapse 与 head mismatch 是否混淆。

**不能回答什么。** 它不自动说明信息是 invariant、causal 还是 spurious，也不包含 source-to-target transport。

**主要来源。** 本项目 `docs/theory/01_dual_track_decomposition.md`；proper-loss 与 feature-information 语言分别对应 Reid--Williamson、van Rooyen--Williamson。结论标签：`exact equality`。

### B. representation-conditioned risk transport：conditional / marginal / support

**隐空间对象。** `P_e^Z` 与 `P_e(Y|Z)`，用 representation hybrid
\[
Q_\phi(z,y)=P_T^Z(z)P_S(y\mid z)
\]
连接 source 与 target。

\[
R_T(f)-R_S(f)
 =\underbrace{R_T(f)-R_{Q_\phi}(f)}_{\text{conditional response shift}}
 +\underbrace{R_{Q_\phi}(f)-R_S(f)}_{\text{latent marginal shift}}.
\]

在 shared support 上，第二项还可用 Radon--Nikodym/Lebesgue decomposition 拆为 density reweighting 与 singular coverage residual。对任一既有分量 integrand 还可以写成

\[
\mathbb E_Ta_T-\mathbb E_Sa_S
 =\int r(a_T-a_S)dP_S
 +\int(r-1)a_SdP_S
 +\int a_TdP_T^\perp.
\]

这是 `exact equality`；将它变成数值 upper bound 是 `conditional theorem/bound`，需要 density-ratio、conditional regularity 和 coverage 假设。

**能回答什么。** 一个隐空间方法改善的是 `P^Z` 的 reweighting、`P(Y|Z)` 的变化，还是仅在 source support 内有效。

**不能回答什么。** `CORAL/MMD` 下降只直接触及某种 marginal moment/discrepancy，不等于 conditional shift 或 target risk 下降。

**主要来源。** Wu et al. (2020)；Shui et al. (2022) 提供带 smoothness/coverage 条件的相关 risk bound。结论标签：`exact equality` + `conditional theorem/bound`。

### C. nested-oracle failure decomposition：四类可操作失败

Galstyan et al. 的 construction 通过连续替换 oracle classifier，把测试误差写成

\[
e=e_0+e_1+e_2+e_3,
\]

其中依次对应：

- `e0`：training-set underfitting；
- `e1`：test-set inseparability；
- `e2`：training-test misalignment；
- `e3`：classifier non-invariance。

这是 `finite/oracle accounting`，不是四个天然正交的 population causes。增量项在某些设置下可能为负，不能无条件解释成非负 error shares。

**隐空间价值。** 它直接提供“固定 representation 后只重新拟合 head”的反事实，因此最适合检查一个正则是否主要伤害表示、还是只把 classifier/head 推离了测试可用解。

**与 A 的边界。** `e3` 与 A 的 `H` 有语义相邻性，但不是同一个量：前者依赖训练/测试 domain 集合和 nested classifier oracle，后者依赖单环境的 conditional-mean projection。没有已证明的 crosswalk，不能令 `e3=H`。

**主要来源。** Galstyan et al. (CVPR 2022)。结论标签：`exact telescoping construction`，分量解释属于 `conditional interpretation`。

### D. latent direct-sum / factor decomposition：invariant、spurious-invariant、variant

一些工作把表示空间写成结构性子空间的直和，例如

\[
Z=Z^{\Gamma}\oplus Z^{\Phi}\oplus Z^{\Xi},
\]

分别表示 domain-invariant features、spurious-but-invariant features 与 domain-variant features。

这是**隐空间结构分解**，不是 risk decomposition。只有在额外假设下，例如正交子空间、head 可加、噪声正交和环境机制固定，才能把各子空间的风险贡献写成可加形式；一般的 latent rotation 会改变坐标或子空间解释。

**能回答什么。** 正则在保留 stable predictive signal、保留 spurious invariant signal，还是压制环境变化 signal。

**必须补的识别条件。** 需要定义“invariant”的统计对象（边际、条件或机制），规定子空间如何对齐，并处理同一预测功能的 latent rotation。没有这些条件，`domain-predictive subspace` 不等于 `spurious subspace`。

**主要来源。** Wang et al. (JMLR 2026) 的 tri-space latent representation。结论标签：`structural decomposition / conditional bound`，不是本项目当前的 exact risk identity。

### E. latent transport regularity：conditional invariance、smoothness、coverage

这一类不把 `Z` 拆成几个坐标，而是拆解从 source 到 target 的**传输行为**：

- feature-conditional invariance error `kappa`；
- representation smoothness，例如 Dobrushin coefficient `alpha_TV(Phi)` 或 Lipschitz/Jacobian proxy；
- raw-space/domain shift `epsilon`；
- 未覆盖区域的 residual。

典型形式为

\[
\mathrm{BER}_T
\leq
\frac1T\sum_t\mathrm{BER}_{S_t}
 +\kappa+\alpha_{TV}(\Phi)\epsilon,
\]

这是 `conditional bound`，不是 exact decomposition。Shui et al. 同时说明常数/塌缩表示可能让 smoothness 项很小而任务风险很大，所以必须和 A 的 information loss 联合测量。

**能回答什么。** 正则是否让表示具有更稳定的 transport behavior，而不是只让 source latent moments 更接近。

### F. latent hypothesis-class learning decomposition：approximation / estimation / optimization / shift

这是用户提到的经典四层框架在表示学习中的**外层脚手架**。写成

\[
\text{observed target error}
 = \text{representation/class approximation}
 +\text{finite-sample estimation}
 +\text{optimization/algorithmic gap}
 +\text{source-to-target transport gap},
\]

必须通过一组嵌套 hypothesis-class oracles 才能把每一项具体化。它不是唯一的隐空间分解，也不能直接把 `approximation` 认作 A 的 `I`：A 是给定真实 `X -> Z` 后的 information loss，F 的 approximation 还包含模型类限制、head 类限制和参数化限制。

**与本项目的正确用法。** F 用来标记实验中的训练误差、probe estimation error、优化残差和 domain transport；A--E 用来说明表示层机制。不能把 F 的四项与 A--E 的分量逐项相加。

**结论标签。** 依赖具体 oracle 设计的 `exact telescoping identity` 或 `bound`；泛化叙述本身只是 `scaffold`。

### G. statistical experiment / feature deficiency：隐空间的决策信息损失

把 `X -> Z` 看作统计实验的 garbling，比较原始观测实验和表示实验在决策问题上的最优风险：

\[
\mathcal V_Z(\ell)-\mathcal V_X(\ell)
\geq 0
\]

在固定损失、决策类和分布下是 feature-induced decision regret；对所有决策问题取 supremum 可进入 Le Cam deficiency/Blackwell comparison。它给 A 的 `I` 一个更一般的统计决策解释，但不能自动分离 domain shift、head training 或 support loss。

**结论标签。** 固定损失下可为 `exact equality/decision-risk identity`；跨所有决策问题的 deficiency 关系是 `bound/order comparison`。主要来源：van Rooyen--Williamson (2014)。

### H. objective-to-mode / spectral path decomposition：mode loading、shrinkage、collapse

这类 construction 关注正则路径上隐空间或预测 mode 的振幅，而不是直接拆 risk。Landau-style 局部模型写成

\[
F_{eff}(q;\lambda)
 =L_0+\frac{r(\lambda)}2q^2+\frac{u(\lambda)}4q^4+O(q^6),
\]

其中 `q` 是 identifiable low-dimensional predictive mode 的 loading。可以区分 onset、selective retention、continuous shrinkage、instability 和 collapse。

**能回答什么。** 为什么同样叫“invariance regularization”的目标会有不同的路径形态；正则是在选择 mode、改变曲率，还是造成全局压缩。

**不能回答什么。** mode loading 下降不是任务信息损失下降，谱秩下降也不是 OOD 风险改善。需要 A、B 或 C 的同步诊断。

**主要来源。** Wang et al. (arXiv:2608.09396, 2026 preprint)；本项目只把它登记为高风险近邻和路径诊断范式。结论标签：`conditional local/asymptotic result`。

## 3. 需要额外验证的隐空间候选诊断

以下量可以作为最小实验 probe，但目前不是已证明的 canonical decomposition：

1. **task-predictive subspace**：由跨 source domain 的 (c_e=\mathbb E_e[ZY]) 或分类的 label-gradient span 定义；报告稳定 span 与 domain-varying span。需处理 rank、rotation 和 scale。
2. **domain-predictive subspace**：由 domain mean/covariance operators（例如 (\mathrm{Cov}_e(\mathbb E[Z\mid e]))）定义；它只能说明 domain information，不能单独命名为 shortcut。
3. **conditional residual subspace**：在拟合 task-predictive component 后分析 `Z` 对 `Y` 的 residual predictability；它连接 A 的 `I/H`，但有限样本 probe 误差必须单列。
4. **support/coverage coordinate**：在 latent space 上报告 source-to-held-out density ratio、nearest-neighbor coverage 或 conformal/mass residual；它连接 B/E 的 coverage，而不是 source-only certificate。
5. **head sensitivity subspace**：测量固定 `Z` 后 head refit、head perturbation 和 derivative response；它连接 C 的 `e3` 与 A 的 `H`，但不把二者相等。

这五个 probe 的共同限制是 latent coordinate 不可识别。因此优先使用投影、谱、预测风险、operator norm 和子空间 principal angles；不要直接比较原始神经元坐标或未经 gauge fixing 的参数范数。

## 4. 语义 crosswalk：哪些可以比，哪些不能比

| 共同语义 factor | 可连接的 constructions | 当前可声明的强度 | 不能做的事 |
| --- | --- | --- | --- |
| task information retained | A `I` 的互补量；G 的 feature decision risk；D 的 task-predictive subspace | 在固定损失/固定决策类下可作 conditional crosswalk | 把任意 linear probe accuracy 等同于 Bayes information |
| fitted-head/response mismatch | A `H`；C 的 `e3`；head sensitivity probe | 仅在共同 head class、共同 oracle protocol 下比较符号 | 直接令 `H=e3` 或跨 construction 相加 |
| conditional response shift | B 的 (P_T(Y\mid Z)) shift；E 的 κ | 若使用同一 latent state 和同一 discrepancy，可给 exact transport identity；bound 需假设 | 用 marginal alignment 代替 conditional invariance |
| marginal/support transport | B 的 density/singular terms；E 的 ε/coverage；C 的部分 misalignment | 可比较 transport 语义，具体数值通常不同 | 把 coverage residual 当 source-only observable |
| invariant/spurious/variant loading | D 的 subspaces；H 的 modes；moment/derivative penalties | 结构诊断或 local response | 把 mode/subspace 名称当作因果真值或风险分量 |
| approximation/estimation/optimization | F 的 outer oracle terms | 只能标注学习过程误差 | 与 A--E 的 terms 逐项相加 |

因此当前唯一安全的跨分解对象是“预先声明的 semantic factor 的 signed response”，而不是把所有 (E_k) 归一化后做总分：

\[
\operatorname{sign}\Delta^{(d)}_{j,k}(\lambda)
\quad\text{only if }(d,k)\text{ maps to a registered semantic factor }q.
\]

若没有 mapping、尺度或共同 oracle，结果写成 `UNDEFINED`，而不是强行比较。

## 5. 现有算法的初始映射

| 算法 | 原始目标主要触及 | 隐空间候选响应 | 必须保留的替代解释 |
| --- | --- | --- | --- |
| ERM | source predictive fit | 作为所有 Δ 的固定 anchor | source fit、head fitting、coverage 均不由 ERM 自动解决 |
| CORAL / MMD | B 的 marginal/moment transport | latent covariance/mean discrepancy、D 的几何变化 | conditional shift、support loss、head mismatch 可能不变或变坏 |
| IRMv1 | B/E 的 conditional/head response 的局部投影 | C 的 classifier non-invariance、A 的 head response、mode selection | scalar projection 缺口、信息损失、非唯一表示和 collapse |
| gradient alignment | head gradient / moment response | derivative discrepancy、C 的 (e_3) 相关量、B 的 conditional proxy | 只控制观测方向不等于完整 head neighborhood 控制 |
| Hessian alignment | curvature / sensitivity transport | head sensitivity、E 的 smoothness-like proxy、H 的 mode curvature | strong-convexity/same-minimizer 条件、representation information loss |
| L1 | F/H 的 compression 与 sparse mode selection | rank/loading/collapse、A 的 `I` 和 C 的 underfit 可能变化 | probe 变好不等于 fitted head 变好，参数 gauge 影响原始 penalty |
| L2 | parameter smoothness/compression | singular spectrum、reachable latent risk、head sensitivity | 统一 shrinkage、source fit excess、target support 未覆盖 |

这些是 `mechanism labels + testable hypotheses`，不是因果结论。每个算法仍需记录

\[
\lambda\mapsto
(\Omega_j,\Delta_A,\Delta_B,\Delta_C,\Delta_D,\Delta_E,\Delta_F,\Delta_G,\Delta_H,G_T).
\]

第一轮不必同时实现全部八类；优先选择 A、C、D、E 四条互补轴：A 分离 information/head，C 分离 nested failures，D 测试 latent factor loading，E 检查 transport regularity。

## 6. 对当前实验的直接改写

现有 latent-oracle 结果不应被称为“最终分解”。它只覆盖 A 的有限样本 proxy，并且当前报告的

\[
R_T(f_j)-R_T(f_{ERM})
=[L_T(Z_j)-L_T(Z_{ERM})]
 +[H_T(f_j)-H_T(f_{ERM})]
\]

是 A 中 representation oracle/head mismatch 的两轴记账。下一版实验报告应在同一模型和 ERM 配对下加入：

- C：nested-oracle 的 (e_0,e_1,e_2,e_3) 或可复现等价 probe；
- D：task/domain/residual subspace 的 rotation-invariant loading 与 principal angles；
- E：conditional invariance、latent smoothness 和 held-out support/coverage；
- F：训练误差、probe estimation error、优化残差单独列出，而不是混入 A 的 `I/H`。

然后对每一项报告 source、held-out source/pseudo-target 和真实 target 的 response，并始终与 ERM 比较。只有某个分解轴在不同 held-out domains、不同 shift regime 下重复区分算法机制，才值得进入 formal upper-bound 或 prior-art 再审查。

## 7. 当前结论与停止条件

### EVIDENCE

- 已有工作覆盖多种表示层风险、nested-oracle、smoothness/coverage、latent subspace 和 objective-path construction；“再提出一个单一隐空间分解”不是安全空白。
- 这些 construction 的对象不同，不能由文献标题或共同使用 `representation` 一词自动拼成统一 error identity。
- 当前实验的 latent oracle/head 结果支持保留双轴，但不足以说明所有隐空间机制。

### INFERENCE

当前最有信息量的下一步不是把 Ω 变成证书，而是用 ERM 配对的多分解 crosswalk 观察：同一正则是否在 A/C/D/E 上给出一致响应，或出现可解释冲突。冲突本身只有在预先声明的 common semantic factor 和替代解释下才有意义。

### RECOMMENDATION

先实现 A+C+D+E 的最小诊断矩阵，再决定是否补 F/G/H。若多分解结果只是复述 Galstyan/Wu 或与 P-008/P-009 等近邻同义，停止“分解框架”论文路线；若出现稳定、可反驳且不依赖 target 调参的 crosswalk 冲突，再进入严格上界/不可辨识性分析。
