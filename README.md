# OOD Regularization Through Semantic Latent Decomposition

本仓库研究的不是“哪一个 DG 算法在 benchmark 上分数更高”，也不是把一个小的正则值直接叫作 OOD guarantee。核心问题是：

\[
\boxed{
\text{正则项实际控制什么}
\;\longrightarrow\;
\text{因此能适应什么跨域变化}
\;\longrightarrow\;
\text{又遗漏什么危险变化}
\;\longrightarrow\;
\text{这些部分如何共同决定目标域误差。}
}
\]

第一阶段只做 **cross-domain**，不做 cross-task。上界、certificate 和 worst-case robustness 是这一问题在 blind residual 被消除、界定或估计时的强特例，不是研究中心。

## 当前状态

已完成的是一个可审计的线性基座，而不是论文结论：

- `C001`：部分可观测 Gaussian SCM 的 exact risk-transport identity 与两类预先声明的干预球上的 worst-case formula；
- `C004a`：source variation 不能观察某个 nuisance direction 的严格构造；
- `C002-IRM`：standard scalar-scale IRMv1 在该模型中存在 source-optimal、zero-penalty 但对允许 sign-flip 有大目标风险的 blind direction。该现象与 Kamath et al. (2021) 高度接近，只作为负对照；
- `C010`：在中心化标量 relation family，三条不同 source relation 使 IRMv1 的二次 radial-response design 满秩，从而将 penalty 条件地上界到有效 nuisance coefficient；两条 relation 可以留下 C002 的盲支。target transport 的总体界仍保留 source fit、观测残差与 target geometry；
- Anchor Regression/DRIG 与当前自由 conditional-relation/moment balls 尚未严格匹配，不能冒称为本框架的正例。

`LATENT-001` 已在第二次 `CODE_GATE` 被独立 Supervisor `VETO`，因此未获授权运行十 seed 主实验。直接阻断原因是冻结合同要求 source-only lambda selection，但 runner 只枚举强度网格，没有实现和测试可审计的选择路径。MVP 仍提供一个有效负结果：五个 projector 在代数上正交且风险记账闭合，但合法投影顺序的最大偏差约为 `0.4116`，超过 `0.25` 阈值，所以当前 SCM 下应判为 `SEMANTIC_DECOMPOSITION_NOT_IDENTIFIED`，不能把这组 projector 称为唯一语义分解。后续只有重新注册协议或显式重置 Supervisor protocol 后才能启动新一轮。冻结合同见 [LATENT-001](docs/experiments/LATENT-001_contract.md)，停止报告见 [LATENT-001 code-gate stop](docs/experiments/reports/LATENT-001_code_gate_stop.md)。

## 形式化对象

### 任务与环境

令 `C_tau` 为最小任务相关潜变量，所有同一任务环境共享：

\[
Y=f_\tau(C_\tau,\epsilon_Y).
\]

环境可改变 nuisance、观测与抽样机制：

\[
A_e=g_e(C_\tau,Y,\epsilon_A),\qquad
X_e=r_e(C_\tau,A_e,\epsilon_X),
\]

但不得修改 `f_tau`。允许的 task-preserving changes 构成 `I_tau`。这只是 task-equivalence class；要获得可学习的 DG 子族，还必须另行声明 task-state coverage、observation recoverability、source diversity 和目标干预几何。共享任务机制本身不蕴含可泛化性。

### 风险对象

区分三个风险，避免把 ERM、观测 oracle 和真实任务 oracle 混成一个记号：

\[
R_e^{C,*}=\inf_h R_e(h(C_\tau)),\qquad
R_e^{X,*}=\inf_{f\in\mathcal F_X}R_e(f),\qquad
R_e(f).
\]

ERM 只作比较基线。对方法 `j`，保留 signed quantity
\(R_T(f_j)-R_T(f_{\rm ERM})\)，必要时再取正部。鲁棒 causal excess 是
\(\sup_{\iota\in\mathcal I_\tau}[R_\iota(f)-R_\iota^{C,*}]\)，它不能被低退化的常数预测器替代。

## 当前研究接口

主线固定为：

```text
regularizer
  -> learned latent semantic subspaces
  -> controlled task-preserving shifts
  -> blind/mixed components and failure
  -> componentwise generalization-error accounting and bound
```

候选分解为

\[
\mathcal Z=\mathcal Z_{task}\oplus\mathcal Z_{relation}\oplus
\mathcal Z_{mean}\oplus\mathcal Z_{covariance}\oplus\mathcal Z_{residual}.
\]

这是需要由 source factorial supervision、cross-fitting、生成机制 oracle recovery、置换负对照和顺序敏感性共同检验的假设。白化只定义度量，PCA、聚类、encoder 坐标和 target 表现都不能命名这些子空间。

## 正则作用算子

对训练目标

\[
\hat f_{j,\lambda}\in\arg\min_f\{\widehat R_S(f)+\lambda\Omega_j(f)\},
\]

不预设不同正则都在学习“因果不变性”。每种方法先抽取它在明确状态空间中的 **实际作用算子** `L_{j,S}`：它可以是 moment discrepancy、environment-wise risk/gradient response、局部 Hessian response、marginal IPM、谱放大或参数/表示大小。

`Omega_j` 与 `L_{j,S}` 不自动等价。只有证明或验证桥接关系

\[
\|L_{j,S}v\|\leq \omega_j(v)\quad\text{或}\quad
\|P_jv\|\leq c\,\Omega_j(f)^{1/2}
\]

后，才允许说该 penalty 控制某个量。

在一个线性、局部线性化或 Hilbert-space setting 中，定义候选受控/盲区：

\[
\mathcal C_j=\mathcal H_{S,T}\cap\overline{\operatorname{range}(L_{j,S}^*)},
\qquad
\mathcal B_j\supseteq\mathcal H_{S,T}\cap\ker(L_{j,S}).
\]

`H_{S,T}` 是对当前 target family 有害的方向；它还必须扣除 source-unobservable directions。非线性或非二次 penalty 中，这些是局部/诊断定义，不能冒充全局正交分解。

研究产物必须同时给出：

1. `controlled component`：正则直接约束的 shift/model/representation mode；
2. `blind component`：penalty nullspace、source-unobservable direction 或模型/干预族失配；
3. `error accounting`：这些部分及统计、coverage、observation remainder 如何进入目标风险。

## 误差记账，而非先验 certificate

一般形式是待证明或待估计的条件结构：

\[
R_T(f)=R_S(f)+E^{\rm ctrl}_j(f,T)+E^{\rm blind}_j(f,T)
{}+E^{\rm interaction}_j(f,T)+E^{\rm stat/obs}_j(f,T).
\]

只有当某个构造给出 exact transport identity 时，右侧才可称 exact equality。通常目标是建立诸如

\[
|E^{\rm ctrl}_j(f,T)|\leq \Psi_j(\Omega_j(f),S,T)
\]

的条件界，并明确 `E_blind` 何时非零、不可由 source-only quantity 识别，或可被独立估计。完整 certificate 只是 `E_blind` 与 remainder 在额外假设下可界定的情况。

在线性平方损失模型中，`C001` 已给出 exact transport。将 nuisance coefficient 按某个正则的 projector 分成 `a=a_ctrl+a_blind` 后，二次型中的 controlled、blind 和 cross terms 给出精确记账；具体定义及符号见 [理论账本](docs/theory/00_theory_ledger.md)。这不是对一般深网络的自动定理。

## 研究计划

1. **冻结高维任务保持 SCM 与 source factorial design**：任务机制固定；source 覆盖 relation、mean、covariance 的前两轴，第三轴作为未见方向。
2. **学习并审计表示**：统一训练 ERM、L1/L2、IRMv1、MMD、CORAL、shared-head gradient/Hessian alignment。
3. **source-only 语义分解**：白化后由标签、环境编号和已声明干预类型回归原始语义算子，按固定层级残差化为正交 projectors。
4. **证伪语义识别**：cross-fit、oracle principal angles、标签置换、投影顺序、bottleneck 与 seed 稳定性缺一不可。
5. **回答成功与失败**：按预注册阈值判断方法能 OOD 哪类 covered shift，并将失败定位到 operator、source design、semantic mixing、collapse、interaction、optimization 或 out-of-family。
6. **误差界**：用同一 projectors 精确分解 target transport；只有存在正则算子到分量能量的 bridge 时才给出 Omega-dependent 条件界。

完整计划在 [课题简报](docs/research/00_project_brief.md)，开放问题在 [问题队列](docs/research/open_questions.md)。

## 证据纪律与协作

- 每条结论必须标为 `definition`、`assumption`、`exact equality`、`conditional theorem`、`bound`、`diagnostic`、`counterexample` 或 `conjecture`。
- “未找到文献”从不等于新颖；每个 Claim 需经过理论、反例、文献和实现审计。
- target risk、target moments 和 coverage residual 不得用于 source-only training/selection；可作为离线评估或理论 remainder。
- 旧的 PCA、cross-decomposition 和 certificate-first 路线保留为历史材料，不能被追溯性改写为本主线的支持证据。错误的 `MECH-001/C011` 已永久删除。
- `LATENT-001` 受四个独立只读 Supervisor gates 约束；Lead 不能越过 `VETO`，同一 gate 最多一次修复。

开始工作前读取 [项目提示词](PROJECT_PROMPT_CN.md)、[上下文管理](CONTEXT_MANAGEMENT.md)、[研究状态](docs/research/research_state.md) 和 [决策记录](docs/decisions/)。
