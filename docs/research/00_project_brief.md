# 课题简报：OOD 正则化的机制、盲区与误差记账

## 研究问题

给定已有 DG/OOD 训练目标

\[
\hat f_{j,\lambda}\in\arg\min_f\{\widehat R_S(f)+\lambda\Omega_j(f)\},
\]

本项目研究：

\[
\boxed{
\Omega_j\ \mapsto\ L_{j,S}\ \mapsto\
(\text{controlled modes},\ \text{blind modes})\ \mapsto\
\text{target generalization error}.
}
\]

目标是对每个正则说明四件事：它实际约束什么、由此能缓解哪些任务保持型 shift、在哪些方向失效，以及这些项如何定量进入目标域风险。它不是“找一个 universal certificate”的项目；certificate 是 blind/remainder 已被额外条件控制时的特殊结果。

## 统一跨域定义

### 共享任务机制：`definition`

所有同一任务环境共享

\[
Y=f_\tau(C,\epsilon_Y).
\]

环境可改变

\[
A_e=g_e(C,Y,\epsilon_A),\qquad X_e=r_e(C,A_e,\epsilon_X),
\]

但不能改变 `f_tau`。`I_tau` 记允许的 task-preserving interventions。它定义了“同任务”，却不保证从 `X` 可以泛化。

### 可学习 DG 子族：`assumption`

对研究目标环境还需固定 `M_tau^DG subset M_tau`，并说明：

- `C` 的 coverage，例如 `P_T^C << P_S^C` 或有界 density ratio；
- observation recoverability，例如 `R_e^{X,*}-R_e^{C,*} <= delta_obs`；
- source environments 是否暴露 relevant harmful directions；
- target intervention 的精确几何/预算。

任何省略这些条件的“共享任务即泛化”表述都是不成立的。

### 风险：`definition`

\[
R_e^{C,*}=\inf_hR_e(h(C)),\quad
R_e^{X,*}=\inf_{f\in\mathcal F_X}R_e(f),\quad
R_e(f).
\]

主目标可用 target risk、robust causal excess
\(\sup_{\iota\in I_\tau}[R_\iota(f)-R_\iota^{C,*}]\)，或相对 ERM 的 signed target gap。三者分开报告；低 degradation 不等于低 absolute/causal excess。

## 机制形式化

### 正则作用算子：`definition / analysis target`

给每个方法在其真实状态空间中定义 `L_{j,S}`。例子包括：

| 方法 | 待推导的实际作用对象 | 不可直接宣称的含义 |
| --- | --- | --- |
| CORAL | environment covariance/moment discrepancy | 所有 conditional task shifts 都被消除 |
| MMD | chosen RKHS 下的 marginal IPM | label-conditional invariance |
| IRMv1 | scalar prediction-rescaling derivative | full predictor gradient invariance |
| full gradient | environment risk-gradient discrepancy | global interventional robustness |
| Hessian | local curvature/response discrepancy | nonlocal target stability |
| L1/L2 | coordinate/parameter magnitude | nuisance-selective invariance |
| spectral/rank | amplification/effective dimension | causal-feature recovery |
| Anchor/DRIG | 原始 uncertainty set 对应的结构响应 | 自由定义的任意 moment ball |

只有建立 `Omega_j` 到 `L_{j,S}` 或某个 projector 的 inequality/equality，才可把 penalty value 翻译为受控量。

### Controlled / blind components：`definition in linear/local setting`

设 `H_{S,T}` 为与目标 shift 相关的有害状态方向。在线性 Hilbert setting 中：

\[
\mathcal C_j=\mathcal H_{S,T}\cap\overline{\operatorname{range}(L_{j,S}^*)},
\qquad
\mathcal B_j\supseteq\mathcal H_{S,T}\cap\ker L_{j,S}.
\]

source response operator 的核也属于可辨识盲区。非线性情形需指定局部基点和切空间；不得把此记号误写为一般网络的全局正交分解。

### 误差记账：`research target`

目标形式为

\[
R_T(f)-R_S(f)=E_j^{\rm ctrl}+E_j^{\rm blind}
{}+E_j^{\rm interaction}+E_j^{\rm stat/obs/coverage}.
\]

这只是工作模板，除非具体模型已导出 equality。理想的条件桥接是

\[
|E_j^{\rm ctrl}|\leq\Psi_j(\Omega_j,S,T),
\]

并配套说明 `E_blind` 的 nullspace、不可识别性或可估计 remainder。只有在后两项已受控时才输出 complete certificate。

## 第一可证明模型与已有结果

使用部分可观测 Gaussian SCM：

\[
C\sim N(0,I),\quad U=LC+\xi,\quad
Y=\beta^\top C+\epsilon_Y,\quad
A=R_eC+\mu_e+\eta_e,
\]

模型观察 `X=(U,A)`，预测器为 `f_w=w_0+w_U^TU+w_A^TA`。令
`D=(1,C,xi,A)`、`b=(-w_0,beta-L^Tw_U,-w_U,-w_A)`、
`M_e=E_e[DD^T]`，已严格得到：

\[
R_e(w)=\sigma_Y^2+b^TM_eb,\qquad
R_T(w)-R_S(w)=b^T(M_T-M_S)b.
\]

这是 `exact equality`。对 correlation 和 nuisance moment intervention balls，最坏风险已有 exact formula，见 [C001](../theory/claims/C001_linear_intervention_risk.md)。

将 `w_A=P_jw_A+(I-P_j)w_A` 代入该二次型会产生 controlled、blind 和 interaction 项。它为 C006 提供 exact accounting；但是 `P_j` 必须从某个正则的真实 operator 导出，不能先按想要的结论任选。

## 已知边界

- `C004a`：source 的环境数不等于 observability；未被 source response operator 暴露的 nuisance direction 可以在 target 改变。
- `C002-IRM`：standard scalar-scale IRMv1 可以保留一个有害 nuisance direction，即使 source ERM 与 penalty 都为零。它是 regularizer-specific blind component，不是 IRMv1 新颖性主张。
- `C010`：在受限 scalar relation family，IRMv1 的每环境 scalar response 对 relation 为二次多项式；full-rank quadratic source design 条件地控制 `w_A^2`，而两 source relation 留下 blind branch。该结果只作为 operator/control/blind/error accounting 的首个样例。
- full-gradient 的 zero set 在当前标量模型可排除 nuisance，但 zero point 牺牲 source observational fit；这是 tradeoff，不能直接叫 positive theorem。
- Anchor/DRIG 的保障依赖其原始结构不确定性集；当前模型尚未与其严格等价。

## 阶段计划与停止条件

1. **C006，统一 accounting**：明确 target harmful space、operator-induced projector 和二次型分块；查清正交/符号/重参数化边界。
2. **方法审计**：从 L2、full-gradient 开始，随后 CORAL/MMD；每种方法先完成 zero set 与 blind-space，再推 controlled bridge。
3. **正反配对**：只保留在相同 SCM、同一 shift family 和可比较 observability 条件下的正/负结果。
4. **有限样本与表示层**：先核对 population theorem 的 estimator error，再推广到 `WB_A`、随后 `J_Af`。
5. **实验**：仅验证理论预言，记录 source fit、operator value、controlled/blind/interaction contribution、observability 和 shift geometry。

论文路线只有在至少一个非平凡 operator-to-error bridge、一个同框架 blind boundary、清晰的 existing-work 差异和 held-out intervention 可反驳预测同时成立时保留。否则停止或将成果定位为机制审计框架。
