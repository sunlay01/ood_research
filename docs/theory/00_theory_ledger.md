# 理论账本：正则作用算子、盲区与 OOD 误差记账

> 每项均明确结论类型。本文不把一般模板误写为 theorem；已有线性结果与待完成 operator 分析严格分开。

## 定义与假设

### DEF-001 — 共享任务机制：`definition`

环境 `e` 属于同一任务，当它们共享

\[
Y=f_\tau(C,\epsilon_Y),
\]

而只改变 nuisance/observation/sampling mechanisms。允许的环境变化为 `I_tau`。共享任务不意味着 `X` 中的任务信息可恢复。

### DEF-002 — 可学习 DG 子族：`assumption`

`M_tau^DG subset M_tau` 需单独规定：(i) task-state coverage，(ii) observation recoverability，(iii) source observability，(iv) target shift geometry/预算。任何风险 bound 都必须列出所用部分。

### DEF-003 — oracle 与风险：`definition`

\[
R_e^{C,*}=\inf_h R_e(h(C)),\quad
R_e^{X,*}=\inf_{f\in\mathcal F_X}R_e(f),\quad R_e(f).
\]

对固定 target 还可报告 signed ERM gap；它不是 causal excess 的替代。

### DEF-004 — regularizer-induced operator：`definition / per-method obligation`

对方法 `j` 在明确状态空间 `V_j` 中提取实际 operator `L_{j,S}:V_j -> W_j`。`Omega_j` 只有在给出 relation
\(\|L_{j,S}v\|\leq\omega_j(v)\) 或 coercive bridge 后才解释为其控制量。`L` 可为线性 operator、局部导数、moment map 或 IPM embedding；后两者不默认有全局线性核空间。

### DEF-005 — harmful, controlled, blind directions：`definition in linear/local setting`

给定 target family 和表示/预测状态的有害集合 `H_{S,T}`，在线性 Hilbert setting 定义

\[
\mathcal C_j=\mathcal H_{S,T}\cap\overline{\operatorname{range}(L_{j,S}^*)},
\qquad
\mathcal B_j\supseteq\mathcal H_{S,T}\cap\ker L_{j,S}.
\]

任何 source response operator 的核应并入 candidate blind set。该定义依赖 state space、source 和目标几何；不是方法的无条件固有属性。

## 已验证线性基座

### ID-001 / C001 — 部分可观测 linear risk transport：`exact equality`

令

\[
C\sim N(0,I),\quad U=LC+\xi,\quad
Y=\beta^TC+\epsilon_Y,\quad A=R_eC+\mu_e+\eta_e,
\]

并令 `f_w(X)=w_0+w_U^TU+w_A^TA`、`D=(1,C,xi,A)`、
`b=(-w_0,beta-L^Tw_U,-w_U,-w_A)`、`M_e=E_e[DD^T]`。独立零均值噪声下：

\[
R_e(w)=\sigma_Y^2+b^TM_eb,\qquad
R_T(w)-R_S(w)=b^T(M_T-M_S)b.
\]

证明、两类 exact robust formula 和执行核验见 [C001](claims/C001_linear_intervention_risk.md)。

### ID-002 / C006 — operator-induced quadratic accounting：`exact equality conditional on a projector`

在固定 `P(C)` 的直接/等价 nuisance block 记号中，令 `a=w_A`、`delta` 为 task-observation residual，且

\[
\Delta R=2\delta^T\Delta M_{CA}a+a^T\Delta M_{AA}a.
\]

给定由某个已定义的 linear/local operator 导出的正交 projector `P_j`，写
`a_c=P_ja`、`a_b=(I-P_j)a`。则恒等地：

\[
\begin{aligned}
E_j^{\rm ctrl}&=2\delta^T\Delta M_{CA}a_c+a_c^T\Delta M_{AA}a_c,\\
E_j^{\rm blind}&=2\delta^T\Delta M_{CA}a_b+a_b^T\Delta M_{AA}a_b,\\
E_j^{\rm interaction}&=a_c^T\Delta M_{AA}a_b+a_b^T\Delta M_{AA}a_c,\\
\Delta R&=E_j^{\rm ctrl}+E_j^{\rm blind}+E_j^{\rm interaction}.
\end{aligned}
\]

这是一条关于 **任意已给 projector** 的代数恒等式，尚不是“某正则控制 `E_ctrl`”的 theorem。下一义务是从 `L_{j,S}` 推导 `P_j`，并给出 `Omega_j -> a_c` 与 target matrix norm 的条件 bridge。若 `P_j` 是事后以 target 风险选取，此式不能称 source-only 分析。

### BND-001 — 受控项的条件界：`conditional bound`

若 `||a_c|| <= q_j(Omega_j(f),S)`，并且目标矩阵块有已声明范数界，则

\[
|E_j^{\rm ctrl}|
\leq 2\|\Delta M_{CA}^T\delta\|\,q_j
{}+\|\Delta M_{AA}\|_{\rm op}q_j^2.
\]

该界不控制 `E_blind` 或 interaction，也不因 `Omega` 小而自动趋零；`delta`、矩阵几何和 bridge 都是必要条件。

### NEG-001 / C004a — source-unobservable direction：`counterexample`

source response operator 可以在某 nuisance coordinate 上为零，即使另一个 coordinate 在 source 中变化。相同任务机制与 source observations 可对应 target 上巨大风险差。详见 [C004a](claims/C004a_source_unobservability.md)。它限制的是依赖该 source response 的统计量，并不否定已知 nuisance identity 的直接 penalty。

### NEG-002 / C002-IRM — scalar-scale IRMv1 blind component：`counterexample`

在 C001 的标量模型，source ERM 同时达到 standard scalar-scale IRMv1 zero penalty，却对允许 correlation sign-flip 有大 robust causal excess。详见 [C002-IRM](claims/C002_irmv1_blind_direction.md)。它证明该 penalty 的实际 scalar rescaling operator 留下有害盲区；不适用于 full-gradient penalty，也不构成 standalone novelty。

### C010 — ERM--IRMv1 mechanism audit：`exact equality / conditional theorem / counterexample`

对平方风险，ERM 的 operator 仅为 source-mixture stationarity；standard scalar-scale IRMv1 的 operator 是每个环境的 radial response `w^T grad R_e(w)`，不控制 tangent gradient。在线性标量、零均值、共同 nuisance variance、零截距的 relation family，令 `q(r)=q0+q1r+q2r^2` 为未缩放 IRMv1 response，则 `q2=w_A^2`。

若 source quadratic design `V_S=[1,r,r^2]` 满列秩，`kappa_V=sigma_min(V_S)>0`，有

\[
w_A^2\leq\frac{\sqrt{m\Omega_{\rm IRMv1}}}{2\kappa_V}.
\]

将此代入 C001，并设 `b_0` 为去除 nuisance coefficient 的 residual，可得

\[
|R_T-R_S|\leq\|M_T-M_S\|_{\rm op}
\left(2\|b_0\|A_\Omega+A_\Omega^2\right).
\]

这是 `conditional theorem`：target geometry、base observation residual 与共同 SCM 是显式前提。零 penalty 加严格 source-fit 条件产生非平凡 U-only predictor，对 relation/mean/covariance nuisance shifts 精确稳定；其 causal excess 仍可为正。两个 source relation 时存在 C002 的 nonzero-nuisance blind branch，故不能推广该正结果。详见 [C010](claims/C010_erm_irmv1_mechanism.md)。

## 待完成的 method-specific bridges

| ID | 方法 | 必须先完成的对象 | 允许的结论 |
| --- | --- | --- | --- |
| C007-L2 | L2 | `L` 的 state space、对 `w_A`/effective sensitivity 的 bridge、task-observation shrinkage | selective failure 或有限条件 tradeoff，不称 invariance |
| C007-G | full gradient | environment gradient-response operator、zero set、source-fit frontier | specific controlled/null directions |
| C008-CORAL | CORAL | covariance operator、conditional-shift nullspace | moment-controlled component + conditional blind residual |
| C008-MMD | MMD | RKHS mean embedding/operator 与 target risk bridge | IPM-controlled term + conditional blind residual |
| C009 | representation | `WB_A` 或 `J_Af` 的坐标不变 state quantity | linear first, then local nonlinear result |

`C010` 已完成 IRMv1 的标量总体审计。它不关闭 vector/nonlinear IRMv1 问题，也不与 C007-G 的 full-gradient operator 合并。

## 记录规则

每项新推导都写入：使用定义/假设、结论类型、完整符号、反例攻击面、文献近邻与可执行核验。不得把 BND-001 的假设当成由任何已有 OOD objective 自动推出的事实。
