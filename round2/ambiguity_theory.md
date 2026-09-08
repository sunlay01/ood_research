# Target-visible / source-unidentified ambiguity

## 定义

对目标矩族定义

\[
\mathcal W_\tau=\operatorname{span}\{M:M\in\mathcal M_\tau\},\qquad
\mathcal N_\tau=\mathcal W_\tau^\perp.
\]

\(\mathcal N_\tau\) 是 target annihilator。风险商为 \(\mathcal G_\tau=\mathbb S^p/\mathcal N_\tau\)，其维数为 \(\dim\mathcal W_\tau\)。source absolute observation 为

\[
\mathcal O_S(Q)=(\langle Q,M_{e_i}\rangle_F)_i.
\]

下文把 source observation 限制到 \(\mathcal G_\tau\) 时，假设每个 source moment
也属于 \(\mathcal W_\tau\)（例如声明的 target family 包含 source environments）。
这是必要的 well-defined 条件：若 source moment 不在 \(\mathcal W_\tau\)，则
\(\mathcal O_S\) 可能依赖同一 quotient class 的代表元，不能直接写成
\(\mathcal O_S|_{\mathcal G_\tau}\)。不满足该条件时，只能在 ambient
\(\mathbb S^p\) 上讨论 source kernel，或扩充 target risk state。

## 定理 T5：ambiguity 的充要条件

令 \(D\in\mathbb S^p\)。则 source absolute risk 看不到 \(D\) 当且仅当

\[
D\in\ker\mathcal O_S.
\]

其在 target risk quotient 中非零，当且仅当

\[
D\notin\mathcal N_\tau.
\]

在上述 well-defined 条件下，target-visible/source-unidentified ambiguity 正是

\[
\mathcal A_{S,\tau}
=\ker(\mathcal O_S|_{\mathcal G_\tau}),
\]

并且

\[
\dim\mathcal A_{S,\tau}
=\dim\mathcal G_\tau-operatorname{rank}(\mathcal O_S|_{\mathcal G_\tau}).
\]

### 证明

\(\mathcal O_S(D)=0\) 的定义就是所有 source pairing 为零。若 \(D\in\mathcal N_\tau\)，则与所有 admissible target moment 的 pairing 都为零，故在商空间中为零；反之若不在 annihilator，存在目标矩与之非正交。最后的维数公式是线性映射的 rank-nullity theorem。

## 定理 T5b：source span 内可恢复

若 \(M_T=\sum_i a_iM_{e_i}\)，则

\[
\langle Q,M_T\rangle_F=\sum_i a_i\mathcal O_S(Q)_i.
\]

所以 target risk functional 可由 source absolute risks 精确恢复。这里使用的是 absolute observation；centered transport observation 不能恢复绝对风险基线。

## 定理 T5c：shift-ball support function

令目标变化空间为 \(\mathcal H\subseteq\mathbb S^p\)，并令

\[
\mathcal M_{\rho}=\{M_S+\Delta:\Delta\in\mathcal H,\ \|\Delta\|_F\leq\rho\}.
\]

则对任意状态差异 \(D\)：

\[
\sup_{M_T\in\mathcal M_\rho}|\langle D,M_T-M_S\rangle_F|
=\rho\|\Pi_\mathcal H D\|_F.
\]

### 证明

Cauchy--Schwarz 给出上界。取 \(\Delta=\rho\Pi_\mathcal H D/\|\Pi_\mathcal H D\|_F\)（非零时）达到上界；零时两边都为零。若同时要求 \(M_T\succeq0\) 或满足 relation 参数化，这个等式需改为相应凸集的 support function，不能继续无条件使用。

## 解释

第一轮的 source kernel 维度不是 ambiguity 维度。只有 source kernel 与 target-visible quotient 的交集才是真正的 DG ambiguity；source 看不到但 target 也看不到的方向必须被 quotient 掉。
