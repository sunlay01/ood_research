# Source Exposure 与 Identifiability

## 定义

source 风险观察算子为

\[
\mathcal O_S(Q)=
(\langle Q,M_{e_1}\rangle_F,\ldots,\langle Q,M_{e_m}\rangle_F).
\]

source exposure 是环境矩实际张成的方向；identifiability 是观察算子是否能区分目标状态。二者不可混同。

## 定理 5：source kernel 造成歧义

若 \(\Delta Q\in\ker\mathcal O_S\)，则 \(Q\) 与 \(Q+\Delta Q\) 的 source risk observations 相同。若存在允许目标矩阵 \(M_T\) 使

\[
\langle\Delta Q,M_T\rangle_F\neq0,
\]

则两者 target risk 不同。

### 证明

第一句直接来自 \(\mathcal O_S(\Delta Q)=0\)。第二句由

\[
\langle Q+\Delta Q,M_T\rangle_F-langle Q,M_T\rangle_F
=\langle\Delta Q,M_T\rangle_F
\]

且该值非零。

## source span 内的正结果

若 \(M_T=\sum_i a_iM_{e_i}\)，则

\[
\langle Q,M_T\rangle_F
=\sum_i a_i\mathcal O_S(Q)_i.
\]

此时 target risk pairing 可由 source observations 线性恢复；这需要 target 矩阵确实位于 source span，不能由“source 有多个环境”自动推出。

若 \(M_T\notin\operatorname{span}\{M_{e_i}\}\)，在完整矩阵线性空间中存在 source-null 的 \(\Delta Q\) 与它非正交。这给出 source-equivalent/target-different 的一般反例。若要求两个状态都必须是 rank-one \(vv^\top\)，还需额外构造；本轮 IRMv1 标量构造完成了这一点。
