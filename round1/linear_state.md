# 线性平方损失状态

## 定理 3：有限维风险配对

令

\[
U=(C^\top,A^\top)^\top,\quad
v=(w_C-\beta,w_A),\quad Q_f=vv^\top,
\quad M_e=\mathbb E_e[UU^\top].
\]

则

\[
Y-f_w(U)= -v^\top U+\epsilon,
\]

从而

\[
R_e(f)-\sigma^2=v^\top M_ev
=\langle Q_f,M_e\rangle_F.
\]

对任意 source mixture \(M_S=\sum_i\pi_iM_{e_i}\)，有

\[
R_T(f)-R_S(f)=\langle Q_f,M_T-M_S\rangle_F.
\]

### 证明

展开平方：

\[
\mathbb E[(-v^\top U+\epsilon)^2]
=v^\top M_ev-2\mathbb E[\epsilon v^\top U]+\mathbb E\epsilon^2.
\]

条件零均值使交叉项为零。矩阵迹恒等式 \(v^\top M_ev=\operatorname{tr}(vv^\top M_e)\) 给出 Frobenius 配对；相减得到 transport identity。

## 定理 4：风险可见商

令

\[
\mathcal W_\tau=\operatorname{span}\{M_e:e\in\mathfrak E_\tau\}
\]

并以 Frobenius 内积定义正交投影 \(\Pi_{\mathcal W_\tau}\)。则对任意允许环境矩阵 \(M\in\mathcal W_\tau\)：

\[
\langle Q,M\rangle_F
=\langle\Pi_{\mathcal W_\tau}Q,M\rangle_F.
\]

因此 \(\Pi_{\mathcal W_\tau}Q\) 是这个线性 moment family 的风险充分状态，\(\mathcal W_\tau^\perp\) 是 risk-blind quotient。

### 证明

写 \(Q=\Pi_WQ+(I-\Pi_W)Q\)。第二项属于 \(W^\perp\)，而 \(M\in W\)，故其内积为零。

## 限制

该定理是对线性风险 pairing 的商化。若坚持 \(Q=vv^\top\)，rank-one 约束可能产生更小的非线性商；因此代码中的 SVD 只核验给定矩族的线性投影，不证明一般深度模型存在相同低维状态。按 \(C/A\) block 展开时，若 \(\Delta M_{CC}=0\)，有

\[
\Delta R=2(w_C-\beta)^\top\Delta M_{CA}w_A+w_A^\top\Delta M_{AA}w_A,
\]

这正是 nuisance use 与 shift geometry 的交互，而不是单独的“表示偏移量”。
