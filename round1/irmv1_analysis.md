# IRMv1：径向响应的定理与盲区

## 定义

对固定 predictor \(f_w\)，标准 scalar-scale IRMv1 为

\[
\Omega_{\mathrm{IRM}}(w)=\frac1m\sum_e
\left[\left.\frac{d}{d\alpha}R_e(\alpha f_w)\right|_{\alpha=1}\right]^2.
\]

它观测的是风险梯度在当前 predictor 射线上的投影，不是完整环境梯度。

## 定理 6：标量 relation response

令 \(U=lC+\xi\)、\(A=rC+\eta\)，\(Y=\beta C+\epsilon\)，均值为零，噪声方差固定，\(f=uU+aA\)。令

\[
q(r)=\mathbb E[(f-Y)f].
\]

则

\[
q(r)=q_0+q_1r+q_2r^2,
\quad q_2=a^2,
\]

其中

\[
q_0=(l^2+\sigma_\xi^2)u^2-l\beta u+\sigma_\eta^2a^2,
\quad q_1=a(2lu-\beta).
\]

且 IRMv1 derivative 为 \(2q(r)\)。

### 证明

有 \(f=(lu+ar)C+u\xi+a\eta\)。因此

\[
\mathbb E f^2=(lu+ar)^2+u^2\sigma_\xi+a^2\sigma_\eta,
\quad
\mathbb E[Yf]=\beta(lu+ar).
\]

相减并按 \(r\) 收集系数即得。

## 定理 7：三点 relation 的条件 bridge

定义 Vandermonde 矩阵 \(V_S=[(1,r_i,r_i^2)]_i\)。若 \(\sigma_{\min}(V_S)>0\)，则

\[
\Omega_{\mathrm{IRM}}=\frac4m\|V_Sq\|_2^2,
\qquad
a^2=q_2\leq\frac{\sqrt{m\Omega_{\mathrm{IRM}}}}{2\sigma_{\min}(V_S)}.
\]

### 证明

\(q=(q_0,q_1,q_2)^\top\)，故 \(\sum_iq(r_i)^2=\|V_Sq\|^2\)。由最小奇异值，\(\|V_Sq\|\geq\sigma_{\min}(V_S)\|q\|\geq\sigma_{\min}(V_S)|q_2|\)，整理即可。

## 失败边界

两个 relation 点时 \(V_S\) 只有两行，必有非零多项式在 source 点为零；现有 C002 参数给出 source-optimal、zero-penalty 但 target risk 显著增加的 rank-one predictor。径向 penalty 仍不约束 tangent gradient。即使三点 bridge 成立，它只控制 relation response；mean shift、covariance shift、低秩设计和 out-of-family target 仍是 blind residual。

因此 IRMv1 的准确结论是：在充分 relation exposure 与非退化 source design 下，能条件控制 relation-induced nuisance use；不是无条件的 OOD certificate。
