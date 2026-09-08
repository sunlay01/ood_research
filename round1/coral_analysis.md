# CORAL：协方差响应与条件盲区

## 定理 8：CORAL 的实际观察对象

对固定线性表示 \(Z=BU\)，令 \(\Sigma_e=\operatorname{Cov}_e(U)\)。则

\[
\operatorname{Cov}_e(Z)=B\Sigma_eB^\top,
\]

所以 CORAL 的 pairwise observation 是

\[
B(\Sigma_e-\Sigma_{e'})B^\top.
\]

### 证明

\(Z-\mathbb EZ=B(U-\mathbb EU)\)，直接展开协方差即可。

## 条件界

若目标只改变表示协方差且 head 为 \(w_Z\)，则 covariance transport 为

\[
w_Z^\top\Delta\Sigma_Zw_Z,
\]

从而

\[
|E_{cov}|\leq\|\Delta\Sigma_Z\|_{op}\|w_Z\|^2.
\]

这只是 covariance term 的条件界；它不包含均值和条件标签机制。

## 反例

只改变 nuisance mean 时，centered covariance 不变，因此 CORAL value 为零；若 predictor 含截距或 nuisance coefficient，风险可因均值改变。只改变 \(P(Y\mid C,A)\) 时，表示协方差也可完全不变而风险改变。若 \(B\) 删除 task-relevant direction，CORAL 还可能通过信息丢失降低 discrepancy，却损害任务风险。

因此 CORAL 能控制的只是其表示和 source design 实际暴露的 marginal covariance mode；mean、conditional response、未暴露协方差和 representation loss 都是盲区。
