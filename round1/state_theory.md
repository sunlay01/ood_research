# 一般风险状态：定理与商空间

## 定理 1：条件损失风险恒等式

设所有环境共享正则条件分布

\[
P_e(X,Y\mid C=c,A=a)=P^*(X,Y\mid C=c,A=a),
\]

而 \(\mu_e=P_e(C,A)\) 可以变化。对任意可积损失和固定 predictor \(f\)，定义

\[
L_f(c,a)=\mathbb E^*[\ell(f(X),Y)\mid C=c,A=a].
\]

则

\[
R_e(f)=\mathbb E_e[\ell(f(X),Y)]
=\int L_f(c,a)\,d\mu_e(c,a).
\]

### 证明

对联合分布使用条件期望塔式法则：

\[
R_e(f)=\int \mathbb E_e[\ell(f(X),Y)\mid c,a]\,d\mu_e(c,a).
\]

共享正则条件分布使条件期望等于同一个 \(L_f(c,a)\)，代入即得。若 \(P_e(X\mid C,A)\) 或 \(P_e(Y\mid C,A)\) 变化，则该条件期望带有环境下标，必须改写为 \(L_{f,e}\) 或扩展状态。

## 定理 2：允许环境族上的风险商

令 \(\mathfrak E_\tau\) 是预先声明的环境测度集合。定义

\[
h\in\mathcal N_\tau
\Longleftrightarrow
\int h\,d\mu=0,\quad \forall\mu\in\mathfrak E_\tau,
\]

以及 \(f\sim_\tau g\) 当且仅当 \(L_f-L_g\in\mathcal N_\tau\)。那么

\[
\mathcal G_\tau=\{L_f\}/\mathcal N_\tau
\]

是该环境族下的风险充分 quotient：两个 predictor 在所有允许环境中的风险相同，当且仅当它们的条件损失状态相差一个 \(\mathcal N_\tau\) 元素。

### 证明

由定理 1，\(R_\mu(f)-R_\mu(g)=\int(L_f-L_g)d\mu\)。右侧对所有 \(\mu\) 为零恰好就是 \(L_f-L_g\in\mathcal N_\tau\)。

## 非循环性审查

\(L_f\) 通常比完整 predictor 小，但对每个 \((c,a)\) 保存条件风险，未必是最小可计算描述。真正的最小对象是上面的商；例如若允许环境族只有一个固定 \(\mu\)，所有零均值方向都被商掉。这个 quotient 是 risk-equivalence 的数学对象，不是 hidden representation 的语义分解，也不保证 source 能识别它。
