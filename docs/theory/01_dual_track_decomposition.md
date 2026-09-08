# 双轨表示误差分解

> 状态：`THEORY_BASELINE / EXACT_IDENTITIES_ONLY`。本文件给出 population/oracle 层的恒等式，不给出 source-only target-risk certificate，也不构成论文新颖性主张。

## 1. 目的与边界

固定环境 (e)，输入 (X)、标签 (Y)、表示 (Z=phi(X)) 与预测头 (h)。本 construction 的作用是把同一个模型的风险拆成：

1. 不受表示或 head 控制的 Bayes/noise 项；
2. 将 (X) 压缩到 (Z) 造成的 representation insufficiency；
3. 给定 (Z) 后所选 head 的 regret/mismatch。

它不替换 Wu et al. 的 source-target hybrid decomposition：后者将 target risk 拆为 conditional-label divergence、covariate shift 和 support residual；本文件先在**单一环境内**分离 encoder 与 decoder。两个 construction 仅能通过后述 transport refinement 关联，不能直接把各自分量相加或排名。

## 2. 平方损失：exact Pythagorean identity

令

\[
m_e(x)=\mathbb E_e[Y\mid X=x],\qquad
m_e^\phi(z)=\mathbb E_e[Y\mid Z=z],
\]

并假设二阶矩有限。对任意平方可积 head (h(Z))，定义

\[
N_e=\mathbb E_e[(Y-m_e(X))^2],
\]
\[
I_e(\phi)=\mathbb E_e[(m_e(X)-m_e^\phi(Z))^2],
\qquad
H_e(h,\phi)=\mathbb E_e[(m_e^\phi(Z)-h(Z))^2].
\]

**ID-011 (`exact equality`).**

\[
R_e(h\circ\phi)=N_e+I_e(\phi)+H_e(h,\phi).
\]

**证明。** 将

\[
Y-h(Z)=[Y-m_e(X)]+[m_e(X)-m_e^\phi(Z)]+[m_e^\phi(Z)-h(Z)]
\]

平方后取期望。第一项与任意 (X)-可测函数正交；第二项对任意 (Z)-可测函数条件均值为零。因此三个交叉项均为零。

对两个算法或正则路径点 (a,b) 使用同一 target 环境，Bayes/noise 项相消：

\[
R_T(f_a)-R_T(f_b)=
[I_T(\phi_a)-I_T(\phi_b)]
+[H_T(h_a,\phi_a)-H_T(h_b,\phi_b)].
\]

这不是正部退化量的线性分解；若报告

\[
D_{T,j}=[R_T(f_j)-R_T(f_{\rm ERM})]_+,
\]

必须先报告上式 signed difference，再单独取正部。

## 3. Proper loss：Bayes/Bregman identity

令 (Y\in\{1,\ldots,K\})，

\[
\eta_e^X=P_e(Y\mid X),\qquad \eta_e^Z=P_e(Y\mid Z).
\]

对可微严格 proper scoring rule \(\ell\)，令 (\underline L\) 为 Bayes envelope，(F=-\underline L) 为凸函数，(B_F) 为其 Bregman divergence。假设所选 head 输出 simplex 内的 (q_h(Z))，且相关期望有限。定义

\[
I_e^\ell(\phi)=\mathbb E_e B_F(\eta_e^X,\eta_e^Z),
\qquad
H_e^\ell(h,\phi)=\mathbb E_e B_F(\eta_e^Z,q_h(Z)).
\]

**ID-012 (`exact equality`).**

\[
R_e(q_h\circ\phi)=R_e^\star+I_e^\ell(\phi)+H_e^\ell(h,\phi),
\qquad
R_e^\star=\mathbb E_e\underline L(\eta_e^X).
\]

**证明义务。** proper-loss regret 等于

\[
\mathbb E_e B_F(\eta_e^X,q_h(Z)).
\]

再使用 (\eta_e^Z=\mathbb E[\eta_e^X\mid Z]) 的 conditional Bregman Pythagorean identity。该论证要求使用完整条件分布的 oracle；有限样本训练输出不能自动满足它。

**special cases。** log loss 中

\[
I_e^\ell(\phi)=\mathbb E_e\mathrm{KL}(\eta_e^X\Vert\eta_e^Z)=I_e(Y;X\mid Z).
\]

平方损失的 (I_e) 和 (H_e) 是同一 Bayes-regret construction 的条件均值版本。

## 4. 跨域 transport refinement

本 refinement 用于一个已定义分量，不能被当作第三个可自由相加的分解。令状态 (U) 的 source/target law 满足 Lebesgue decomposition

\[
P_T^U=r\,P_S^U+P_T^\perp.
\]

令 (a_e(U)) 是环境相关、可积的分量 integrand。则

\[
\mathbb E_Ta_T-\mathbb E_Sa_S
=\underbrace{\int r(a_T-a_S)dP_S}_{\text{functional/conditional shift}}
+\underbrace{\int(r-1)a_SdP_S}_{\text{density reweighting}}
+\underbrace{\int a_TdP_T^\perp}_{\text{singular coverage residual}}.
\]

这只是加减 (\int r a_SdP_S) 后使用 Radon--Nikodym 定理得到的 `exact equality`。对 head term 取 (U=Z)、(a_e=B_F(\eta_e^Z,q_h)\)；对 information term 取 (U=(X,Z))、(a_e=B_F(\eta_e^X,\eta_e^Z)\)。因此 conditional response、density reweighting 与 coverage 均可分别审计，但 target-dependent 项不能伪装为 source-only 指标。

## 5. 正则路径接口

令 

\[
J_S(\theta,\lambda)=R_S(f_\theta)+\lambda\Omega(\theta),
\]

且 (\theta_\lambda) 是孤立可微驻点，

\[
\nabla_\theta J_S(\theta_\lambda,\lambda)=0,
\qquad
\nabla_\theta^2J_S(\theta_\lambda,\lambda)\text{ 可逆}.
\]

对任一可微分量 (E_{k,e}(\theta))，隐函数定理给出

\[
\frac{d}{d\lambda}E_{k,e}(\theta_\lambda)
=-\nabla_\theta E_{k,e}(\theta_\lambda)^\top
[\nabla_\theta^2J_S(\theta_\lambda,\lambda)]^{-1}
\nabla_\theta\Omega(\theta_\lambda).
\]

这是 `conditional local result`，不是对任意深网或非唯一训练解的结论。它说明正则化的可解释接口是由 
\(\nabla\Omega\)、训练曲率和分量梯度耦合形成的 tangent response，而不是原始标量 \(\Omega\) 的数值。

## 6. 最小反例与审计规则

- **collapse：** (Z=0)、(Y\in\{-1,1\}\) 且 (m(X)=Y) 时，(I=1,H=0)。零 alignment 可以对应最大的表示信息损失。
- **head mismatch：** (Z=X\)、(m(Z)=Y)、但 (h=0) 时，(I=0,H=1)。不丢表示信息不等于训练头正确。
- **parameter gauge：** 线性 (z=Bx,h(z)=w^\top z) 中，(B\mapsto cB,w\mapsto w/c) 保持预测、风险、(I,H) 不变，却一般改变 L2/L1 参数正则。因此没有规范化/gauge 固定时，原始 \(\Omega\) 的数值不可作为 representation-error 的坐标无关解释。
- **unseen support：** 同一 source observations 可对应不同 (P_T^\perp) 或 target conditionals；上述 source 项本身不能无条件控制 target coverage residual。

## 7. 文献边界

- Wu et al. (2020) 已给 representation-conditioned Bayes predictor、hybrid 与 Lebesgue target-risk decomposition；本文件不替代或重新命名该结果。
- van Rooyen and Williamson (2014) 以 Le Cam deficiency 讨论 feature information 与 decision risk；它支持“表示充分性是决策风险对象”的统计基础。
- Reid and Williamson (2011) 给 proper loss、Bayes risk、Bregman divergence/statistical information 的标准联系；它支持 proper-loss 表述。

因此 `ID-011`、`ID-012` 和 transport lemma 是研究审计基座，不得单独作为新颖性主张。下一步只有在六轴文献比较完成后，才判断“regularizer response across these quantities”是否可形成不同于现有工作、可反驳的贡献。
