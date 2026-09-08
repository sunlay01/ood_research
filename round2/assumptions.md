# 第二轮假设

本轮继承第一轮的平方损失线性 SCM。任务机制固定，环境通过二阶矩集合 \(\mathcal M_\tau\) 改变；风险状态使用有限维对称矩阵空间 \(\mathbb S^p\)。target 环境只用于离线检验，不参与 source-side cost、参数或模型选择。

本轮中的公共空间是风险商空间，不是神经网络 hidden space：

\[
Q_f=vv^\top,\qquad R_e(f)-\sigma^2=\langle Q_f,M_e\rangle_F.
\]

因而所有“可见/盲”结论都相对于预先声明的目标矩族而言。任何 representation-level 结论还必须额外声明表示类、head、gauge 和 fiber 非空性。
