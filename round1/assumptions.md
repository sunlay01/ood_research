# 第一轮假设

本文只研究同一任务下的环境变化，不允许目标环境改变任务机制。最小模型为

\[
C\sim P(C),\quad Y\sim P^*(Y\mid C),\quad X\sim P^*(X\mid C,A),
\]

其中环境只改变 \(\mu_e=P_e(C,A)\)。若观测机制 \(P_e(X\mid C,A)\) 也变化，则必须把它加入环境状态；不能继续使用一个跨环境共享的 \(L_f(c,a)\)。

线性核验采用平方损失、有限二阶矩和

\[
Y=\beta^\top C+\epsilon,\qquad
f_w(C,A)=w_C^\top C+w_A^\top A,
\]

并假设 \(\mathbb E[\epsilon\mid C,A]=0\)、\(\mathbb E\epsilon^2=\sigma^2\)。所有环境状态均为二阶矩；target 只用于离线核验，不用于选择模型、正则强度或状态定义。

本文中的“风险充分”只表示对预先声明的环境族计算风险时充分，不表示对所有可能分布充分；“可识别”只表示由声明的 source observable 唯一确定，不表示因果变量被恢复。
