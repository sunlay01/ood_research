# 3A Pure-family geometry

为避免把补充实验中带有 mixed component 的 `kind` 标签误当作 pure family，本收尾重新生成了七组单轴 structural probes：

\[
V_{\mathrm{coremean}},V_{\mathrm{corecov}},V_{\mathrm{relation}},
V_b,V_{\mu_\xi},V_{\Sigma_\xi},V_{\mathrm{task}}.
\]

每个结构坐标使用正负 finite shift。结果同时在 risk-statistic lifting space 和 model-response space 中计算 rank、intersection、principal angles、inclusion residual 与 incremental rank。

一个关键 exact structural observation 是：

\[
V_b=V_{\mu_\xi},
\]

因为 (A=\Gamma C+b+\xi) 的线性 Gaussian moments 只依赖 (b+\mu_\xi)。因此从当前 observed moments 不能把这两个 structural sources 分开。

其余 family 之间通常出现部分重叠或零交，而不是正交 direct sum。所有结果只属于 response-space algebra；family 名称是 structural bookkeeping，不能在 3A 中升级为机制语义。
