# 第三轮假设

主轨使用 population linear Gaussian SCM：

\[
C\sim N(\mu_C,\Sigma_{CC}),\quad A=\Gamma C+b+\xi_A,
\quad Y=\beta^T C+\epsilon.
\]

task mechanism 与 observation mechanism 固定；core marginal、nuisance marginal 和 relation 可以变化。task/observation shift 只作为 out-of-family stress test。所有机制语义都相对于声明的 structural family，不声称可从任意 observational distribution 唯一恢复。
