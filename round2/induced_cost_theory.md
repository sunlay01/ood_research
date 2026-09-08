# Quotient-induced regularization

## 定义

令 \(q:\Theta\to\mathcal G_\tau\) 为参数 realization 到风险商的映射，\(\Omega_j:\Theta\to[0,\infty]\)。定义 fiber-induced cost：

\[
\bar\Omega_j(g)=\inf_{\theta:q(\theta)=g}\Omega_j(\theta),
\]

空 fiber 的 infimum 取 \(+\infty\)。若 \(\Omega_j=\Omega_j^\sharp\circ q\)，称其 direct descent；否则只能称 induced。

## 定理 T6：infimal projection

若 \(R_S\) 只依赖 \(q(\theta)\)，且 \(\lambda\geq0\)，则

\[
\inf_\theta[R_S(q(\theta))+\lambda\Omega_j(\theta)]
=
\inf_{g\in q(\Theta)}[R_S(g)+\lambda\bar\Omega_j(g)].
\]

### 证明

按 fiber 分组。固定 \(g\) 时第一项为常数，第二项在该 fiber 上的下确界是 \(\bar\Omega_j(g)\)。再对所有非空 fiber 取 infimum，即得等式。若需要 argmin correspondence，必须额外假设 fiber infimum attain 且全局 infimum attain；等式本身不保证 minimizer 存在。

## 重要审查

T6 是标准 infimal projection 事实，不是论文贡献。它只有在具体正则产生非退化、可比较的 \(\bar\Omega_j\) 时才有研究价值。表示缩放可能使 induced cost 退化；因此不能未经 gauge 就声称 CORAL 在风险商上有几何。
