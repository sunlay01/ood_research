# 3A 四维 shift null 的解释

当前 300 个 shift lifting 向量组成的矩阵 \\(\Psi\\) 在 21 维 quadratic-risk statistic space 中的数值 rank 是 17，所以右 null space 是 4 维。一个 null vector (n) 代表：

\[
n_M\!:\!\Delta M_{XX}+n_{XY}^{\top}\Delta m_{XY}+n_{Y^2}\Delta m_{Y^2}=0
\]

对当前所有 shifts 成立。它表示的是 **risk-statistic combination 没有被当前 shift design 激活**，不是 predictor 的 causal 方向，也不是 latent mechanism。

用完整 structural family（线性 Gaussian，允许 \\(\beta\\) 变化）做 reachability 对照后，数值 span rank 为 20。于是当前四维 null 中只有一维与完整 structural null 相交；这一区域由

\[
\Delta M_{00}=\Delta\mathbb E[1^2]=0
\]

给出。其余三维是当前有限 shift design 的 design-only null。若固定 task mechanism，reachable rank 会进一步下降；因此必须在报告中明确 target/environment family 的声明边界。

null basis 的具体系数见 `results/nullspace_functionals.csv`。坐标锚定 basis 便于阅读，但 null subspace 本身才是 invariant object。
