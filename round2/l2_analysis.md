# L2 induced geometry

## 定理 T7

令 \(Q=qq^\top\)，\(q=(q_C,q_A)\)，且直接 predictor 的误差状态满足

\[
w=(\beta,0)+s q,\qquad s\in\{-1,+1\}.
\]

对 \(\Omega_2(w)=\|w\|_2^2\)，有

\[
\bar\Omega_2(Q)
=\min_s\| (\beta,0)+sq\|_2^2
=\operatorname{tr}(Q)+\|\beta\|_2^2-2|q_C^\top\beta|.
\]

### 证明

由 rank-one 分解，\(Q=qq^\top\) 的向量代表只有 \(q\) 和 \(-q\)。展开平方并对两个符号取最小值即可。

## 泛化界

由风险配对和谱范数不等式：

\[
|R_T-R_S|\leq\|M_T-M_S\|_{op}\|v\|_2^2.
\]

若固定参数化下 \(\|v\|\leq c_0+c_1\|w\|\)，才可写成 L2 相关的条件界。L2 本身不提供 task/nuisance selective bridge。

## 非选择性

\(\bar\Omega_2\) 依赖整个 residual state 及其与 \(\beta\) 的符号关系，而不是只依赖 \(q_A\)。因此两个 target risk 意义不同的方向可以具有相同或相近 induced cost。结论是 `fiber-induced, nonselective`，不是 invariant representation theorem。
