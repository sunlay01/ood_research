# IRMv1 induced geometry

## 定理 T9：relation response bridge

在标量模型 \(A=rC+\eta\)、零均值和共同噪声方差下，令 \(f=uU+aA\)。IRMv1 的未缩放响应满足

\[
q(r)=q_0+q_1r+q_2r^2,\qquad q_2=a^2,
\]

且 penalty 为

\[
\Omega_{IRM}=\frac4m\|V_Sq\|_2^2,
\qquad V_S=[1,r_i,r_i^2]_i.
\]

若 \(\sigma_{min}(V_S)>0\)，则

\[
a^4=q_2^2
\leq
\frac{m\Omega_{IRM}}{16\sigma_{min}(V_S)^2}.
\]

### 证明

每个环境的 scalar derivative 为 \(2q(r_i)\)，故 penalty 等式成立。最小奇异值给出 \(\|V_Sq\|\geq\sigma_{min}(V_S)\|q\|\geq\sigma_{min}(V_S)|q_2|\)，平方并整理即可。

## 定理 T9b：两点盲区

当 source 只有两个 relation 点时，\(V_S\) 至多 rank 2，存在非零二次 response polynomial 在两点同时为零。第一轮的 rank-one branch 是该盲区的具体实例：source risk 与 IRMv1 penalty 都相同/为零，但 target relation 改变后风险不同。

## Failure separation

对第一轮 branch，若差异状态 \(D\) 满足 \(D\in\ker\mathcal O_S\) 且 \(D\notin\mathcal N_\tau\)，失败应标为 `source-unidentified`；只有 source observations 能区分该方向而 IRMv1 induced cost 不增加时，才能称为 `regularizer-blind`。

三点满秩只对 relation-response quotient 提供 coercivity，不自动控制 mean/covariance shift、完整 gradient tangent 或 out-of-family target。

还要注意：一般 IRMv1 penalty 不一定是 \(Q=vv^\top\) 的函数。\(Q\) 识别 \(v\) 与 \(-v\)，但径向响应包含 \(w^\top\Sigma_e w-w^\top c_e\)，符号可能改变该量。因此 IRMv1 在公共风险商上通常只能通过保留 realization 信息或对 fiber 取 infimum/其他聚合来定义；它不自动 direct-descend 到风险 quotient。对标量直接 predictor 的 rank-one sign fiber，代码中的 `irmv1_direct_induced_cost` 计算两种符号 realization 的精确下确界；这不等于一般 representation fiber 上的 attain 结论。
