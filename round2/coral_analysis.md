# CORAL induced geometry

## 定理 T8：无 gauge 时退化

令 \(Z=BX\)、\(\hat Y=u^\top Z\)，最终 predictor 为 \(w=B^\top u\)。若 \((B,u)\) 是一个可行 realization，则对任意 \(c>0\)：

\[
B_c=cB,\qquad u_c=u/c
\]

保持 \(B_c^\top u_c=w\)，但

\[
\Omega_{CORAL}(B_c)=c^4\Omega_{CORAL}(B).
\]

因此只要存在一组可行 realization，且 CORAL 值有限，便有

\[
\bar\Omega_{CORAL}(g)=0.
\]

### 证明

最终 predictor 直接相消；每个 covariance difference 变为 \(c^2B(\Sigma_e-\Sigma_{e'})B^\top\)，平方 Frobenius norm 变为原来的 \(c^4\) 倍，令 \(c\downarrow0\) 即得下确界 0。

## Gauge 条件

若加入 \(\|B\|_F=1\)、\(BB^\top=I\) 或其他 normalization，缩放退化被移除。但此时 induced cost 还依赖 representation fiber，不再是 risk quotient 的函数自动决定的量。

CORAL 观察：

\[
B(\Sigma_e-\Sigma_{e'})B^\top.
\]

均值变化和 conditional label mechanism 不进入 centered covariance，因此需要额外 state 才能分析它们。

## 结论

无 gauge：`degenerate_infimum`。有 gauge：`conditional representation-state cost`，不是单独的 risk-quotient cost；没有 unconditional CORAL-to-target-risk bridge。
