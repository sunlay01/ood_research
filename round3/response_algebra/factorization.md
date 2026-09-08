# 3A 因子化

对 (win\mathbb R^5)，令

\[
\phi(w)=(\operatorname{svec}(ww^\top),-2w,1),
\qquad
\psi(T)=(\operatorname{svec}(\Delta M_{XX}),\Delta m_{XY},\Delta m_{Y^2}).
\]

`svec` 复制对角元素，并将严格上三角元素乘以 \\(\sqrt 2\\)，所以

\[
\langle A,B\rangle_F=\operatorname{svec}(A)^\top\operatorname{svec}(B).
\]

因此平方风险 transport 满足 exact identity：

\[
R_T(w)-R_S(w)=\phi(w)^\top\psi(T).
\]

对模型和 shift 逐行堆叠即得

\[
\mathsf R=\Phi\Psi^\top.
\]

代码 residual 只用于数值核验，不替代理论恒等式。
