# L2：全局范数与非选择性收缩

## 定理 9：generic moment bound

在定理 3 的线性模型中，若 \(M_T,M_S\) 有限，则

\[
|R_T-R_S|
=|v^\top(M_T-M_S)v|
\leq\|M_T-M_S\|_{op}\|v\|_2^2.
\]

若参数化固定且 \(\|v\|\leq c_0+c_1\|w\|\)，才有

\[
|R_T-R_S|
\leq\|\Delta M\|_{op}(c_0+c_1\sqrt{\Omega_{L2}})^2.
\]

### 证明

第一步是对称矩阵二次型的谱范数不等式；第二步代入假设。注意 \(\|\beta\|\) 和参数化是额外条件。

## 结论与反例边界

L2 同时压缩 task coefficient 与 nuisance coefficient，不能从目标函数本身区分二者，因此没有 shift-specific bridge。它可以在已知全局 moment budget 下限制整体风险放大，但不能说明某一类 OOD 变化被控制。

对表示 \(Z=BU\)、head \(w_Z\)，变换 \(B\mapsto TB\)、\(w_Z\mapsto T^{-T}w_Z\) 保持 predictor 和 target risk 不变；但原始 encoder/head 范数一般改变。因此原始 L2 数值不是坐标不变的 OOD state。
