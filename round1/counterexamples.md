# 反例目录

## 反例 1：source kernel

定理 5 给出一般构造：source-null state 在 source 中不可区分，而 target moment 若不在 source span 内可以检验该差异。

## 反例 2：IRMv1 两 relation blind branch

在当前代码的标量 SCM 中，relation 为 \(0.7,-0.1\)，噪声方差为 \(0.02\)。得到 predictor

\[
w=(0,0.7384168123,-0.7947227078),
\]

其 IRMv1 penalty 约为 \(2.6\times10^{-32}\)，source risk 为 \(0.51\)，但 relation 为 \(-1\) 的 task-preserving target 上 risk 为 \(1.1000\)。这不是“IRM 一定失败”的结论，而是两点 source design 无法识别二次 response 的严格盲区。

## 反例 3：CORAL mean/conditional blind

centered covariance 对 location shift 不变；因此 CORAL 可以观察为零，而含 nuisance use 的 predictor 的风险仍随均值变化。类似地，改变条件标签机制不必改变 representation covariance。

## 反例 4：L2 非选择性

两个 predictor 可以具有相同或相近整体范数，却把范数分配到 task 与 nuisance 的方式不同；在 nuisance shift 下 target risk 因此不同。要把该反例写成严格等范数构造，需要固定设计与参数约束，不能仅凭一次数值结果宣称。
