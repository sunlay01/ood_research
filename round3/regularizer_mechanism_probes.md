# Regularizer mechanism probes

本轮只把 ERM、L2、IRMv1、CORAL 作为 probes。检查对象是是否存在：

\[
\Omega_j(f)\ \text{to}\ \|S_k(f)\|_{*,k}
\]

的 bridge。L2 目前只提供 global magnitude envelope；IRMv1 只在 relation-response family 下有条件 bridge；CORAL 仍需 representation gauge，不能直接推出 structural sensitivity control。没有 bridge 时记录 `NO MECHANISM-SENSITIVITY BRIDGE`。
