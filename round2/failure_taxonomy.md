# Failure taxonomy

## Source-unidentified

\[
[D]\neq0\text{ in }\mathcal A_{S,\tau}.
\]

target 能看到，但 source information 不足。

## Regularizer-blind

source quotient 已能区分方向，但某个 \(\bar\Omega_j\) 沿该 target-harmful direction flat 或曲率不足。它是算法没有利用已有 source information。

## Destructive

存在 target-good / target-bad states，但 induced cost 偏好 target-bad state；或局部有 \(DR_T[d]<0\) 而 \(D\bar\Omega_j[d]>0\)。

## Quotient-external degeneracy

表示缩放、gauge 或 statistical complexity 影响无法由 risk quotient 表达时，单独标注 `quotient-external/parameterization-degeneracy`，不强行塞入前三类。
