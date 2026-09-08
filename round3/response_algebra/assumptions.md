# 3A 假设与边界

本子课题只研究固定表示 (X=(1,C,A)) 下的 population squared-loss raw risk response。主数据来自第三轮补充实验的 660 个已训练线性 predictor 与 300 个 shift probes。

本轮使用 quadratic-risk lifting，不把 lifting feature、response subspace 或 quotient 解释成生成机制。clustering、response-order、mechanism semantics 和 regularizer control 均延期。

所有 numerical rank 使用同一个相对奇异值阈值；结果同时保存 tolerance profile。分组只用于线性代数审计，不代表 semantic label。
