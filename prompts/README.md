# Prompt 索引

这些 prompt 不是把所有工作塞进一次对话，而是为不同阶段提供最小、可验收的任务边界。每次使用前先读取 `../PROJECT_PROMPT_CN.md`。

| 文件 | 何时使用 | 主要 skills | 成功产物 |
| --- | --- | --- | --- |
| [00_project_preflight.md](00_project_preflight.md) | 新会话、续接或普通项目任务开始前 | 按任务路由 | 已确认的阶段、输入、文件与闸门 |
| [01_novelty_and_literature_gate.md](01_novelty_and_literature_gate.md) | 找近邻工作、判断是否已做过 | `idea-to-experiment-harness` + `ars-codex` | Evidence Pack、六轴比较、先验艺术判定 |
| [02_theory_decomposition.md](02_theory_decomposition.md) | 推导一个明确的理论对象 | `ars-codex`（证据与反方审查） | 账本中的定义、假设、命题或反例 |
| [03_mechanism_probe.md](03_mechanism_probe.md) | 候选想法有区别但风险高 | `idea-to-experiment-harness` Stage P | `PROBE_AUTHORIZED` 或停止理由 |
| [04_local_experiment.md](04_local_experiment.md) | 做 ≤1 小时的可区分本地实验 | `local-experiment-validation-harness` + ARS | 预注册、可复现实验和严格判定 |
| [05_independent_review.md](05_independent_review.md) | 评估是否该继续、写作或投稿 | `ars-codex` reviewer | 独立审稿与 `PASS/REVISE/PIVOT/STOP` |

不应跳过 `01` 就开始追求“新 loss”，也不应把 `03` 的 `PROBE_AUTHORIZED` 写成新颖性已通过。
