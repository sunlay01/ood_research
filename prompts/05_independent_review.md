# Prompt：独立严格审稿闸门

```text
请对当前候选工作做独立、挑剔的研究审稿。目标是决定是否应继续，不是帮助项目显得有价值。

先阅读 PROJECT_PROMPT_CN.md 和最新的 evidence register、literature matrix、ADR、theory ledger、experiment ledger。使用 $ars-codex:academic-research-suite 的 academic-paper-reviewer methodology-focus 模式；按其角色隔离要求，不要让审稿结论继承先前“想让项目通过”的叙述。

审稿包只应包含：当前 RQ、一个句子的 novelty claim、核验过的 evidence pack、六轴近邻比较、精确假设/定理状态、实验预注册与结果。逐项评价：
- 是否已被相同或 close-equivalent 工作覆盖；
- 定义和定理是否避免循环论证、不可辨识性与隐藏强假设；
- 实验是否真能区分机制，是否有必要的 baseline、反事实和负对照；
- 正则路径、有效秩或 OOD accuracy 是否被过度解释；
- 可行性、复现性、统计有效性和顶会级意义。

先写 Findings，再给出恰好一个结论：PASS、REVISE、PIVOT 或 STOP；同时给出 TOP_TIER_PLAUSIBLE、BORDERLINE_TOP_TIER、LOWER_TIER_ONLY 或 NOT_PUBLISHABLE_AS_RESEARCH 的 venue-fit 标签。只有证据充分、近邻差异可辩护、结论可检验且无致命问题时才可 PASS。REVISE 只能指定一个主要、可修复的缺陷和对应回流阶段；已做过、缺乏新颖性、不可行或低价值时必须 STOP。

将审稿意见写入 docs/decisions/ADR-XXX-independent-review.md，并明确下一闸门或停止理由。
```
