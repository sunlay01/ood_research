# Prompt：文献检索与新颖性闸门

```text
请为本仓库执行“文献与新颖性闸门”，目标不是支持一个预设结论，而是判断这个研究方向是否已被相同或等价工作覆盖。

先阅读 PROJECT_PROMPT_CN.md、CONTEXT_MANAGEMENT.md、docs/research/00_project_brief.md、docs/research/01_evidence_register.md 和 docs/research/02_literature_matrix.md。

必须使用 $idea-to-experiment-harness，并遵守其 Required ARS-Codex Coordination：
1. 若课题范围仍不足以比较，先完成最小的 Stage 0 intake；不要把问题扩展成论文大纲。
2. 执行 Stage 1 的 prior-art gate，使用 $ars-codex:academic-research-suite 的 deep-research three-way-scan 路线。检索按语义邻近而非仅按关键词，至少覆盖：representation risk decomposition、hybrid / counterfactual decomposition、nested oracle error、IRM objective reinterpretation、gradient/Hessian/moment alignment、regularization path 和 feature-mode dynamics。
3. 对最接近工作做六轴比较：problem/task、mechanism/method、objective/training signal、assumptions、evaluation、central claim。单纯换数据集、backbone 或 penalty 系数不构成新颖性。
4. 对每项文献核验原始链接、版本和可支持的精确主张；把聊天讨论中的论文名一律视为 UNVERIFIED 种子。
5. 先输出 Reasons to stop now，再输出 Possible route to continue。只在有检索证据时给出以下判定之一：EXACT_ALREADY_DONE、CLOSE_EQUIVALENT、CROWDED_INCREMENTAL、DISTINCT_BUT_RISKY、CLEAR_NOVELTY_GAP、INSUFFICIENT_PRIOR_ART_EVIDENCE。

写回仓库：
- 更新 docs/research/01_evidence_register.md（含链接、来源类型、核验状态和支持范围）；
- 更新 docs/research/02_literature_matrix.md；
- 新建 docs/decisions/ADR-001-prior-art-gate.md，记录当前判定、最强反证和下一闸门。

不要提出新方法或编造 theorem。若结果为 EXACT_ALREADY_DONE、CLOSE_EQUIVALENT 或无法建立可核验的 gap，停止并解释应当如何转向。
```
