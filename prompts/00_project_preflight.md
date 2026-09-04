# Prompt：项目预检与任务路由

复制以下内容给 Codex，并在最后补充本次具体任务。

```text
我正在维护仓库 ood-representation-regularization。请先重新阅读：
1. PROJECT_PROMPT_CN.md
2. CONTEXT_MANAGEMENT.md
3. docs/research/00_project_brief.md
4. 与本次任务相关的 prompt 和 ledger。

然后不要立刻泛泛讨论。先用不超过 6 行给出：
- 当前研究阶段与本次任务类型；
- 已有的 VERIFIED / UNVERIFIED 输入；
- 将使用的已注册 skill 及其原因；
- 计划修改或新建的文件；
- 本次推进闸门与停止条件。

根据任务路由使用 skills：
- 文献、相似工作或新颖性：$idea-to-experiment-harness（并使用其 ARS 后端）；
- 精读、核验、文献矩阵：$ars-codex:academic-research-suite；
- 已通过闸门的本地机制实验：$local-experiment-validation-harness；
- 正式论文/审稿：$ars-codex:academic-research-suite 的对应 paper/reviewer 模式。

先检查当前环境中目标 skill 是否可用并阅读其 SKILL.md；若不可用，说明影响并采用最接近的有证据流程。除非我明确要求，不创建代理团队或并行子任务。

本次任务：<<在这里填写>>
```
