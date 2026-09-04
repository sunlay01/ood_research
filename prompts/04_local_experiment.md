# Prompt：本地机制验证实验

```text
请把一个已通过新颖性闸门的候选理论/机制，转化为一次不超过一小时、只用本地计算的可证伪实验。不要先扩展为完整 benchmark，也不要把运行成功当作证明。

先读取 PROJECT_PROMPT_CN.md、docs/research 中的 verdict/evidence、docs/theory/00_theory_ledger.md、docs/experiments/00_experiment_ledger.md。只有下列任一条件满足才继续：
- prior-art verdict 为 DISTINCT_BUT_RISKY 或 CLEAR_NOVELTY_GAP；
- 有完整的 PROBE_AUTHORIZED（含近邻比较、区分预测、对照和失败准则）。
否则回到 $idea-to-experiment-harness 的 prior-art gate。

必须使用 $local-experiment-validation-harness，并遵守其 Stage 0–6B 结构和 ARS coordination：
1. 先执行 local feasibility triage；若不可能在一小时内产生可解释证据，停止，不写代码。
2. 使用 $ars-codex:academic-research-suite 的 experiment-agent plan mode，预注册 RQ、假设、独立/因变量、生成机制、baseline、candidate、随机种子、指标、成功/失败/无结论阈值。
3. 仅实现一个干净的最小生成模型、一个最强近邻 baseline 和一个候选机制。配置、种子、命令和输出路径必须可复现。
4. 先做独立 code audit，再 smoke test 和有界主运行。主运行时记录资源使用与原始指标；不吞异常、不以重试掩盖根因。
5. 使用 experiment-agent validate 进行指标/统计/复现检查，并使用独立 reviewer 给出 `PASS_LOCAL_SIGNAL`、`REVISE_AND_RERUN_ONCE`、`PIVOT`、`STOP` 或 `INCONCLUSIVE_STOP`。

把预注册、命令、配置哈希、结果、失败和替代解释追加到 docs/experiments/00_experiment_ledger.md；代码进入 src/，测试进入 tests/，可复现实验配置进入 configs/。本地输出进入 artifacts/ 且不提交。任何正信号都只能作为证据，之后仍须回到 $idea-to-experiment-harness 的新颖性/审稿闸门。
```
