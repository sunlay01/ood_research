# Prompt：五日内机制区分 Probe

```text
某个 OOD/DG 候选机制与近邻工作接近，但可能仍有形式上的区别。请不要给出可发表性结论；只判断它是否值得做一个最多五个工作日、可明确失败的机制 probe。

先阅读 PROJECT_PROMPT_CN.md、现有 evidence register、literature matrix、最近 ADR 和 theory ledger。必须使用 $idea-to-experiment-harness 的 Stage P（Mechanism Probe Gate），并使用其指定的 ARS 角色边界；先刷新与 active/counterfactual probes、domain augmentation、worst-case DG、disagreement-based selection 的近邻检索。

仅当能完成下列六项时才输出 PROBE_AUTHORIZED：
1. 候选与最近工作在 target object、mechanism/update、objective、assumptions、central claim 上的一条形式化差别；
2. 一条可测的、双方预测相反的结果；
3. 明确 intervention、公开数据/模型、对照与指标；
4. fixed probe、random probe、最强 passive metric、proposal strength/mode 等去混淆对照；
5. 预注册的失败准则（失败要否定机制，而非仅否定某次实现）；
6. 不超过五个工作日的资源预算，并说明正/负结果各自会改变什么。

若不能满足，给出 PROBE_NOT_DIAGNOSTIC 或 PROBE_NEEDS_RETRIEVAL。写入 docs/experiments/00_experiment_ledger.md 和一个 ADR，明确：PROBE_AUTHORIZED 不是新颖性、实用性或顶会潜力的通过；任何正结果都必须回到 Stage 1 prior-art gate。
```
