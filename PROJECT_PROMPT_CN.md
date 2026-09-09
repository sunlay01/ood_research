# 项目协作提示词

本文件只保留稳定协作规则，不再记录当前科学状态。当前状态必须读取：

- `AGENTS.md`
- `CURRENT_STATE.md`

## 启动规则

1. 不要从全仓搜索或旧报告考古开始。
2. 先读 `AGENTS.md` 和 `CURRENT_STATE.md`。
3. 只有存在 `active/TASK.md` / `active/CONTEXT.md` 时，才把它们作为当前任务上下文。
4. 只有被 `CURRENT_STATE.md`、registry 或 active task 指向时，才读取历史报告。
5. 打开历史文件前记录一行 `REOPEN_REASON:`。

## 写入规则

- `state_write_authorized: true` 时，任务可以更新 canonical state，并必须记录改变。
- `state_write_authorized: false` 时，只能把建议写入 `active/STATE_DELTA.md`。
- 不删除、不移动历史研究目录。
- 不恢复、不总结、不引用已删除的 `MECH-001/C011` 实质内容。

## 研究纪律

- 当前科学对象、阶段、冻结结论和下一步只以 `CURRENT_STATE.md` 为准。
- 旧 semantic-latent、Round-1、Round-2 和早期 Round-3 文件是历史证据，不是默认 authority。
- 不把 source-response recovery、local affine regret 或 CMNIST bridge 解释为 causal identification、finite-sample guarantee、target-risk lower bound 或 universal DG theorem。
