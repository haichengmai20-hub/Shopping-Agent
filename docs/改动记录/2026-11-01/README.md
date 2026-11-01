# 2026-11-01 改动记录

## 当日总览

按外部升级方案完成 M1（可靠性）与 M2（E1 状态化环境与任务准入）。M1 修复判分与数据血缘五个已核验问题，四个规则策略基线逐位不变；M2 交付隔离的下单环境、任务准入与全套验收证据。

| 功能 | 记录 | 状态 |
| --- | --- | --- |
| agentic_rl 判分与数据血缘修复（M1） | [agentic_rl判分与数据血缘修复](agentic_rl判分与数据血缘修复.md) | 回归 5 红→全绿；`agentic_rl check` 通过；基线对照逐位相等（证据 [agentic-rl-m1-20261101](../../eval/verification/agentic-rl-m1-20261101/)）；已提交 `f4b7e26` |
| E1 状态化下单环境与任务准入（M2） | [E1状态化下单环境与任务准入](E1状态化下单环境与任务准入.md) | 单测 22/22；准入 72/72 见证通过；重放 120/120 一致；8 副本并发隔离（证据 [agentic-rl-m2-20261101](../../eval/verification/agentic-rl-m2-20261101/)） |

## 共享边界

- 只改 `agentic_rl/`（reward / evaluate / filters / trajectory / `__main__`）与 `tests/test_agentic_rl.py`；`app/` 一行未动。`scripts.eval.harness impact` 判定 `harness_changed: false`，无需付费评测。
- 修复前基线 [agentic-rl-env-20261019](../../eval/verification/agentic-rl-env-20261019/)（历史证据，未改动）；修复后基线 [agentic-rl-m1-20261101](../../eval/verification/agentic-rl-m1-20261101/)（249 秒，0 次模型调用）。12 个策略×切分的任务层成功率逐位相等，SFT 319 条与 token 统计逐位一致。
- 推演产物 `agentic_rl/runs/` 未重生成，一致性自检 `python -m agentic_rl check` 通过（该检查不重算奖励，历史推演数字不受影响）。
- 学习资料同步：06 三处口径加注、`CLAUDE.md` 第 8 节补 M1 条目，见[学习改动记录](../../学习路线/学习改动记录.md)当日行。

## 关联版本

- 修复基线 HEAD `288366e`；本次改动截至记录时未提交（等用户确认后 commit）。
