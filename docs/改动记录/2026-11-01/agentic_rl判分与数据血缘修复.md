# agentic_rl 判分与数据血缘修复（M1）

需求来源：外部升级方案第 4 节（2026-11-01 经三个只读审计子代理逐条核验，三个 bug 断言动态复现成立）+ 审计补充发现两项。本记录的五个问题修复前全部先写失败回归用例（`tests/test_agentic_rl.py` 新增 `test_m1_*` 5 个），修复前 5 红、修复后 14 绿。

## 五个问题与修法

| # | 问题（修复前行为） | 位置 | 修法 | 回归用例 |
| --- | --- | --- | --- | --- |
| 1 | 推荐目录中真实合格但从未检索过的商品：`success=1`、`p0=1`、`reward=-1` 三者并存——统计口径与训练信号分裂 | `reward.py` reward_verified | `success` 默认口径改为**安全成功**（task_success 且无 P0）；`task_success` / `safe_success` 单列；`p0_reasons` 进入失败原因分类（evaluate 的 `failure_reasons` 合并统计） | `test_m1_unsearched_valid_product_is_task_success_but_not_safe` |
| 2 | 复核验证只看「追问后存在带 product_id/sku_id 参数的调用」：调用失败（ok=False）、目标不对齐（查了别的商品甚至不存在的 P9999）都算复核通过 | `reward.py` 原 `:103-107` | 新 `_recheck_ok`：调用须成功（`c.ok`）、目标对齐（目标=追问点名的商品，追问不点名时=第一次作答推荐过的商品）、结果命中目标；时序门槛（追问之后）保留 | `test_m1_verify_recheck_needs_success_and_target_alignment`（失败调用 / 目标错位 / 合格三例） |
| 3 | 超限轨迹截断结果无人消费：导出与 token 统计仍取原始 Episode 全量消息（构造样本 70036 token 超 16384 仍入库），`stats.json` 系统性高估 | `filters.py` clean、`__main__.py` 导出、`trajectory.py` to_record | 红线 2 改为**整条剔除**（`too_long` 计数）——本实现的截断必丢终局作答，示范不完整（对齐升级方案 6.2「首版优先剔除」）；删除死代码 `truncate()`；新增 `filters.final_messages` 作为导出/统计消费消息的统一入口，`to_record` 支持传入 messages | `test_m1_overlong_trajectory_is_dropped_not_exported` |
| 4 | 幻觉与泄露只查最后一次作答：changed/verify 场景第一次作答里的幻觉商品 ID 不受罚（审计补充发现） | `reward.py` 原 `final_text` 单点 | 新 `answer_texts`：fabricated / leak 检查覆盖**每一次**作答；任务层「推荐」判定仍以终答为准 | `test_m1_first_answer_hallucination_is_punished` |
| 5 | `seen_ids` 汇总整条轨迹，无证据边界：changed 场景改条件前的检索结果对改条件后的作答仍是「有效证据」 | `reward.py` seen_ids | changed 场景以买家改条件的追问为界：界后作答只认界后检索；界前检索被界后作答引用记 `stale_evidence`（计入 P0） | `test_m1_changed_condition_requires_research` |

## 关键源码

- `agentic_rl/reward.py`：`answer_texts`、`_user_turn_of`、`seen_ids(min_user_turn, max_user_turn)`、`_recheck_ok`、重写后的 `reward_verified`（返回新增 `task_success` / `safe_success` / `p0_reasons` / `stale_evidence` 字段）。
- `agentic_rl/evaluate.py`：`summarize` 增 `task_success` 与 `by_scenario_task` 口径；`failure_reasons` 并入 `p0_reasons`；`markdown_table` 首列改「安全成功 / 任务成功」双列。
- `agentic_rl/filters.py`：红线 2 改剔除；`final_messages`。
- `agentic_rl/trajectory.py`：`to_record(..., messages=None)`。
- `agentic_rl/__main__.py`：SFT 导出与 token 统计统一走 `filters.final_messages`。

## 验证方式与证据

1. 回归：`tests/test_agentic_rl.py` 修复前 5 failed / 9 passed，修复后 **14 passed**（Windows、Python 3.11，8.8 秒）。
2. 一致性自检：`python -m agentic_rl check` → OK（推演产物与文档关键数字全部一致，未受影响）。
3. 范围检查：`python -m scripts.eval.harness impact --files agentic_rl/*.py tests/test_agentic_rl.py` → `harness_changed: false`。
4. 基线对照：修复后 `python -m agentic_rl real --out eval/verification/agentic-rl-m1-20261101`（249 秒，0 次模型调用）。与历史基线逐项对照：
   - 12 个策略×切分的 `task_success` 与历史 `success` **逐位相等**（任务层逻辑零扰动）；
   - 所有规则策略 safe == task（`p0_rate` 全 0.0）——修复不冤枉诚实轨迹；
   - SFT 数据 `input 320 / kept 319 / dropped {score:1} / 场景分布 / tokens_mean 3410.6 / tokens_max 6123` **全部一致**（历史数据从未触发截断，红线 2 语义变化零影响）；
   - `verify_compliance` 不变（规则策略的复核本就是成功的精确检索）。

## 未完成项与边界

- 全量测试套件（1796 个）本机未跑：Windows 首跑基线未建立，本次改动经 impact 判定不在 Harness 与应用代码范围内；`tests/test_agentic_rl.py` 14/14 为与变更直接相关的测试。
- 动态价格/库存下的 `evidence_epoch` / `state_version` 属 M2（静态目录下已按「改条件后强制重新检索」合同实现）。
- 模型策略（ModelPolicy）下的实际拦截效果未验证（无模型调用授权）；当前证据仅覆盖规则策略零误伤 + 回归用例拦截能力。

## 启用与回滚

- 行为开关：无配置开关；`success` 字段语义变化是本次唯一破坏性变更（消费者：`evaluate.summarize`、`filters.clean` 评分门，二者已同步）。旧口径可用 `task_success` 字段完整还原。
- 回滚：还原本目录涉及文件至 `288366e` 即可；新证据目录 `agentic-rl-m1-20261101/` 与历史证据互不依赖。
