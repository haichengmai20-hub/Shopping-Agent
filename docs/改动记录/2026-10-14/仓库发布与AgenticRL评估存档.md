# 2026-10-14 仓库发布与 Agentic RL 评估存档

**当前状态**：文档与版本管理改动已完成并提交；**未运行**评测、未启动训练、未做真实模型链路验证。

**关联代码版本**：合并上游 `origin/main` @ `6fdc9be` 后的 `main`；本地新增提交 `afa0570`、`972b730`、`5aa9336`、`e11d744`。发布目标仓库 `haichengmai20-hub/Shopping-Agent`。

## 问题或需求

用户要把本地 `D:\RL及harness\globex-agent` 发布到自己的 GitHub 仓库，并附上一轮关于「这个项目用来做 Agentic RL 合不合适、是有参数训练的 RL 还是 Agent Skill 自进化」的问答，让其他 AI 和面试官知道这个仓库被用来做什么。

发布前存在两个状态问题：

1. 本地 `main` 比 `origin/main` 落后 7 个提交，同时还压着 4 个已改文件和一批未跟踪的学习材料，直接推送会丢掉上游功能。
2. 上游 `f10ed0e` 以「设计文档、改动记录与图解仅在本地维护」为由**删除了全部 58 个 `docs/` 文件**并加入 `/docs/` 忽略规则，而本地的学习材料正放在 `docs/` 下。

## 本次变更

### 1. 本地改动先落盘（推送前）

- `afa0570`：`app/infrastructure/persistence/json_file_stores.py` 的 `fcntl` 文件锁改为 `portalocker`（Windows 无 `fcntl`），同步 `pyproject.toml`、`uv.lock`。这是 `2026-09-25` 留下的未提交改动，本次一并提交。
- `972b730`：归档 `docs/学习路线/`（5 篇）、`docs/改动记录/2026-09-25~2026-10-07/`、对应 `eval/verification/stage*-learning-*` 与 `windows-setup-20260925`，以及 `docxs/` 教程原件，并更新 `docs/改动记录/README.md` 日期索引。

### 2. 合并上游

- `5aa9336`：合并 `origin/main`（7 个提交）。68 个文件修改、94 个新增、58 个删除，只有 `docs/改动记录/README.md` 出现 modify/delete 冲突，按「保留本地索引」解决。
- `json_file_stores.py`、`pyproject.toml`、`uv.lock` 自动合并成功，合并后 `portalocker` 修复仍在（已复核）。

### 3. 保留文档（与上游的显式差异）

- `e11d744`：从 `8cbd4d6` 恢复上游删除的 57 个 `docs/` 文档；`.gitignore` 中把 `/docs/` 改为 `!/docs/` 并写明差异来源与还原方式。

### 4. 新增「其他 AI 须知」

- 新增 `docs/项目定位与其他AI须知.md`：保存完整的 Agentic RL 可行性评估问答原文，附「证据锚点」表（按当前代码复核的文件与行号）、与上游的已知差异、阅读注意事项。
- `AGENTS.md` 顶部新增「先读：本仓库的用途」，`README.md` 首段后新增同义提示，均指向该文档。
- 文档明确边界：**当前是 Agent 应用 + 评测 + 离线 GEPA 文本优化，没有参数训练链路**，不能按已完成的 SFT／RL 项目描述。

## 验证与证据

| 检查项 | 方式 | 结果 |
| --- | --- | --- |
| 待推送内容无凭据 | `git ls-files` 确认 `.env` 未跟踪；对 `docs/`、`eval/verification/` 正则扫描密钥样式 | 无匹配（唯一命中是 `tests/test_harness_guards.py` 的示例串） |
| 合并后文件锁修复仍在 | 检索 `json_file_stores.py`、`pyproject.toml`、`uv.lock` 中的 `portalocker` | 三处均在 |
| 文档恢复完整 | `git ls-files docs` 计数、比对 `8cbd4d6` 与 `HEAD` 的 docs 清单 | 75 个文件，无缺失 |
| 评估文档引用可核 | 逐个复核 `scripts/optimize_prompts.py`、`llm.py`、`catalog_search.py`、`tracing.py`、`eval_regression.py`、`gepa.json` 的对应分支 | 全部相符，行号已更新为合并后位置 |
| 推送结果 | `git ls-remote` 与远端文件树核对 | 见当日总览与提交记录 |

**未验证**：上游合并进来的代码改动在本机**没有跑过** `make eval-harness`、没有启动服务、没有真实模型调用；这些改动的验证状态以上游仓库为准。`pytest` 与前端测试本次均未执行。

## 遗留问题与后续

- 上游 7 个提交里的 Harness 评测套件尚未在本机跑过，需要模型凭据；有凭据后建议先跑确定性安全契约档位。
- `docs/项目定位与其他AI须知.md` 中的结论有明确前提：如果后续真的新增训练入口、奖励服务或权重产物，必须更新该文档，不能沿用「没有参数训练」的结论。
- 本仓库与上游将长期存在 `docs/` 与 `.gitignore` 的差异，下次从上游拉取时需注意不要再让 `/docs/` 忽略规则覆盖。
- 训练方向本身仍未开工：任务生成器、可重置环境、轨迹与奖励服务、SFT／RL 对照实验都还只是评估文档里的建议。
