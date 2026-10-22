# CLAUDE.md — Shopping-Agent（Globex 学习与 Agentic RL 推演仓库）

写给在本仓库工作的 Claude 会话。仓库自带的工程协作约定在 AGENTS.md，Claude Code 不会自动读取它，这里导入：@AGENTS.md

本文件把用户在 Code-R1-reaserch（`/Users/s11437/Code-R1-reaserch`，RLVR 学习仓库）形成的工作方式迁到本仓库（2026-10-19）。冲突时：工程代码改动服从 AGENTS.md，学习资料的写法服从本文件。

## 1. 这个仓库是什么

- Globex：AgentScope 2.0.8 + FastAPI + AG-UI + React 的跨境电商选购 Agent。上游 `pao-coder66/globex-agent`；2026-10-14 合并上游 `6fdc9be` 后发布到 `https://github.com/haichengmai20-hub/Shopping-Agent`（分支 `main`，2026-10-19 拉取时 HEAD 为 `1672fe4`）。与上游的差异：保留了上游删除的 `docs/`，Windows 文件锁改用 portalocker。
- 当前能力是 **Agent 应用 + 评测 + 离线 Prompt/Skill 优化（GEPA）**，**没有参数训练链路**：全仓没有 torch、trl、peft、verl 等训练依赖或 trainer；`learning_store` 导出的训练样本写死 `trainer_started=False`。不要把它描述成已完成 SFT/RL 的项目。
- 学习材料：
  - `docs/学习路线/` 01–05：按 docxs 导读的 Step 1–5 写成，2026-09-26 至 10-07 完成，**源码基线是 `8cbd4d6`**。
  - `docxs/`：13 份 .docx 教程原件，**只读**。
  - [项目定位与其他AI须知](docs/项目定位与其他AI须知.md)：AI 生成的 Agentic RL 可行性评估与推荐路线，是训练推演的出发点。
- 工作重点是学习资料与训练推演。Globex 源码是学习对象和推演用的环境底座，不是开发对象。

## 2. 工作目标与协作约定

### 目标（用户 2026-10-19 确定）

1. **先学懂 Globex**：按 `docs/学习路线/` 01–05 学。
2. **再推演一次 Agentic RL 训练实验**：做法同 Code-R1 的 `05_实跑推演_Qwen3-8B.md`。配置、日志、曲线、评测、归因等产物的形态对齐真实实验，数字来自推演假设，不是实跑。
3. **最后用推演的收获反过来补学习资料**，必要时按下文的「反向补充」补代码。

用户在备考业务算法岗（后训练与后训练数据构造），面试约在 2026-12 至 2027-01。讲解重心放在与后训练相关的部分：轨迹与工具调用、评测与奖励、数据构造、可重置的环境。前端与部署细节按需讲。

### 资料的优先级（沿用 Code-R1，冲突时高优先级胜出）

1. **懂机制**：Agent 每个环节为什么这样做，改一个条件会怎样。
2. **能过面试**：每个机制能用四句口述（机制 → Globex 怎样实现 → 一个反例或取舍 → 当前证据能证明到哪里）。
3. **易读易记、便于口述**：先给结论和人话，再给源码定位；一个机制配一句定义、一个例子、一个反例。
4. **最后才是与源码逐字贴合**：允许简化，标一句「简化」即可。

底线：推断写推断，实跑写实跑，推演写推演，不伪造运行证据。

### 协作约定

- 用中文。源码定位写「文件 + 函数（+ 行号）」，行号以当前 HEAD 为准；学习路线正文按 `8cbd4d6` 写成，引用前先复核。
- **证据分五类**，写进资料时标明：读代码；免费本地运行（单测、确定性契约、`--dry-run`，不调模型）；真实模型运行（花钱）；效果验收（固定协议加配对基线，见 AGENTS.md）；推演（只用于练决策，不能当实跑证据）。测试通过不等于外部链路可用，更不等于有效果收益。
- **费用**：用户 2026-10-19 决定「先只读代码，不花钱」。下列操作都会调用模型或 embedding，每次都要先问用户：
  - `make eval-harness` / `eval-harness-dev` / `eval-harness-release`（`eval-harness-contracts` 不调模型）；
  - `scripts/optimize_prompts.py`、`run_learning`，以及任何 `--mode model`、`--with-model`；
  - 启动后端（首次会给商品和知识建向量索引；换新的 `DATA_DIR` 会重新 embedding）和真实页面对话。

  不读、不打印 `.env`。
- **角色**：用户是学习者，Claude 默认是导师兼资料维护者。作业（画图、问答）先由用户独立完成，再核对讲解；用户明确要答案时直接给。
- **每学一篇的固定流程**（沿用 Code-R1 2026-10-14 的做法；用户 2026-10-19 再次确认：先建 Shopping Agent 的纵览并持续维护，之后每篇先建总览，再写文字说明。[学习路线 README](docs/学习路线/README.md) 的「讲解约定」已同步）：
  1. 先在对话里给三段讲解：纵览（Globex 一次请求的七站，以及 01–05 和之后的训练推演各在哪）、本篇总览（在流程里做什么、长什么样）、本篇文字版导读。
  2. 写入前，逐条对照 **HEAD 源码和资料的最新状态** 核对数字与说法，能免费跑的就跑。旧的、错的不记；发现的旧说法顺手修掉，并同步所有引用处。
  3. 核对后写入：
     - 纵览写进学习路线 README 的 `overview-all`（已有就按最新状态更新），同时更新同页的学习进度表 `progress`。
     - 总览和导读写进该篇开头的 `overview-N`、`guide-N`，原有的开头说明并进总览。导读每节末尾链接到正文对应小节，正文小节要补显式锚点。
     - 最后在学习改动记录里记一行。
- **全局视角**（用户 2026-10-19 要求，「不要陷入局部的牛角尖」）：
  - 每篇总览按固定顺序写：先讲这一篇在整个系统里的位置，再讲要回答的问题和答案，然后是「放到训练推演里看」，最后是全篇结构、用时与边界、源码依据、前后衔接。
  - 导读讲机制、取舍和边界，只保留学习者必须认识的名字。行号和函数级证据只写在本文件，不进学习文本。
  - 每篇末尾有「60 秒口述与面试追问」：口述从整个项目讲到这一层，追问至少一条和训练有关。
  - 写完要派一个对抗性子 agent 独立核验并修正，主会话审它的改动、复跑检查、记进学习改动记录。
- **全局一致**：改任何一篇时，同时对照其他各篇、README 和项目定位文档的术语、例子、数字、命令与互链，冲突就地统一并记一行。同一内容只在一处详写，其余地方放链接。术语用代码里的真名（`product_search_tool`、`task_dispatch`、`price_max_major`、`landed_price` 等）；docxs 里的 `product_search`、`price_compare`、`item_picker`、`item_id` 在代码中不存在，引用时要注明。
- **贯穿例子只有一个：「轻便背包、寄中国、300 元」**（用户 2026-10-19 定），原句是 01 的「帮我找一件轻便背包，寄到中国，总到手价不超过 300 元人民币。」所有机制例子、作业和反例都用它。定义、允许的变化、例外和逐处修改记录都在 [贯穿例子维护记录](docs/学习路线/贯穿例子维护记录.md)，这里不重复。
  - **改商品例子必须用这份文档维护记录**（用户 2026-10-19 要求）：新增、更换或调整任何一处商品例子，先在那里登记变化或例外，并逐处写清阶段、位置、原来和现在；再在学习改动记录记一行，链接过去。
  - 预算口径一律写清：代码按 `price_max_major` 过滤单件价，`landed_price` 只按数量 1 报价，不保证「总到手价」。
- **反向补充**（沿用 Code-R1 2026-10-14 的原则）：资料（含 docxs 08 系列的 SFT/RL 设计）覆盖多于源码、并且设计靠谱时，写成代码补进来，资料再讲「原版 vs 补强版」。本仓库的约束：
  - 新代码放平行目录，不改 `app/` 现有文件；确需加接入点时只增不改，并先问用户。
  - 动手前用 `uv run python -m scripts.eval.harness impact --files <路径...>` 确认改动不落在 Harness 范围内，否则要跑花钱的 `make eval-harness`。
  - CPU 可验证即算完成。补哪些、放哪里、叫什么，学到时先和用户确认方案。
- **推演规模**（沿用 Code-R1 2026-10-16 的要求）：保持个人研究量级。单机 4×A800 或 4×A100 80GB，按约 ¥10/卡时估算，单轮几千元以内；贵的选项要标出代价，并给更省的替代。
- **两套改动记录**（用户 2026-10-19 决定）：学习资料、本文件和推演产物的改动记进 [学习改动记录](docs/学习路线/学习改动记录.md)，一行一条；2026-10-07 及以前的学习记录留在 `docs/改动记录/` 原处。工程代码的改动仍按 AGENTS.md 写进 `docs/改动记录/YYYY-MM-DD/功能点.md`。
- **落盘检查**（本仓库 stage1–5 的做法，加上 Code-R1 的链接复查）：
  - 从磁盘读回，确认 UTF-8 无 BOM。
  - 代码围栏成对：按 CommonMark 识别，不要「行首三个反引号就切换」。
  - 相对链接和锚点全部可达；新增锚点用显式 `<a id="...">`。
  - docxs 13 份的 SHA-256 写前写后一致。
  - `git diff --check` 退出 0。

  检查结果记进学习改动记录的那一行。
- **git**：只在用户要求时 commit 或 push；提交前用 `git status` 确认只有预期的文件；commit message 末尾按当前会话提示加 `Co-Authored-By` 行。

## 3. 本机环境（2026-10-19 实测，macOS arm64）

| 项目 | 状态 |
|---|---|
| Python | 仓库要求 `>=3.11,<3.14`，`.python-version` 是 3.11。系统 `python3` 是 3.9.6，不要用；`uv sync` 已按 `.python-version` 装好 3.11.17，统一用 `uv run` 或 `.venv/bin/python` |
| 依赖 | `uv sync --frozen` 已完成（2026-10-19，约 23 秒）：agentscope 2.0.8、fastapi 0.141.1、pytest 9.1.1 等。可选依赖 transformers、fastembed、gepa 没装 |
| Node.js / npm | 没装，前端起不来；学后端不需要 |
| `.env` | 不存在，只有 `.env.example`；shell 里也没有任何模型密钥变量。默认模型是 DashScope 的 `qwen3-max`，回退 `qwen-plus`，embedding 用 `text-embedding-v4` |
| 已跑过的验证（2026-10-19） | 全量单测：1796 通过、44 跳过、0 失败，94 秒。跳过的 43 个需要本机 `redis-server`（可设 `GLOBEX_REDIS_SERVER_BIN`），1 个缺 `gepa`。Harness 契约：838 通过，0 失败、0 跳过，证据在 `eval/verification/harness-local-20261019-153048/` |
| 测试时的密钥 | `app/presentation/server.py:359` 在模块级执行 `app = build_app()`，加载配置时要求 `LLM_API_KEY` 非空。所以在没有密钥的机器上，README:278 的测试命令收集 `tests/test_queue_api.py` 时就报「未配置 LLM_API_KEY，无法启动」并中断（2026-10-19 实测），要设一个明显无效的占位值。契约档按 CI 的做法用空值即可 |

## 4. 免费命令（不调模型；单测和契约 2026-10-19 已在本机跑通）

```bash
cd /Users/s11437/Shopping-Agent
uv sync --frozen    # 下载 Python 3.11 和依赖，不花模型的钱

# 全部单测，约 95 秒（README:278 的命令加占位密钥；测试用桩模型和 MockTransport）
LLM_API_KEY='placeholder-not-a-real-key' LANGFUSE_BASE_URL='' LANGFUSE_PUBLIC_KEY='' LANGFUSE_SECRET_KEY='' uv run python -m pytest -q

# Harness 确定性安全契约，约 16 秒：suite.json 写明「不调用模型」；输出目录默认在 eval/verification/ 下按时间戳新建
LLM_API_KEY='' make eval-harness-contracts
uv run python -m scripts.eval.harness impact --files <路径...>   # 只做路径匹配，判断是否落在 Harness 范围内

# 零外部调用的数据与评测检查（出处：docs/正式评测选集与证据清单.md）。--report-dir 必须用新目录
uv run python scripts/eval/validate_datasets.py
uv run python scripts/eval_regression.py --cases eval/v1/agent_cases.yaml --split release --dry-run --report-dir <新目录>
```

macOS 没有 GNU 的 `timeout` 命令，不要在命令前加它（2026-10-19 踩过，命令会直接以 127 退出）。

两个会改写已有证据的陷阱：
- 评测清单文档里的示例命令把 `--report-dir` 写成历史证据目录 `eval/verification/split-20260909`，照抄会覆盖历史证据。
- `python -m scripts.eval.harness verify <目录>` 会重写该目录的 `summary.json` 和 `report.html`，只对自己新跑出的目录用。

## 5. 学习材料的结构

| 路径 | 内容 |
|---|---|
| `docs/学习路线/README.md` | 首页（2026-10-19 重组）：项目全景 `project-map`（产品是什么、四块分层、页面功能挂在哪、没有训练）；纵览 `overview-all`（七站的进出与所在阶段、五个阶段各答什么、训练推演对应表 `training-map`、上游新增补在哪、本机能跑的 8 份单测、全程图）；学习进度 `progress`；讲解约定（含全局视角写法）。docxs 不改 |
| `01_项目地图与前后端流程.md` | Model / Agent / AgentLoop / Harness 的分工，服务启动和一次请求两条线；对应导读 Step 1，30 分钟。2026-10-19 开头加了总览 `overview-1` 和文字版导读 `guide-1`；正文锚点 `request-1`、`concepts-1`、`frontend-backend-1`、`startup-1`、`request-flow-1`、`source-map-1`、`reading-1`、`assignments-1`、`answers-1`、`acceptance-1`、`sources-1`；末尾 `oral-1`（60 秒口述与面试追问） |
| `02_AgentLoop与子Agent派发.md` | FunctionTool、tool_call_id、task_dispatch、并发、AG-UI 事件；Step 2，45 分钟。2026-10-19 开头加了总览 `overview-2` 和文字版导读 `guide-2`；2026-10-20 正文一到六节按全局视角重写（每节开头「在项目里管什么」、结尾「放到训练里看」，补循环的三种出口和工具三类，作业考点不变）；正文锚点 `loop-2`、`tools-2`、`call-id-2`、`dispatch-2`、`concurrency-2`、`events-2`、`reading-2`、`assignments-2`、`answers-2`、`acceptance-2`；末尾 `oral-2` |
| `03_Harness与状态管理.md` | 六类状态、result_ref、revision/fence、审批与订单幂等、刷新恢复、护栏；Step 3，60 分钟。2026-10-19：开头 `overview-3`、`guide-3`；2026-10-22 正文一到七节按全局视角重写（每节开头「在项目里管什么」、结尾「放到训练里看」，作业考点不变）；正文锚点 `story-3`、`state-kinds-3`、`context-3`、`compaction-op-3`、`approval-3`、`recovery-3`、`guards-3`、`reading-3`、`assignments-3`、`answers-3`；补充 `upstream-3`（会话固定 Prompt 与模型、执行记录、下单证据核验、子 Agent 共用护栏、默认关闭的 Prompt Cache 与续写、可见证据规则）；末尾 `oral-3` |
| `04_评测失败分析与训练边界.md` | 确定性断言与 Judge、`score_case`、manifest、Bad Case、训练边界；Step 4，45 分钟。2026-10-19：开头 `overview-4`、`guide-4`；2026-10-22 正文一到八节按全局视角重写（同 02 的写法）；正文锚点 `scope-4`、`case-4`、`runner-4`、`assertions-judge-4`、`score-gate-4`、`metrics-manifest-4`、`bad-case-4`、`training-boundary-4`、`reading-4`、`assignments-4`、`answers-4`；补充 `upstream-4`（评测分三层：端到端回归、Harness 套件、Skill 评测；学习数据回流）；末尾 `oral-4` |
| `05_商品检索与前后端闭环.md` | Qdrant 建库、ProductSearchSpec、单件价与 `landed_price`、三个出口、排障表；Step 5，60 分钟。2026-10-19：开头 `overview-5`、`guide-5`；2026-10-22 正文一到九节按全局视角重写（同 02 的写法）；正文锚点 `scope-5`、`route-5`、`catalog-5`、`params-5`、`search-5`、`price-5`、`outlets-5`、`history-5`、`debug-5`、`reading-5`、`assignments-5`、`answers-5`；上游新增（精确 ID 查询、增量建库、检索阶段采集、下单证据核验、执行记录与工作台）补在第三、四、五、七节；末尾 `oral-5` |
| `06_训练推演_澄清检索补证.md` | 训练推演（2026-10-20）：任务与五类场景、可重置环境、两版奖励、四个规则基线与 SFT 清洗（真实运行），两轮 GSPO 的配置、曲线、评测与归因（推演）。开头 `overview-6`、`guide-6`；正文锚点 `task-6`、`env-6`、`reward-6`、`baseline-6`、`runs-6`、`attribution-6`、`assignments-6`、`answers-6`、`reproduce-6`；末尾 `oral-6`。代码在 `agentic_rl/`，单测 `tests/test_agentic_rl.py` |
| `docs/学习路线/学习改动记录.md` | 2026-10-19 起的学习资料改动记录 |
| `docs/学习路线/贯穿例子维护记录.md` | 贯穿例子的唯一登记处：定义、允许的变化、例外、预算口径、逐处修改记录 |
| `docxs/` | 01–04-2 讲 AgentLoop、派发、选型、检索；08、08-1、08-2 讲 Rubric、SFT 冷启动和 Agentic RL；09、09-1、14 讲项目总览、数据底座、主 Loop 组装；导读把 Step 1–5 对应到 01–05。读取时用 `textutil -convert txt` 转到会话 scratchpad，不要在仓库里生成文件 |
| `docs/设计演进记录.md`、`docs/教程实现对齐清单.md` | 设计取舍与踩坑；教程与实现的差异（明确写着训练侧 04-2、08 系列不在补齐范围内） |
| `eval/verification/stage1-learning-20260926` … `stage5-learning-20261007` | 以往每次落盘的静态检查记录 |

01–05 每篇都有：总览、文字版导读、学习目标、plain text 图、限时阅读表、作业（02–05 是 2 道画图或推演题加 8 道问答）、完整参考答案和自查标准、60 秒口述与面试追问。合并上游后新增的内容，2026-10-19 已补进 03、04、05（位置见 README 纵览）。

## 6. 源码地图（2026-10-19 核对，HEAD `1672fe4`）

这一节是查阅工具。回答时先讲机制，需要证据时再来找位置。

**一次请求**
- `POST /commerce/ag-ui/run`（`app/presentation/ag_ui.py`）→ `app/presentation/ag_ui_runtime.py` 的 `AGUIRuntime` → `app/application/agents/orchestrator.py` 的 `handle_intent`（`:192`）、`_handle_intent`、`_consume_reply`（消费 `agent.reply_stream`）。组装入口是 `app/composition.py` 的 `build_container`。
- 主 Agent 的 `max_iters=15`（`app/application/agents/main_agent.py:214`）。SearchAgent 和 TradeAgent 的 `max_iters=6`（同目录 `search_agent.py:119`、`trade_agent.py:108`），由 `app/application/tools/task_dispatch_tool.py` 的 task_dispatch 每次新建，只回传 JSON 结论。主 Agent 自己也持有全部业务工具，可以不派发。
- 派发（2026-10-19 核对）：`task_dispatch_tool.py` 按类型新建子 Agent（`:106-108`），demands 包成 `UserMsg`（`:115`），只给 SearchAgent 注入长期偏好（`:62`、`:116-118`）。返回 `summary`、`evidence_refs`、`verified_product_ids`、`unverified_product_ids`、`status`；回复里出现本轮证据核实不了的商品 ID 时 `status=unverified`（`:136-142`）。主 Agent 工厂把顺序检查和循环检测交给两个子 Agent 工厂共用（`main_agent.py:107-108`），注册 `task_dispatch` 时声明 `is_concurrency_safe=True`（`:147-159`）。
- 并发：`FunctionTool` 默认 `is_concurrency_safe=True`（`agentscope/tool/_adapters.py:60`）。AgentScope 2.0.8 按这个声明把同一轮的工具调用分批（`agentscope/agent/_agent.py:2103-2135`：相邻可并发的合成一批，不可并发的单独顺序执行），并发的那一批用 `asyncio.gather` 执行（`:2266`）。分批循环只判 `is_concurrency_safe`（docstring 提到 `is_read_only`，代码没用）。`is_read_only` 经 `tool/_base.py:258` 的 `check_read_only` 进权限引擎的只读快速通道（`permission/_engine.py:_check_read_only_fast_path`），直接 ALLOW；非只读工具的 `FunctionTool.check_permissions` 默认 ASK（`tool/_adapters.py:116-135`），要靠 `permissions.py` 的 `_AUTO_ALLOWED_TOOLS` 放行，记忆写工具不在名单里，另由 `MemoryPermissionMiddleware` 强制 ASK。SearchAgent 的工具全是只读，所以它不需要放行名单。Globex 里真正挡副作用的是待确认单和记忆审批。
- 循环出口（2026-10-20 核对，`agentscope/agent/_agent.py:_next_action`）：模型给出纯文字回答即完成；有待审批的调用（记忆写工具的 ASK）时本轮暂停（`:3520-3532`），买家确认后编排器把 `UserConfirmResultEvent` 交回框架，从暂停处接着跑、不重建 ReplyContext（`:1080-1095`；`orchestrator.py:264-271`），表单提交则是新的一轮；`cur_iter == max_iters` 时再强制一次 `tool_choice=none` 的总结（`:3688-3702`），结束原因记 `EXCEED_MAX_ITERS`，编排器把它当失败（`orchestrator.py:568-571`）。澄清表单只在工具结果里提示「结束本轮」（`shopping_form_tool.py:52-53`），框架不强制。`scripts/verify_parallel.py` 用派发的开始、完成时间判断是否重叠。
- 结果回流：`ag_ui_adapter.py` 的 `on_agent_event` 处理回复、文本、思考、工具调用和工具结果等原生事件（`:139-219`，02 只讲其中五个关键事件），`on_trade_event` 把商品工具的 `tool.result` 写进 `state.products`（`:296-332`）。编排器用 `tool_call_id → 工具名` 把结果配回调用（`orchestrator.py:557-582`）。事件先写进 `ag_ui_runs.db`，再由 SSE 回放；前端 `commerceClient.ts` 的 `onStateChanged` 用 `readProducts` 读商品，`onEvent` 不记高频的文本和参数片段（`:524-559`）。

**工具与预算口径**
- 只读工具：`product_search_tool`、`category_insight_tool`、`conversation_fact_lookup`、`query_order_tool`、`load_agent_skill_tool`、`lookup_strategy_memory_tool`、`web_search_tool`（配了 `TAVILY_API_KEY` 才注册）。
- 有副作用的工具：
  - `create_order_tool`、`cancel_order_tool` 只写一张 300 秒有效的待确认单。用户经 HTTP 确认后，`ConfirmationService.resolve` 才真正扣库存或补库存。
  - `remember_preference_tool` 等写长期记忆的工具走审批；写入前会额外调一次 LLM 和 embedding，要花钱。
- `app/application/usecases/catalog_search.py:358-362` 的 `_within_price_cap` 只比主规格的单件价（换算到目标币种），不含运费和关税，也不乘数量；`_to_card` 在 `:437` 用 `quantity=1` 报 `landed_price`。

**Harness**
- 上下文：默认是 AgentScope 原生压缩加 `EvidenceCompactionMiddleware`；设 `CONTEXT_STRATEGY=layered` 才启用 `context_governance.py` 的 `LayeredContextMiddleware`。
  - 它在 `:440` 调用 `app/infrastructure/visible_evidence.py`：用正则识别「重新核验 / 再查」就强制调用检索工具；`visible_answer_request` 会把 `tool_choice` 设成 none，直接用已有证据作答。
  - 这等于由规则替策略决定「要不要补证据」，训练这个决策前要先关掉或绕开。
- 重复检测：`harness_middleware.py:123` 调用 `LoopDetector.observe`（`app/application/harness/loop_detector.py:66-91`）。最近 6 次已完成的调用里，同工具、同参数、同结果累计 3 次才追加提示，只提示不终止。旧的按工具名计数的 `check()` 已没有运行时调用方，`settings.py` 里的注释还是旧口径，不要引用。
- 会话固定与默认关闭的开关（2026-10-19 核对）：
  - Prompt 版本按会话分配后固定（`prompt_registry.py:_assign:184`，工具契约变化时抛 `PromptContractChanged`）。
  - 模型选择只写下一轮的 pending，每轮经 fence 冻结快照（`session_store.py:select_model:275`、`bind_run_model:289`；`model_selection.py:resolve_session_snapshot:10`）。
  - 默认关闭的开关（`settings.py:82-100`）：`prompt_cache_mode=passthrough`；`context_prompt_layout=legacy_system`，stable_prefix 待收益验收；`model_recovery_mode=off`；`learning_capture_enabled=False`。`assertions.py` 的 `SequencingTracker` 在没检索过就下单时硬拒；它的 `reset` 定义了，但没有调用方。
- 模型：`app/infrastructure/llm.py:714` 的 `create_chat_model`（会话冻结的模型快照优先，否则读配置和环境变量）；`:582` 的 `_call_with_fallback` 默认回退到 `qwen-plus`；orchestrator 外层还有整轮重试 `_reply_with_retry`。
- 遥测：`app/infrastructure/tracing.py:224` 的 `_sanitize_attributes` 把消息正文、工具参数和结果换成长度，并标记 `globex.content.redacted`。所以从 Langfuse 的 trace 复原不出轨迹。

**状态与持久化**（都在 `DATA_DIR`，默认 `./data`）
- 会话、对话流水、订单账本（只支持 SQLite）、证据、记忆、Skill、模型和 Prompt 各有 SQLite 库；运行恢复用 `ag_ui_runs.db`。Qdrant 的本地目录有进程锁，只能单进程使用；Redis 可选。
- **没有环境重置**：`app/infrastructure/persistence/sql/trade_store.py:168` 的注释写明「重启或目录刷新只更新报价，绝不能重置已扣减的持久库存」。唯一办法是换一个新的 `DATA_DIR`，而新目录启动时会重新给商品做 embedding（要花钱）。

**评测、GEPA 与数据**
- 用例：
  - `eval/cases.yaml`：13 条老回归用例。
  - `eval/v1/agent_cases.yaml`：100 条，dev 70 / release 30。
  - `eval/learning/skill-cases-v2.json`：600 条，dev 240 / holdout 360。
  - `eval/harness/v1/cases.json`：48 个场景，dev 15 / holdout 33；按 `mode` 分为 snapshot 32、long 9、workflow 7；不用 Judge。`suite.json` 里的档位：contracts 不调模型；smoke 6 个场景×1 次；dev 15×2；release 33×3。release 门禁的计费配置 `pricing.verified=false`，所以必然被拦下。
  - Skill 用例按 `group` 分 6 组各 100 条（expression、revision、negative、conflict、safety、lifecycle），60 个家族，同一家族不跨集合。
- 打分：`scripts/eval_regression.py:199-208` 的 `score_case` 算 `0.5·P0 + 0.35·P1 + 0.15·P2`（P0 是数字事实和安全底线，P1 是行为与命中，P2 是表达），同时返回「P0 是否全过」。判 PASS 还要 trace 上的确定性 P0/P1 全过、总分 ≥0.7，并且同一份 trace 重放结论不能翻转。
- Rubric 由 LLM 裁判打分；能由 trace 证明的条目先从裁判输入里剔除，改由 `scripts/eval/evidence.py` 按工具事件判定。Bad Case 走 `scripts/eval/feedback.py`：导入、人工审核、导出提案，不改正式集。
- GEPA（`scripts/optimize_prompts.py`）：
  - 候选只能改 `instruction`。
  - 评分 `tool_argument_constraint_preservation` 只看检索参数有没有保留约束，0 或 1 分，不评最终推荐。
  - 历史实验只有 12 个合成场景（4/4/4）。`eval/verification/upgrade-20260918/gepa.json` 里 `candidate_changed=false`；holdout 从 1/4 到 2/4 是同一段文本两次生成的波动；`full_agent_release_gate=NOT_RUN`。
- 学习数据回流 `app/infrastructure/learning_store.py`：
  - 默认关闭（`app/infrastructure/settings.py:100`）。
  - 回流记录最多只有本轮提问、最终回答和 12 个商品事实（网页运行失败时只留引用）；工具调用次数只用来判断证据够不够（至少 2 次），不存。
  - 记录分 diagnostic、sft、rl_task 三类（`learning_store.py:262-277`）：diagnostic 只存不导出（`export_dataset` 排除它）；sft、rl_task 由诊断记录加人工给定的固定答案派生，sft 的目标必须等于期望输出，rl_task 不带目标。样本都是单轮的，不是 docxs 08-1 的多轮轨迹格式；没有系统提示、工具 schema、逐步观察、logprob、mask 和奖励。
- 检索（2026-10-19 核对）：
  - `execute` 先用结构化的 `sku_id` / `product_id`（规格优先），都没传才在查询词里找 ID；显式 ID 未命中不退回模糊检索（`catalog_search.py:160-167`）。`ProductSearchSpec` 校验 ID 格式和同属关系。
  - 建库是增量的：`index_bootstrap.py:bootstrap_product_index` 按 `embedding_identity` 只补缺失或变化的商品，每 10 件一批，失败不阻塞启动。
  - 检索阶段采集是评测开关（`catalog_search.py:145`、`:295`；`scripts/eval/retrieval_stages.py`）。下单核验在 `order_tools.py:_verified_order_items:50`。
  - 执行记录写在适配器状态的 `activities` 里（`ag_ui_adapter.py:66-132`）；检索那一步的摘要只显示 ID、配送国、币种、预算和件数（`activity_summary.py`）。
- 商品目录：`data/catalog-v1/v2/v3.jsonl` 分别是 500、1000、3700 个商品（v3 共 7105 个 SKU），都是合成数据。**运行时读 v3**（`app/infrastructure/persistence/seed_products.py`），**Harness 冻结在 v1**（`scripts/eval/harness/contracts.py:70`），所以历史分数不能和现在的运行直接比。`knowledge/` 有 45 篇合成文档加一个 manifest。

**改造成 RL 环境的接缝（训练推演用）**
- 逐步观测：`app/infrastructure/context_usage.py:9` 的 ContextVar `evaluation_evidence_sink`。评测显式安装后，能收到 `model_request`（`llm.py:143-149`，发给网关前的 messages、tools、tool_choice）和 `tool_result`（`app/infrastructure/harness_middleware.py:120`）。
- 隔离多轮的现成样板：`scripts/eval/harness/runner.py:184` 为每个场景建临时目录；`workflows.py:94` 的 `_run_workflow` 自建隔离 SQLite，用脚本驱动多轮对话。
- 模拟用户的插入点：每两次 `handle_intent` 调用之间。澄清表单走 `ShoppingFormStore.submit`，交易确认走 `ConfirmationService.resolve`。
- 换策略模型：`main_agent.py:206` 直接调用 `create_chat_model`，没有注入口。测试里的做法是覆写 `ThrottledChatModel._invoke_upstream`；也可以把网关指向一个 OpenAI 兼容的自建推理服务（推断）。
- 没有 logprob。最接近 episode 的概念是 session（一个线程）、run（一轮），以及回复结束原因 `exceed_max_iters`。

## 7. 已核对、但资料尚未收录的事实（候选，并入前请用户复核）

2026-10-19 起，第 5 条已写进 03 的补充一节，第 6、7 条已写进 04 的总览、补充一节和口述。

1. AGENTS.md 链接的 `docs/Harness评测与收益门禁.md` 不存在，git 历史里也没有。
2. `eval/verification/` 里没有任何一次 Harness 运行的证据。
3. 运行读 v3 目录、Harness 冻结在 v1：09-09 等历史分数与现在的运行不可比。
4. 没有环境重置，库存持久；换 `DATA_DIR` 会触发重新 embedding（花钱）。
5. layered 模式下，`visible_evidence.py` 用正则和 `tool_choice` 替策略做「要不要补证据」的决定；澄清与派发规则写死在 `globex.yml` 和主 Agent 的 prompt 里。
6. `score_case` 的加权分与硬门禁是分开的：P0 只过 4/5、P1 和 P2 全过，加权分仍有 0.9。直接把加权分拿来当奖励会丢掉 P0 这条底线。
7. `learning_store` 的样本离 docxs 08-1 的轨迹格式差很多（见第 6 节）。
8. docxs 08 系列引用的 `distill_trajectories.py`、`app/eval/tool_replay.py` 在仓库里不存在；08-2 设计的五维奖励（0.40 回答质量 + 0.20 过程合规 + 0.15 Query 拆解 + 0.15 相关性 + 0.10 安全，安全不合格或任一 P0 失败时 R=-1）、GSPO、录制回放环境都只在教程里。
9. GEPA 的历史实验规模太小，当不了训练实验的对照组；要做对照得重新设计。

## 8. 建议的下一步（截至 2026-10-19）

1. 已完成（2026-10-19）：环境基线（第 3 节）；贯穿例子定为轻便背包，01–04 已统一（见贯穿例子维护记录）。
2. 带学：2026-10-19 已按全局视角重写 01–05 的总览、导读和口述，补上上游新增内容，并经对抗性复核。接下来等用户依次做 01–05 的作业；每开始学一篇，再按固定流程复核一次。
3. 训练推演（2026-10-20 完成）：用户定了四个边界——训「澄清、检索、补证」；Qwen3-8B + 4×A800；环境建在真实业务层上的轻量环境（复用线上检索工具，Agent 循环、模拟用户、奖励是平行模块）；代码放新建的顶层 `agentic_rl/`，不改 `app/`。
   - 真实运行（CPU，0 次模型调用）：`python -m agentic_rl real`，证据在 `eval/verification/agentic-rl-env-20261019/`。原版奖励和真实成功率只有弱相关（trust_filter 推荐 37% 陷阱商品仍得 1.01；train 四策略合并 1280 条 r≈0.24；held-out 规则成功率 0.10 时原版仍有 1.02）；careful 规则 dev 100%、held-out 10%（认不出城市名、剥不掉 B 套开场白，中文数字能解析；120 题 108 题没推荐）；训练环境的召回只走关键词兜底 `keyword_2gram`（按 ID 核对走精确查找），没接 embedding 和 rerank；SFT 示范 320 留 319；序列级 clip 0.2 不触发，3e-4/4e-4 在演示噪声下裁掉 75%–93%。
   - 推演：`python -m agentic_rl scenario`，产物在 `agentic_rl/runs/qwen3-8b-gspo-20261020/`（来源见 `PROVENANCE.md`）。run-01 held-out 0.31（结果一般），归因三条：奖励漏洞、clip 等于没开、SFT 分布窄；run-02 只改这三处，0.58。run-02 的 `pg_clipfrac` 约 0.15 是假设（倒推每 token 漂移约 0.007），已写进归因。
   - 学习文档 [06](docs/学习路线/06_训练推演_澄清检索补证.md)，单测 `tests/test_agentic_rl.py` 9 个（含「归因里的 clip 假设与 run-02 对得上」「成本只在成功时扣」）。
4. 推演完成后，反过来补 01–05 和 docxs 对照表：06 第九节已列出要补回 02–05 的四条收获，还没写进各篇。
5. 2026-10-21 用户拍板：grilling 压力测试带回的 7 个取舍全部按推荐执行，并要求「从全局最优和一致的角度做，不陷入局部最优或局部幻觉」。已落地：
   - 护栏写出的观察（`[harness]` 提示）训练时当环境反馈保留，不剔除（上线时模型也会看到）；写在 06 第九节。
   - 02 作业 C 不加题；录轨迹时 `task_dispatch` 不注册，已写成 06 第二节的设定。
   - run-02 的 SFT 改写从 B 套换成与 A、B 都不重叠的 C 套（防止把评测说话方式泄进训练）；scenario.py、attribution、06 已同步。
   - `eval/results.md` 和 06 补了三行单因素消融（只换奖励 0.45、只换 clip 0.34、只加 C 改写 0.38），标「推演（消融假设）」，并注明真跑三组约 ¥3,200、可先只跑「只换奖励」。
   - 补强版的两个已知局限（提到已排除的 ID 也判失败；先说「没有」再给 ID 也算成功）写进 06 第三节，代码不改。
   - 奖励成本项改成只在成功时扣、失败一律 0（`reward.py`，2026-10-21）：旧版失败记 -cost，全错组里「少问」反被组内标准化放大成正优势。已重跑 real（138 秒）：成功率、陷阱率、SFT 319 条全部不变，`always_ask` 补强版奖励 0.859→0.860，held-out 补强版奖励从负数变 0.09 左右；06 表格取两位小数无需改。单测加 `test_cost_is_only_charged_on_success`，共 9 个。
