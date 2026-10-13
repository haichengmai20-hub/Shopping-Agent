# 第二阶段 AgentLoop 工具调用与子 Agent 派发

本阶段要把[第一阶段项目地图](01_项目地图与前后端流程.md)中的 MainAgent 展开：模型怎样选工具、Python 怎样执行、工具结果怎样进入下一轮，以及子 Agent 怎样作为一个工具被派发。建议用约 45 分钟学习，再用约 20 分钟完成手画流程图与问题。内容以本地源码提交 8cbd4d6b94eb9713952d3bd8f050da176accdca1 为准；工作区中的本机环境准备改动不改变本章的 AgentLoop、工具与前端事件合同。

学完后要能解释：一条用户消息为何可以触发多次模型和工具调用；模型生成调用与后端实际执行的区别；结果如何进入下一次模型输入；主 Agent 单干与子 Agent 派发的分界；前端如何区分“参数已就绪”和“执行已有结果”；实际并发需要什么证据。

下文的保温杯、背包和耗时是教学情境，没有调用真实模型；某一种示意路径不代表每次买家请求都必然这样执行。

## 一、先看整个循环

~~~text
当前用户需求与会话信息
          |
          v
    模型生成下一步
          |
     +----+-----------------------+
     |                            |
     v                            v
  可以回答                   需要调用工具
     |                            |
     v                            v
  生成回答              工具名 + 结构化参数
                                  |
                                  v
                        后端查找并执行注册工具
                                  |
                                  v
                         返回结果或明确的错误
                                  |
                                  v
                        结果进入 Agent 上下文
                                  |
                                  +----> 再次请求模型
~~~

工具结果进入上下文，是下一次模型判断能利用上一步观察结果的前提。后端运行这条循环时也产生事件，经 AG-UI 送给前端显示进度。

例如买家说：“我第一次买保温杯，不知道怎么挑。预算 200 元人民币，寄到中国，先讲讲选购要点，再推荐具体商品。”一种合理的路径如下：

~~~text
用户提出需求
    |
    v
模型请求 1
    | 判断需要先取得选购知识
    v
category_insight_tool
    | 返回品类选购依据
    v
模型请求 2
    | 根据知识、预算和配送地形成检索参数
    v
product_search_tool
    | 返回具体候选、价格信息和过滤原因
    v
模型请求 3
    | 综合前面的观察结果
    v
选购要点 + 商品建议 + 取舍说明
~~~

“用户对话回合”指用户发来一条消息并等待这一轮处理；“模型调用”指后端向模型服务发起一次生成请求；“工具调用”指实际执行一次工具。上图只有一个用户对话回合，却示意三次模型调用和两次工具调用。具体调用顺序、候选及回答取决于当轮需求、模型和工具结果。

项目在 [MainAgentFactory.build()](../../app/application/agents/main_agent.py) 中设置 ReActConfig(max_iters=15)。它是主 Agent 循环的迭代上限，既不是固定要走 15 步，也不等于所有 HTTP 请求总数；模型调用还可能因上游异常而重试，同一次模型决策也可能产生多个工具调用。SearchAgent 与 TradeAgent 各设置 max_iters=6。循环本体在锁定的 AgentScope 2.0.8 中，项目通过[编排器的 _consume_reply()](../../app/application/agents/orchestrator.py)调用 agent.reply_stream() 并消费事件。

后端知识补充：普通函数常在完成时给出一个返回值；reply_stream() 则连续交出事件。Python 的 async for 每收到一个事件就处理一个，页面因而可以在最终回答完成前先看到文本片段、工具状态或商品卡。

~~~text
时间向后推进 →
Agent：文本片段 → 工具调用 → 工具结果 → 新文本片段 → 结束
后端：逐个接收 → 逐个处理 → 逐个处理 → 逐个处理 → 收尾
~~~

这里的 Think → Act → Observe → Reflect 是教学上的观察角度，不意味着项目每轮都会额外调用一个独立“反思模型”。

## 二、Python 函数怎样成为模型可调用的工具

打开 [SearchAgentFactory.build_tools()](../../app/application/agents/search_agent.py)，先看 product_search_tool 的注册。FunctionTool 将一个 Python 函数包装成 Agent 可用的工具：函数名与文档描述帮助模型判断用途，参数签名生成输入字段说明，真正的函数留在后端等待执行。

~~~text
Python 工具函数
    |
    +-- 函数名 ----------------> 工具名称
    +-- 参数与类型 ------------> 输入字段说明
    +-- docstring 文档字符串 --> 用途、参数含义和使用时机
    |
    v
FunctionTool 包装
    |
    +-- 工具定义交给模型
    +-- 真实 Python 函数留在后端，等待调用
~~~

JSON Schema 是描述数据结构的格式。第一遍可以把它理解成“工具参数填写说明”：有哪些字段、字段是什么类型。下面是生成工具参数的教学示意，并非完整 HTTP 报文：

~~~json
{
  "normalized_query": "保温杯",
  "ship_to": "CN",
  "price_max_major": 200,
  "target_currency": "CNY",
  "top_k": 5
}
~~~

CN 是中国的二位国家码，CNY 是人民币币种代码，top_k 指最多返回多少个候选。模型生成工具名和这些参数后，AgentScope 找到注册工具；后端会进行权限与工具中间件处理，再执行 [product_search_tool.py](../../app/application/tools/product_search_tool.py) 中的函数。工具将参数整理为 ProductSearchSpec，调用 [CatalogSearchUseCase.execute()](../../app/application/usecases/catalog_search.py)，最后以 ToolChunk 返回内容和状态。

~~~text
模型生成工具名和参数
    |
    v
框架定位已注册工具
    |
    v
权限判断及工具中间件
    |
    v
product_search_tool(...)
    +-- 将可接受的数字字符串转成数字
    +-- 构造 ProductSearchSpec
    +-- 调用 CatalogSearchUseCase.execute(...)
    |
    v
业务结果封装为 ToolChunk
~~~

中间件是包在工具调用前后的代码，可处理顺序检查、超时和熔断等工程问题。ToolChunk 是工具返回容器；业务用例负责实现查询、过滤及商品卡构造。

**要同时检查“模型传对”和“业务代码按参数执行”。** 当前 price_max_major 默认是 None；如果模型漏掉这个字段，用例会跳过价格上限检查。因此，教程第 02 章中“做成参数后，模型忘不忘都不影响结果”的表述过强。对预算类需求，既要确认模型把上限传入工具，也要确认业务过滤执行且结果符合约束。[工具参数默认值](../../app/application/tools/product_search_tool.py)、[用例价格检查](../../app/application/usecases/catalog_search.py)

**预算口径补充（2026-10-03 核验）：** 即使 price_max_major 正确传入，当前 _within_price_cap() 也只比较换算后的单件商品金额，未比较含运费和关税的 landed_total_major。要判断总到手预算，还需对照商品卡中的费用范围核验，不能把参数名当成实现保证。

另有两个常见的工具声明：

| 声明 | 当前用途 | 需要注意的边界 |
| --- | --- | --- |
| is_read_only=True | 声明这是业务只读工具，供框架权限判断 | 不等于重试零成本，也不保证库存快照永不变化 |
| is_concurrency_safe=True | 允许调用进入框架的并发批次 | 不证明任务之间独立，更不证明一次运行已经并发或一定提速 |

## 三、调用和结果怎样配对，并影响下一轮

一次工具调用包含工具名、参数及调用 ID。工具结果带同一调用 ID，框架据此把这次调用和对应结果联系起来，再将结果加入 Agent 上下文。模型之后才有机会利用新观察继续判断。

~~~text
用户消息
    |
    v
assistant：调用 category_insight_tool，ID=call_A
    |
    v
tool：返回选购知识，关联 call_A
    |
    v
assistant：调用 product_search_tool，ID=call_B
    |
    v
tool：返回商品结果，关联 call_B
    |
    v
assistant：综合已有结果，生成回答
~~~

调用 ID 不能省略。假设同轮两次调用 product_search_tool，第一次查保温杯、第二次查背包，而背包先返回；只看工具名或返回顺序，就可能把两份结果配错。项目[编排器的事件消费](../../app/application/agents/orchestrator.py)也用 tool_call_id 联系工具开始和结果结束事件。

“Observe → 再次决策”的人话解释就是：上一步的工具结果被放进下一次模型可读的上下文，它才能据此改写检索词、换工具、承认未找到，或者收尾作答。工具返回错误也属于观察，不能被包装为“成功检索到空列表”。

## 四、子 Agent 怎样作为一个工具被派发

主 Agent 在[工具集装配](../../app/application/agents/main_agent.py)时持有直接业务工具，也持有 task_dispatch。后者对主 Agent 来说是一次工具调用；内部却会创建并运行另一个 Agent，这叫 SubAgent as Tool。

~~~text
MainAgent 的模型
    | 生成 task_dispatch 调用
    v
task_dispatch(subagent_type, demands)
    +-- 选择 SearchAgentFactory 或 TradeAgentFactory
    +-- 新建子 Agent
    +-- 把 demands 作为子任务输入
    +-- 等待 worker.reply(inputs)
    +-- 整理结论、证据引用和核验状态
    |
    v
以工具结果形式回到 MainAgent
~~~

在 [task_dispatch_tool.py](../../app/application/tools/task_dispatch_tool.py) 中，可以顺序找 worker = ...build()、UserMsg(..., demands)、await worker.reply(inputs)。子 Agent 是当前后端进程里新建的 Python 对象，有自己的模型循环、提示词、工具集和 AgentState；每次派发不会自动启动另一个 Docker 容器或操作系统进程。

~~~text
主 Agent
    +-- 自己的对话历史与 AgentState
    +-- 生成 demands
             |
             v
       新建的子 Agent
          +-- 子任务描述 demands
          +-- 自己的提示词与业务工具
          +-- 独立 AgentState
          +-- 服务端按配置注入的相关信息
~~~

独立状态不意味着没有可信身份。子 Agent 可读取服务端传递的买家和会话上下文；当前派发工具还会按配置为 SearchAgent 注入相关长期偏好。但它没有自动复制主 Agent 的完整对话历史。本轮商品目标、预算、收货地、商品 ID 等仍应写进自包含的 demands。“按刚才那个要求查一下”不足以支持可靠派发。

主 Agent 的[当前提示词](../../app/application/prompts/globex.yml)给出三类考虑派发的条件：独立子任务可并行、子任务中间信息需要隔离、子任务内部调用链较深。一次简单查询或查单可以直接调业务工具。后端提供派发能力并设置护栏，具体是否派发仍取决于模型的选择；这是策略，不是一个能完全判断业务依赖的确定性分类器。

~~~text
MainAgent 收到目标
    |
    +-- 单步业务工具足够完成？
    |       +-- 是：MainAgent 直接调用
    |
    +-- 有独立可并行的子任务、上下文隔离或较深内部链路？
            +-- 是：考虑 task_dispatch
            +-- 否：继续由主 Agent 处理
~~~

例如“查这件商品现在能否寄中国”可由主 Agent 直接查询；“分别研究保温杯和背包，各自比较候选”可考虑拆分；“先找商品，再对选中的 SKU 准备订单确认”有前后依赖，不能忽略前一步结果直接并发。

当前派发返回值包括 summary、evidence_refs、verified_product_ids、unverified_product_ids 与 status。status 为 unverified 时，主 Agent 应核对证据并考虑再查。即便 status 为 completed，也只表示当前代码完成了它能执行的相应检查，不证明所有推荐理由和事实都正确。工具调用本身成功返回与子任务业务结论可信，是两个层次。

## 五、async、并发和实际执行证据

Python 中的 await 表示等待当前异步操作，同时让事件循环有机会处理其他工作。连续执行 await A() 再 await B()，通常是先等 A 结束才开始 B。要让两个独立任务重叠，需要先调度两者，再等待收齐。

~~~text
串行：
启动 A → 等 A 完成 → 启动 B → 等 B 完成

并发：
启动 A ───────────────→ A 完成
启动 B ──────────────────────→ B 完成
                    等待两者收齐
~~~

主 Agent 把 task_dispatch 注册为 is_concurrency_safe=True。同一轮模型生成多个可并发调用后，锁定的 AgentScope 2.0.8 会将它们分批调度，其并发路径使用 asyncio.gather。声明只是必要条件：模型还需在同轮发出多个适合同时执行的调用；外部模型网关也可能有配额或限流。

~~~text
主模型同轮生成两个 task_dispatch
         |
         v
框架建立可并发批次
    +----+--------------------+
    |                         |
    v                         v
子 Agent A                 子 Agent B
研究保温杯                 研究背包
    |                         |
    +------------+------------+
                 |
                 v
          两份结果回到主 Agent
                 |
                 v
              汇总回答
~~~

若 A、B 单独耗时分别为 5 秒和 8 秒，且耗时固定、忽略额外开销，则串行约 13 秒，并发约 8 秒。这只是学习用计算，不是项目实测。实际成本还包括主模型决策、子 Agent 初始化、限流、重试与汇总。

项目的 [verify_parallel.py](../../scripts/verify_parallel.py) 通过派发开始与完成时间检查执行区间是否重叠，并比较墙钟时间。声明可并发、模型同轮多派、时间区间重叠分别对应能力、行为和执行证据。是否提高质量或降低成本，还需其他评测。

## 六、前端看到了什么，后端如何交付事件

前端不是从一大段模型文字里猜测工具执行状态。后端的[AGUIRunAdapter](../../app/application/agents/ag_ui_adapter.py)接收两类事件：AgentScope 原生的文本与工具事件，以及商品、派发和确认等业务事件。它们转为 AG-UI 事件或状态，写入持久运行日志，经 SSE 回到浏览器。[AGUIRuntime](../../app/presentation/ag_ui_runtime.py)是这一运行与日志过程的承载者。

~~~text
AgentScope 原生事件                 业务工具事件
文本、工具参数、工具结果             商品、派发、确认等
            |                              |
            v                              v
AGUIRunAdapter.on_agent_event()  AGUIRunAdapter.on_trade_event()
            |                              |
            +---------------+--------------+
                            |
                            v
                   AG-UI 事件或状态
                            |
                            v
                  持久运行日志 → SSE
                            |
                            v
                  浏览器 CommerceClient
                   +--------+--------+
                   |                 |
                   v                 v
               消息与进度         商品等结构化状态
                   +--------+--------+
                            |
                            v
                       React 重新渲染
~~~

前端知识补充：浏览器的 CommerceClient 维护“有哪些消息、商品、工具进度、当前运行状态”等数据。[onMessagesChanged](../../frontend/src/lib/commerceClient.ts)更新消息；onStateChanged 更新商品和进度；onEvent 保存适量诊断事件。状态改变后 React 根据新数据重新显示页面。高频文本片段和工具参数片段不会都变成一条时间线记录。

请把以下五个时刻区分开：

| 原生事件 | 当前含义 | 前端大致怎样表现 |
| --- | --- | --- |
| TEXT_BLOCK_DELTA | 模型新生成文本片段 | 逐步更新文本 |
| TOOL_CALL_START | 开始生成某次工具调用 | 记录工具名与调用 ID |
| TOOL_CALL_END | 这次工具调用的参数流结束 | 可以说“参数已就绪” |
| TOOL_RESULT_START | 工具结果开始到来或进入执行阶段 | 进度标为运行中 |
| TOOL_RESULT_END | 工具结果收口，并带状态 | 标为完成或错误 |

~~~text
TOOL_CALL_END：参数已生成完
          |
          v
权限判断和真正的工具执行
          |
          v
TOOL_RESULT_END：结果与状态已返回
~~~

TOOL_CALL_END 不能显示为“查询成功”。商品卡也不靠正则分析模型回答中的价格：商品工具发布的业务 tool.result 由适配器更新 state.products，前端通过 onStateChanged 和 readProducts() 接收结构化商品并渲染。[商品状态适配](../../app/application/agents/ag_ui_adapter.py)、[前端状态订阅](../../frontend/src/lib/commerceClient.ts)

子 Agent 内部当前调用 worker.reply()；其业务工具事件仍可由 EventBus 上报，但页面看到子任务进度，不等于它接收了子 Agent 完整的模型文本和内部历史。

## 七、45 分钟阅读安排与知识补充

| 时间 | 先看项目哪个位置 | 读完应能解释 |
| --- | --- | --- |
| 0～10 分钟 | [orchestrator.py 的 _consume_reply()](../../app/application/agents/orchestrator.py) | 一次用户请求怎样消费文本、工具和结果事件 |
| 10～20 分钟 | [search_agent.py](../../app/application/agents/search_agent.py) → [product_search_tool.py](../../app/application/tools/product_search_tool.py) | 工具怎样注册、传参和返回 |
| 20～32 分钟 | [main_agent.py](../../app/application/agents/main_agent.py) → [task_dispatch_tool.py](../../app/application/tools/task_dispatch_tool.py) | 主 Agent 持有哪些工具、子任务怎样创建和回传 |
| 32～40 分钟 | [ag_ui_adapter.py](../../app/application/agents/ag_ui_adapter.py) → [commerceClient.ts](../../frontend/src/lib/commerceClient.ts) | 原生事件及商品状态怎样到页面 |
| 40～45 分钟 | [verify_parallel.py](../../scripts/verify_parallel.py) | 如何检查并发时间区间 |

docxs 只作概念补充，按疑问定位相应节：

| 要补的概念 | 资料 |
| --- | --- |
| 单轮、多轮、FunctionTool 和 reply_stream 类型事件 | [02 AgentLoop 快速入门](../../docxs/02_AgentLoop快速入门与多轮工具调用.docx)第 3～6 节 |
| 三件事判断、自包含 demands、SubAgent as Tool | [03-0 派发策略](../../docxs/03-0_主AgentLoop按需派发子Agent的策略与三件事判断.docx)第 3～5、8 节 |
| Supervisor-Workers 的架构取舍 | [03-1 技术栈与架构选型](../../docxs/03-1_技术栈与多Agent架构选型.docx)第 4 节 |
| 主 Agent 工具装配、并发和迭代上限 | [14 主 AgentLoop 组装](../../docxs/14_主AgentLoop组装与Supervisor-Workers协同机制.docx)第 2、3、4.2、7 节 |
| AG-UI 事件持久化与网页恢复 | [AG-UI 持久运行与断线恢复](../AG-UI持久运行与断线恢复-2026-09-09.md)开头与“存储与执行” |

本章有关漏传预算时的实际行为、服务端偏好注入、子 Agent 返回状态及前端事件含义，优先按当前源码核对。

## 八、作业：先独立画图和回答，再看答案

### 作业 A：画一次工具循环，约 5 分钟

任务：“我不熟悉保温杯，想先了解怎么挑，再找总到手价不超过 200 元人民币、可寄中国的候选。”画出一种合理路径，标注至少两次工具调用及各自调用 ID、工具结果怎样进入下一次模型输入、给 Agent 的结果回路与给前端的事件回路，并写清 TOOL_CALL_END 和 TOOL_RESULT_END 分别出现的位置。

### 作业 B：画一次派发，约 5 分钟

任务：“分别研究保温杯和背包。保温杯单件到手价不超过 200 元人民币，背包不超过 300 元人民币，都寄中国，各给候选与比较理由。”假设主 Agent 选择两个检索子任务。画主 Agent、两个子 Agent、各自工具及汇总；各写一段自包含 demands。再假定 A 耗时 5 秒、B 耗时 8 秒，分别画串行与并发时间线，标注这些数字是假设。

### 作业 C：八个问题，约 10 分钟

1. 模型在回答里写“我调用了商品搜索”，能否证明工具已执行？需要核对什么？
2. 一条用户消息产生三次模型请求，是否意味着三轮用户对话？max_iters=15 限制什么？
3. 用户要求预算 200 元，但工具调用没有 price_max_major，后端一定会过滤超过 200 元的商品吗？
4. await A() 后面再写 await B()，是否自动并发？is_concurrency_safe=True 又能证明什么？
5. 子 Agent 看不到主会话完整历史，是否意味着它不知道买家是谁，也无法取得长期偏好？
6. 前端收到 TOOL_CALL_END，是否可以显示“查询成功”？同名工具被调用两次时如何区分结果？
7. 两个简单查询都能直接用业务工具完成，是否一定要派两个子 Agent？派发增加什么成本？
8. task_dispatch 成功返回，但业务内容中 status 为 unverified，主 Agent 应怎样处理？

## 九、参考答案

### 作业 A：工具循环参考图

~~~text
用户需求
    |
    v
模型 1 → category_insight_tool，ID=call_A
    |
    +-- TOOL_CALL_END：参数生成完毕
    |
    v
后端执行工具
    |
    +-- TOOL_RESULT_END：结果收口，检查状态
    |
    v
call_A 的结果进入上下文
    |
    v
模型 2 → product_search_tool，ID=call_B
         明确带预算、币种和配送地
    |
    v
后端检索并按传入的约束过滤
    |
    +-- call_B 的结果进入上下文 → 模型 3 综合回答
    |
    +-- 商品业务事件 → AG-UI 状态 / 日志 → SSE → 页面商品卡

全过程的原生事件 → AG-UI 适配 → 页面文字与执行进度
~~~

这是合理路径示例，不要求模型一定按此顺序调用。若工具返回失败、没有候选或参数缺失，下一步判断应据真实结果调整。

### 作业 B：派发参考图与任务说明

~~~text
MainAgent 同轮生成两个 task_dispatch
         |
    +----+-------------------------+
    |                              |
    v                              v
SearchAgent A                  SearchAgent B
保温杯≤200元                   背包≤300元
配送 CN                        配送 CN
    |                              |
各自的模型与工具循环           各自的模型与工具循环
    |                              |
    +---------------+--------------+
                    |
                    v
        MainAgent 收到两份结论及证据
                    |
                    v
              汇总推荐与取舍
~~~

A 的 demands 示例：“检索可寄中国、单件总到手价不超过 200 元人民币的保温杯，比较容量和便携性，返回候选标识、价格依据及限制条件。”

B 的 demands 示例：“检索可寄中国、单件总到手价不超过 300 元人民币的背包，比较重量和容量，返回候选标识、价格依据及限制条件。”

~~~text
仅作计算示意，假定 A=5 秒、B=8 秒，忽略启动与限流：
串行：A [0────5]  B [5────────13]  总约 13 秒
并发：A [0────5]
      B [0────────8]                 总约 8 秒
~~~

真实项目应根据两段实际开始和完成时间判断重叠，再统计总体耗时；这两个示意数字不是项目测试成绩。

### 作业 C：八题参考答案

1. 不能。模型写出“我调用了”是文本，并不证明后端真的执行过。需要核对结构化工具调用、调用 ID、后端执行事件、与之对应的结果和业务状态。
2. 仍是一个用户对话回合。为了处理本轮需求，Agent 可以多次调用模型和工具；max_iters 控制 Agent 循环迭代上限，不等于用户消息数，也不是所有 HTTP 请求数。
3. 不一定，当前实现中不会仅凭原始用户句子自动补齐该参数。price_max_major 默认 None；没有上限时用例跳过价格检查。应检查模型传参以及业务过滤后的结果。
4. 连续两个 await 通常先等 A 完成才执行 B。并发需要两项工作都进入可同时执行的调度；is_concurrency_safe=True 只表示该工具允许被放进并发批次，还需模型同轮多调用与实际时间重叠证明。
5. 不是。子 Agent 的 AgentState 独立，但仍能接收服务端可信的买家、会话上下文；当前代码可按配置给 SearchAgent 注入相关长期偏好。本轮预算、收货地和商品目标仍须写清，不能让它猜主会话里“刚才”的含义。
6. 不可以。TOOL_CALL_END 只说明参数流结束，成功或失败要等工具结果；同名的两次调用靠 tool_call_id 关联各自结果，不能看工具名或完成先后。
7. 不一定。MainAgent 已持有业务工具。派发增加任务说明、子 Agent 模型调用、额外上下文和结果汇总的成本，应在并行、隔离或较深子任务确有价值时使用。
8. 应按“内容仍待核验”处理。查看 unverified_product_ids 和 evidence_refs，必要时直接用业务工具复核；不能将 ToolChunk 成功返回误当成全部商品事实已经成立。即使 status 为 completed，也不代表所有推荐理由都经过完整核验。

## 十、验收与边界

合上资料后，你应能画出“工具结果进入下一次模型输入”的回路，解释主子 Agent 的状态和输入边界，并根据前端事件区分参数生成、实际执行和结果收口。若仍把模型的一句话当成工具执行证明，或把所有可并发声明当作实际加速证明，应回到第二、五、六节复查。

本章展示的是当前代码能力与教学情境；没有在本章执行真实模型、商品检索或并发实验。业务上的推荐正确率、延迟与成本需要在对应配置、数据和实验范围内独立验证。
