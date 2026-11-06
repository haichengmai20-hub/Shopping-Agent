# 07 · 检索增强 RAG：商品召回、知识库与偏好记忆

<a id="overview-7"></a>

## 总览

**这一篇在整个系统里的位置。** 05 讲了「一次商品检索的流程」——查询怎么进来、结果怎么出去。这一篇往下钻一层：检索本身是三种**向量检索（RAG）**在支撑，外加一套共享的 embedding 底座。它们是 Globex 的「外部知识层」：模型的参数里没有商品、没有品类知识、没有你的偏好，这三样东西都靠检索在运行时喂给它。

**这一篇要回答的问题和答案。**

- 问：这个项目里有 RAG 吗？——有，而且三个落点、三种形态（读代码 + 真实运行核实，2026-11-04/05）：
  1. **商品向量召回**（`product_search_tool` 背后）：3700 商品文本 → embedding → Qdrant 近邻检索，叠加预算/目的国/材质的**结构化过滤**；
  2. **品类知识 RAG**（`category_insight_tool` 背后）：45 篇知识文档切块入向量库，检索后还带**证据边界守卫**（防幻觉层）；
  3. **偏好语义召回**（长期记忆读路径）：买家偏好带向量，按与本轮问题的相关性挑 top-k 注入——但黑名单永不截断。
- 问：它们和教科书里的 RAG 一样吗？——形态不同。教科书 RAG 是「检索片段拼进 prompt 再生成」；这里前两种是 **agentic retrieval**（模型自己决定调不调、调什么参数，检索结果是工具返回的证据），第三种是**编排层注入**（模型没得选）。三种形态在同一个仓库里并存，正好构成一个 RAG 形态谱系。
- 问：怎么保证检索坏了系统还能用？——**降级链**贯穿始终：embedding 挂了退关键词 2-gram，精排挂了退向量分，知识答不了就明说答不了。

**放到训练推演里看。** 06 的训练环境当时只做了 `keyword_2gram` 兜底召回（没接 embedding 和 rerank，见 06 第二节）。学完这篇你会精确知道训练环境比线上「薄」在哪里：少了语义召回这一层，同义词/跨语言的商品就召不回——这直接改变 rollout 的难度分布和奖励的判分口径（见本篇第二节的三级对照实验，证据在仓库根本地记录与 `~/globex-01-demo/`）。**检索质量是环境质量**，这是 RAG 与 RL 的连接点。

**全篇结构**：第一节形态地图；二、三、四节分别拆三个落点；五节讲共享底座与设计哲学；六节回到训练。源码依据 `app/infrastructure/rag/`（目录名就叫这个）、`app/infrastructure/vector/`、`app/application/usecases/catalog_search.py`、`app/application/memory/preference_selector.py`。预计 50 分钟。边界：不讲前端展示、不讲 Qdrant 部署运维；rerank 只讲到协议层（本项目默认关闭）。前接 05（检索流程）与 03（记忆与注入），后接 06（训练环境）。

<a id="guide-7"></a>

## 导读

先读总览，然后按顺序读正文一到六节；第四节读完回去翻 03 的记忆一节会更有感觉；最后做[作业](#assignments-7)、用[60 秒口述](#oral-7)自测。每节开头「在项目里管什么」告诉你这节在系统里的位置，结尾「放到训练里看」告诉你它对做 RL 的人意味着什么。

---

## 一、形态地图：RAG 在这个项目里的三个落点

<a id="rag-map-7"></a>

**在项目里管什么。** 一次购物对话里，有三类信息模型天生不知道，都要靠运行时检索：

```
                 ┌────────────────────────────────────────────┐
                 │            模型参数（qwen-plus）             │
                 │   语言能力 ✓  世界常识 ✓  以下三样 ✗         │
                 └──────┬──────────────┬──────────────┬────────┘
                        │              │              │
          ①商品向量召回       ②品类知识 RAG       ③偏好语义召回
          （有什么货）        （怎么选的专业知识）   （这个买家是谁）
                        │              │              │
    product_search_tool │ category_insight_tool │ 编排器注入 hint
    （模型主动调）        │ （模型主动调）        │ （模型没得选）
                        │              │              │
    3700 商品 searchable │ 45 篇文档切块入库     │ memory_facts 带向量
    _text → embedding   │ → 检索+边界守卫       │ → top-k 选择注入
    → Qdrant 近邻        │                      │
                        ▼              ▼              ▼
                 结构化过滤（预算/目的国/材质）── 与向量检索组合
```

三者的**调用形态**是本节重点：①② 是工具调用——模型看得到工具 schema，自己决定何时调、传什么参数，检索结果是 `tool.result` 证据，模型可以选择不信（场景 7 里主 Agent 就否决过子代理的 `unverified` 结果）；③ 是编排器在组 prompt 时就注入进去，模型只能看到、不能拒绝。**同一个「检索增强」，控制权位置完全不同**——面试里讲 RAG 时能分清这两种形态，比背「RAG 就是检索+生成」高一个层次。

**放到训练里看。** ①② 这种「模型决定调不调」的形态，恰好是 Agentic RL 要训练的决策点之一（调不调工具、传什么参数都是 policy 的一部分）；③ 这种「环境注入」则是环境的一部分，policy 学不到也改不了。06 录轨迹时把 `[harness]` 提示当环境反馈保留，同样的逻辑适用于注入的偏好 hint。

## 二、商品向量召回：embedding + 结构化过滤的组合拳

<a id="product-rag-7"></a>

**在项目里管什么。** `product_search_tool` 的召回链（`catalog_search.py`），从商品文本到候选列表：

1. **入向量库的文本**（`product.py:69 searchable_text`）：标题 + 品牌 + 品类 + 原产国 + 描述 + 亮点 + 材质 + 关税品类的八字段拼接——召回粒度是「整个商品一段文本」，不是逐 SKU。
2. **建库是增量的**（`vector/embedding_identity.py`）：身份哈希 = sha256(endpoint + model + version + dim)，商品侧再记 `text_hash` + schema 版本。换 embedding 型号或改了商品文案，只重嵌变化的部分；其余复用（`index_bootstrap` 每批 10 件，失败不阻塞启动）。
3. **检索 = 向量近邻 + 结构化过滤**：查询词嵌成向量 → Qdrant 近邻候选 → 预算（`price_max_major` 单件价）、目的国（`ship_to` + 不可送达过滤）、材质黑白名单在检索侧**硬过滤**，不是让模型看结果自己筛。实测三种过滤原因：`over_price_cap` / `out_of_stock` / `ship_to_unavailable`（见 01 跑通记录场景 2）。
4. **召回阶梯**（`catalog_search.py:12`）：`embedding_rerank → embedding_only → keyword_2gram`。embedding 服务异常时退 2-gram 关键词召回（`_keyword_recall`）；reranker 配了走 HTTP 精排（`HttpReranker`），没配或失败就按向量分排序（`rerank_applied=false`）。每次结果里 `recall_strategy` 字段如实记录走了哪条路。

**一个真实的三级对照实验**（2026-11-04；前两级为免费假网关实测〔data/〕，第三级为真实 text-embedding-v3〔data-real〕，分级标注见 01 跑通记录场景 2）：同一句「轻便背包」——随机向量召回 → 全乱（窗帘陶瓷杯）；词袋向量 → 关键词重叠商品；真语义向量 → 语义近邻（Wanderlite 轻量旅行腰包）。**价格过滤在三级里都正常**——结构化过滤和向量召回是两层，坏一层不连累另一层。这个实验建议亲手复跑（`~/globex-01-demo/`）。

**放到训练里看。** 06 的训练环境只实现了第 4 层的兜底（keyword_2gram），没接第 1-3 层。这意味着：训练 rollout 里「防水背包」召不回「Mochila de caminhada」（葡语徒步背包），而线上真实环境可以。如果你要在 06 环境上做严肃实验，把 embedding 召回接回去是第一优先级的补强——否则策略在「同义/跨语言商品」上的行为永远学不到。

## 三、品类知识 RAG：带证据边界守卫的知识库

<a id="knowledge-rag-7"></a>

**在项目里管什么。** `category_insight_tool(question, top_k)` 背后是 `app/infrastructure/rag/`（目录名就叫 rag），管线复用 AgentScope 2.0 的组件：`TextParser → ApproxTokenChunker（按 token 切块）→ QdrantStore → KnowledgeBase`，embedding 配置（端点/型号/密钥）与商品召回共用同一份 settings，但客户端实现是 AgentScope 的 `OpenAIEmbeddingModel`（与商品侧自研 httpx 客户端是两套）。45 篇合成知识文档（`knowledge/`）切块入库时，manifest 里的**来源与时效元数据随 chunk 一起入库**（`load_knowledge_metadata`）。

检索侧四个零件，每个都在防「检索增强」变成「检索幻觉」：

1. **标题定位补检**（`targeted_documents`）：先做整体向量检索，再按已登记文档标题在问题里定位主题（多主题问题如「背包和水壶怎么选」），对纯向量没命中的主题文档补一次定向检索，合并时目标文档优先——纯向量容易只命中一个主题。
2. **可回答性判定**（`has_answerable_knowledge`，最低相似分阈值 0.20）：检索分数都太低就承认「库里没有」，不硬凑。
3. **证据边界守卫**（`unsupported_fact_reason`）：一组正则拦截**明显越过知识边界**的问题——未来事实（「明年售价」）、缺必要参数（「没有目的国能不能精确算税」）、未证实来源（「网传…直接告诉我」）、范围外实体（未知品牌的官方承诺）——返回拒答原因而不是让模型编。这是把「不该答的问题」挡在生成之前。
4. **词法兜底**（`keyword_fallback_insights`）：向量检索不可用时退关键词匹配。

**实测与一个 grounding 边界样本**（真实运行，2026-11-04，journal 回放工具原始返回）：问「先别搜商品，讲讲户外背包怎么选」→ 工具真实返回 `travel-gear.md` 的品类洞察 chunk（材质判断口径「再生尼龙仍属合成聚合物、硬约束应读 material_tags」、自重口径、热卖款型、价格区间，score 0.70，metadata 标着 `source_type: synthetic_evaluation_fixture` 与生效期）。但模型回答里出现的「420D/600D 尼龙」「容量虚标 10-15%」**不在检索结果里**——那是模型参数知识在证据之上的再创作。教训：**RAG 的输出里「检索到的」和「模型补的」肉眼看不出边界**，判分要对着工具原始返回核（04 的口径），这也解释了为什么轨迹记录必须留 `tool.result` 原文。检索结果本身的元数据还带时效：`policy_fact_status` 先核来源类型，非官方快照（当前 45 篇全部是 synthetic_evaluation_fixture）直接判 `non_authoritative_source`（不可作确定事实），只有官方快照才按生效期判断过期。

**放到训练里看。** 证据边界守卫是「工具拒绝执行」的一个真实类别：环境不是有问必答的，`unsupported_fact_reason` 这类守卫决定了 rollout 里「不合理请求 → 被拒 → 策略要学会改问法或明说不知道」的转移。给训练环境设计工具时，「什么时候该拒绝」和「什么时候该返回」同样重要。

## 四、偏好语义召回：会自我保护的记忆检索

<a id="pref-rag-7"></a>

**在项目里管什么。** 长期偏好（03 讲过写路径与审批）的**读路径**是一次教科书级的小型 RAG（`memory/preference_selector.py`）：

1. 每条偏好落库时带**语义向量**（`buyer_memory.db.memory_facts` 的 vector 列）；
2. 每轮对话，**主 Agent 的记忆中间件**（`BuyerMemoryMiddleware`，`main_agent.py:130` 挂载）用**本轮问题**对偏好做向量检索，按相关性挑 top-k 注入 prompt——不全集注入，防止 50 条无关偏好稀释上下文；派发子 Agent 时 `task_dispatch` 用 demands 再做一次同样的选择（只给 search_agent 注入）。注意执行者分工：默认装配（语义记忆）下编排器本身不做 top-k，它只负责把 `material_exclusion_tags` 写进 ShoppingContext（`orchestrator.py:670`）；按 query 的 `PreferenceSelector` 路径（`:681`）只在旧版非语义 store 时才走。
3. **安全底线：`dislike`（黑名单类）全量保留，不参与 top-k 截断。** 注释里的理由值得背下来：query「咖啡杯」和偏好「不要塑料」的向量相似度很低，纯 top-k 会把它丢掉，于是推出一只塑料杯——**漏掉黑名单是安全问题，不是相关性问题**；只有正向偏好（like）才按相关性裁。
4. **偏好还能升级成硬过滤**：`material_exclusion_tags` 用正则识别「全面材质否定」（如「不要塑料材质」），映射成目录的 `合成聚合物` 标签；编排器把它放进 `ShoppingContext`（`orchestrator.py:670`），`product_search_tool` 在参数层合并去重（`:120-124`）——**即使模型忘了传，编排层也兜底**。带范围限定的否定（「不要塑料食品盒」）不升级，仍走完整注入让模型自己解释。

实测（真实运行，2026-11-04，01 跑通记录场景 11/12）：写入「不要塑料材质」→ 审批 → 落库 → **全新会话**问收纳包，回答「所有商品都避开了塑料材质（根据您的长期偏好）」，且检索参数里出现 `excluded_material_tags:["合成聚合物"]`（编排层与模型双通道，工具层合并）。

**放到训练里看。** 这就是 Mem0/MemGPT 类「记忆即检索」思想的最小实现。对训练的意义：偏好注入是 prompt 的一部分 → 改变观测分布 → 影响 policy。若在做多轮偏好遵循的训练，注意黑名单「不截断」这类**规则性保证**——它应该原样进环境（同 06 把 `[harness]` 提示保留的决策）。

## 五、共享底座与设计哲学

<a id="base-7"></a>

**在项目里管什么。** 三个落点共用一套基础设施，三个设计决定反复出现：

1. **同一套 embedding 配置、两套客户端与两套维度口径**：商品与偏好侧用自研 `openai_embedding_client.py`（httpx 裸调，不引 SDK）；知识库用 AgentScope 的 `OpenAIEmbeddingModel`。维度上，知识库按 settings 的**显式维度**（环境变量 `EMBEDDING_DIM`，默认 1024）建 collection，商品索引按首批向量推断——实测坑（免费运行，2026-11-04）：给错维度时商品库照收、知识库建库失败但不阻塞启动。
2. **增量与身份**：换 embedding 型号/端点 = 身份哈希变 = 只重嵌该身份下缺的；商品文本变 = `text_hash` 变 = 只重嵌该商品。
3. **处处降级、如实报告**：召回 `embedding_rerank→embedding_only→keyword_2gram`；知识库向量→词法；偏好注入失败「不能让派发挂掉」（warning 后跳过）；每次降级都在结果里留痕（`recall_strategy`、`rerank_applied`）——**降级不是失败，是一种要能被观测到的状态**。

**放到训练里看。** 「降级留痕」对训练数据构造直接有用：录轨迹时 `recall_strategy` 字段告诉你这条观察来自哪一层召回——如果训练环境和线上的召回层不同（06 就是这样），这个字段就是诊断分布偏移的现成探针。

## 六、回到训练：检索质量就是环境质量

<a id="train-7"></a>

**在项目里管什么。** 把四条线索拧起来：

- 06 的训练环境只接了 `keyword_2gram` 兜底（06 第二节明说），线上是三层阶梯——**训练环境和线上环境在检索层不同构**；
- 三级 embedding 对照实验显示：召回层从「随机」到「词袋」到「语义」，候选分布完全不同——这意味着（推断，未实测策略侧）同一策略的可达奖励分布也不同；
- 偏好注入、知识边界守卫这些「环境侧规则」进了 rollout 就是转移函数的一部分；
- `recall_strategy`/`rerank_applied` 这些留痕字段是现成的分布诊断探针。

所以 RAG 对做 RL 的人不是「另一个方向」，而是**环境工程**：你训练的 agent 调用的每一个检索工具，都是环境的一部分；环境薄一层，策略就少学一层。如果后续要把 06 环境升级到与线上一致，顺序应该是：embedding 召回（语义这一层收益最大）→ 结构化过滤对齐（已对齐）→ rerank（默认关闭，可最后）。

---

<a id="assignments-7"></a>

## 作业

1. **画图题**：把「买家说：帮我找个不要塑料的收纳盒」从一句话到候选列表的完整路径画出来，标出：偏好检索发生在哪一步、`material_exclusion_tags` 从哪来（两个通道）、向量近邻和结构化过滤谁先谁后、如果 embedding 服务挂了会发生什么。
2. **推演题**：假设要在 06 的训练环境里把 embedding 召回接上，写出：改动落在哪个文件、`embedding_identity` 该复用还是新建、旧 rollout 数据还能不能和新数据混着训练（提示：观察分布变了），以及你用什么字段证明「环境升级生效了」。

<a id="answers-7"></a>

## 参考答案要点

1. 路径：本轮 query → 主 Agent 记忆中间件取偏好（dislike 全量 + like 按向量 top-k）注入 prompt → 编排器用 `material_exclusion_tags` 正则识别全面材质否定、写入 ShoppingContext → 模型调 `product_search_tool`（可能自己也传 excluded_material_tags，也可能不传）→ 工具层合并两个来源（`product_search_tool.py:120-124` 去重）→ 查询词嵌向量 → Qdrant 近邻 → 材质/预算/目的国结构化过滤 →（rerank 配置了才精排）→ top_k 卡片。embedding 挂：召回退 `keyword_2gram`，`recall_strategy` 如实标注。
2. 要点：召回层加一个 embedding 客户端调用（替换 keyword_2gram 为阶梯的顶层）；身份应**新建**（训练环境的端点/型号是自己的，身份哈希天然分开，不会误用线上缓存）；旧数据谨慎混训——观察分布变了（能召回跨语言/同义商品），混训前先跑一组固定任务的召回分布对比；用 `recall_strategy` 字段统计新旧 rollout 各走哪层，作为「环境升级生效」的证据。

<a id="oral-7"></a>

## 60 秒口述与面试追问

**口述**：Globex 的检索增强有三个落点——商品召回（3700 商品文本嵌向量进 Qdrant，叠加预算/目的国/材质结构化过滤，召回有三级降级链）、品类知识库（45 篇文档切块入库，检索带可回答性判定和证据边界守卫，防检索变幻觉）、偏好记忆（向量 top-k 挑相关偏好注入，但 dislike 黑名单永不截断——漏黑名单是安全问题不是相关性问题）。形态上前者是模型主动调工具、后者是编排层注入，控制权位置不同。对做 RL 的人，这些检索工具就是环境：我们训练推演的环境只接了关键词兜底，和线上不同构，检索质量直接决定奖励分布。

**追问**：
- *为什么 dislike 不走 top-k？*（安全问题 vs 相关性问题——向量相似度低≠可以违反；规则性保证应在环境层硬编码，不交给 policy 或检索概率。）
- *embedding 服务挂了怎么办？*（三级降级：关键词 2-gram 兜底 + `recall_strategy` 留痕；建库失败不阻塞启动，增量补齐。）
- *怎么防止 RAG 检索不到还硬答？*（可回答性分数阈值 + 证据边界守卫（未来事实/缺参数/未证实来源拒答）+ 结果字段留痕，让上层能判定证据等级。）
- *训练和线上的检索层不同构会怎样？*（观察分布偏移：训练召不回的线上能召回，policy 学到的「问法→结果」映射失效；用 recall_strategy 之类留痕字段可诊断。）

<a id="sources-7"></a>

## 源码与实测依据

- 商品召回：`app/application/usecases/catalog_search.py`（召回阶梯 `:12`、rerank `:185-187`、关键词兜底 `:402`）；`app/infrastructure/vector/qdrant_product_index.py`、`embedding_identity.py`、`index_bootstrap.py`；`app/domain/catalog/product.py:69 searchable_text`（八字段）。
- 知识 RAG：`app/infrastructure/rag/category_knowledge.py`（KnowledgeBase 构造 `:193`、建库管线 `:219-243`、`policy_fact_status` `:53-75`）、`knowledge_retrieval.py`（边界守卫 `:8-24`、标题定位 `:31-52`）。
- 偏好召回：`app/application/memory/preference_selector.py`（dislike 不截断、`material_exclusion_tags`）；`app/application/agents/main_agent.py:130-133`（BuyerMemoryMiddleware 挂载）与 `middleware.py:22`（按 query 的 top-k 选择）；`app/application/agents/orchestrator.py:670,681`（ContextVar 注入；`:681` 为非语义 store 的旧路径）；`app/application/tools/product_search_tool.py:120-124`（双通道合并去重）。
- rerank：`app/infrastructure/rerank/http_reranker.py`（默认 `RERANKER_BASE_URL` 空 = 关闭）。
- 实测证据（详见仓库根 `01跑通记录-草稿.md`，本地）：场景 2（过滤三原因、`recall_strategy`）、场景 11/12（偏好全链、双通道排除）；三级 embedding 对照——前两级（随机/词袋）为免费本地运行〔假网关，data/〕，第三级（真语义）为真实模型运行〔text-embedding-v3，data-real〕；知识洞察实测含工具原始返回回放（journal `session-cli-1104183047-run-1`，2026-11-04）。
