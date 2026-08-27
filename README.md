# langcompress

> 面向 LangGraph / LangChain agent 的生产级上下文压缩中间件。开源、可插拔、五级分层 token 压缩。版本：v0.1。

`langcompress` 提供一个 `CompressionMiddleware`，渐进式压缩对话历史（内容过滤 → token 剪枝 → 引用替代 → 语义摘要 → 外部存储），在不破坏 agent 决策能力的前提下最大化 token 利用率。

## 安装

```bash
# 核心包（仅抽象层，langchain-core + pydantic）
pip install langcompress

# 带 LangGraph/LangChain 中间件适配器（推荐）
pip install langcompress[middleware]

# 全量安装
pip install langcompress[all]
```

| Extra | 附加能力 |
|---|---|
| `middleware` | `CompressionMiddleware` 适配器（需要 `langchain`） |
| `tiktoken` | 精确 token 计数 |
| `llm` | 通过 `langchain-openai` 进行 LLM 摘要 |
| `all` | 以上全部 |

## 最小用法

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langcompress import CompressionConfig, CompressionMiddleware

mw = CompressionMiddleware(CompressionConfig(
    summary_model=ChatOpenAI(model="gpt-4o-mini"),
    token_threshold=0.8,   # 达到模型输入窗口的 80% 时触发压缩
    keep_recent=6,
))
agent = create_agent(model=..., tools=[...], middleware=[mw])
```

四个扩展点 hook（`summary_message_builder`、`summary_llm_config_provider`、
`post_compress_hook`、`should_summarize_hook`）加上可选的 `content_classifier`，
让宿主项目无需 fork 本包即可适配框架特定关注点（前端同步、消息标记、
业务场景路由）。

## 摘要模型：独立、便宜、快——绝不是 agent 的主模型

`summary_model` 是**必填项**：没有默认值、没有环境变量兜底、也不会隐式复用
agent 的主 LLM——不配置它就是硬错误，因此静默透传主模型是不可能的。宿主在
构造时传入一个独立的摘要模型：要么是配置好的 `BaseChatModel` 实例，要么是
模型标识字符串（由父中间件经 `init_chat_model` 解析）。本包自身零 LLM 配置——
不读取任何模型环境变量、不创建模型实例、不绑定厂商（`langchain-openai`
等仅为可选的 `[llm]`/`[all]` extras）。遵循项目约定：标量来自环境变量，
类型与可调用对象来自构造参数——模型实例属于后者，绝不会从环境隐式加载。

选型原则：**能力就低、成本就低**——mini/flash/haiku 级模型（上面示例中的
`gpt-4o-mini` 是刻意的，不是占位符）是推荐默认：

- **成本**——摘要是纯开销路径：输入 ≈ `trim_tokens_to_summarize`（默认
  4000 tokens），每次压缩 0-2 次调用，无用户可见输出。八段式模板的质量
  门槛足够低，便宜模型即可达标。
- **延迟**——压缩运行在同步的 `before_model` 关键路径上；一个快速的独立
  模型能把触发回合的额外停顿从秒级降到亚秒级。
- **故障隔离**——摘要调用消耗自己的限流/配额而非主模型的，且摘要失败已有
  完整的兜底链（Plan A 用更简单的提示词重试 → Plans B/D/C 结果级降级）；
  架构本身设计为可容忍它。

之前写 `CompressionConfig(summary_model=agent_llm, ...)` 的宿主应改为
构造独立模型，例如
`CompressionConfig(summary_model=ChatOpenAI(model="gpt-4o-mini"), ...)`。

## L2 源头压缩（`wrap_tool_call`）

大型工具输出（网页、PDF、API 响应）在源头就被外部化为轻量引用——由独立的
`ToolCallExternalizerMiddleware` 完成——因此它们永远不会膨胀消息历史。中心化
L3 摘要（`CompressionMiddleware`）与之并行运行；两个中间件组合无冲突
（设计 §4.3 / §12.3）。

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langcompress import (
    CompressionConfig,
    CompressionMiddleware,
    FilesystemExternalizer,
    ToolCallExternalizerMiddleware,
)


@tool
def fetch_page(url: str) -> str:
    """返回一个不应膨胀上下文的大型页面正文。"""
    ...


# L2 在源头：大型工具结果在进入历史前被替换为引用（file:// URI）。
# L3 照常摘要。
agent = create_agent(
    model=...,
    tools=[fetch_page],
    middleware=[
        CompressionMiddleware(CompressionConfig(summary_model=..., token_threshold=0.8)),
        ToolCallExternalizerMiddleware(FilesystemExternalizer()),
    ],
)
```

`ToolCallExternalizerMiddleware` 是一个**骨架**——它不内置任何工具特定的
压缩逻辑（设计 §12.3）。它把超大的 `ToolMessage` 内容替换为引用字符串，
保留 `tool_call_id` / `name`，并把引用存入
`ToolMessage.additional_kwargs["external_ref"]`。需要时可在
`post_compress_hook` 中通过 `aggregate_external_refs` 辅助函数把这些引用
聚合进 state；默认情况下中间件不触碰 state。`external_refs` state 通道带
**dict-merge reducer**，因此 hook 只需返回*本次*压缩的新引用，reducer 会在
多次压缩间累积（不会在 `REMOVE_ALL_MESSAGES` 时因 last-write-wins 丢失）：

```python
def post_compress(state, result):
    return {**result, "external_refs": aggregate_external_refs(result)}
```

## 摘要质量校验 + 优雅降级（设计 §8.2）

当摘要 LLM 行为异常（空输出、`Error generating summary:` 字符串、过短 /
畸形摘要）时，`CompressionMiddleware` 不再把坏摘要塞进上下文——而是先校验，
再降级：

- **Plan A——重试**：换用更简单的 `FALLBACK_SUMMARY_PROMPT`（由
  `QualityValidator.validate(...).suggested_plan == "A"` 驱动）。
- **Plans B / D / C——结果级替换**：通过可插拔的 `DegradationStrategy`
  （默认 `D → B → C`）：配置了 `Externalizer` 时把待摘要头部外部化
  （D，可找回），否则加宽保留的近期窗口（B，无 I/O），否则截断到
  `min_keep` 条近期消息（C，永不失败）。策略本身崩溃则回退到原始结果，
  因此 agent 永远继续运行。

```python
from langcompress import (
    CompressionConfig, CompressionMiddleware,
    HeuristicQualityValidator, FilesystemExternalizer,
)

mw = CompressionMiddleware(CompressionConfig(
    summary_model=...,
    token_threshold=0.8,
    keep_recent=6,
    # 可选的更严格质量门（默认配置为 no-op）
    quality_min_reduction_ratio=0.5,
    # 启用 Plan D：失败时外部化头部而非截断
    degradation_externalizer=FilesystemExternalizer(),
    degradation_min_keep=3,
))
```

默认的 `HeuristicQualityValidator` 刻意保守（只标记明确的失败），因此良构
摘要总是通过；可选参数（`quality_min_reduction_ratio`、
`quality_require_segments`）让更严格的宿主无需 fork 即可收紧门槛。
`QualityValidator` 与 `DegradationStrategy` 是 ABC——换入 LLM-as-judge
校验器或自定义降级链的方式与换 `Externalizer` / `Summarizer` 相同。

## 管线健壮性

压缩管线的四项非破坏性增强：

1. **`external_refs` dict-merge reducer**（设计 §13.1）——引用现在跨压缩
   累积而非 last-write-wins，因此 `REMOVE_ALL_MESSAGES` 替换或并发工具
   调用不再丢失早先的引用。宿主的 `post_compress_hook` 只返回*本次*
   压缩的新引用，reducer 把它们合并进 `state["external_refs"]`。
2. **真正的非阻塞异步外部化器**——`Externalizer.aexternalize` /
   `aretrieve` 现在默认 `asyncio.to_thread(self.externalize, ...)`，因此
   文件系统外部化器不会阻塞事件循环。零新增依赖；语义上等价于 `aiofiles`
   覆写。后端真正异步原生的子类仍可直接覆写异步方法。
3. **`aggregate_external_refs` 扫描全部消息**——此前扫描仅限
   `ToolMessage`，会静默丢弃 L3 Plan-D 引用（加盖在摘要形态的
   `HumanMessage` 上）。现在从任意 `BaseMessage` 收集 `external_ref`，
   因此 L2（源头 `wrap_tool_call`）与 L3（Plan-D 降级）的引用都会被聚合。
   L2 的 `ToolMessage` 路径（值 = 工具名）不变；Plan-D 的 `HumanMessage`
   引用值为 `""`（无 `name` 属性）。`additional_kwargs["external_ref"]`
   是 **langcompress 保留键**。
4. **降级 / 质量校验可观测性**——每次 Plan-A 重试、Plan-B/C/D 降级以及
   降级策略失败都会通过 `langcompress.middleware` logger 发出一条 INFO
   日志（层级化——调高该 logger 级别即可只静音适配器）。降级结果还会在
   首条非哨兵消息上携带
   `additional_kwargs["degradation"] = {"plan", "reason"[, "external_ref"]}`，
   因此事件在 state/checkpoint 中可观测。`additional_kwargs` 绝不会渲染
   进 LLM 提示词（已验证：`get_buffer_string` 只读取 `function_call`）。

完整设计见 [docs/design.md](docs/design.md)，可运行示例见
`examples/basic_usage.py`。版本历史见 [CHANGELOG.md](CHANGELOG.md)，
开发环境搭建、架构约束与 PR 约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 外部化内容生命周期（设计 §18）

外部化的数据块不会永久累积：围绕两阶段删除构建的保留体系确保清理绝不
静默破坏"有损但可恢复"的承诺（§3.3）：

```python
from datetime import timedelta
from langcompress import (
    FilesystemExternalizer, RetentionManager, TTLPolicy, NullPolicy,
    collect_live_refs,
)

ext = FilesystemExternalizer(base_dir="./cache")
# 环境变量驱动的标量：LANGCOMPRESS_RETENTION_TTL_HOURS（未设 → None → 禁用）。
policy = TTLPolicy.from_env() or NullPolicy()
manager = RetentionManager(ext, policy)

# 由宿主决定何时运行（会话结束、cron、post_compress_hook）：
manager.run(keep_refs=collect_live_refs(state, messages))
```

每个引用的状态机：`Active → Stale（软删除进 .trash/，仍可读取）→
Purged（宽限期后，不可逆）`。五条不变量成立（设计 §18.2）：根集引用绝不
被清除（陈旧的自动恢复）、一次运行中驱逐的记录绝不在同一次运行中被清除、
任何环境变量都无法单独开启清理、不支持生命周期的后端静默退化为 no-op、
每次运行产出可审计的 `PurgeReport` 而非抛异常。

## L0 常驻内容过滤（设计 §4.1/§12.3）

L0 内容过滤器（`L0Filter`）已接入 `before_model`，**每一回合**都运行——
无论 L3 摘要阈值是否达到——在纯机械噪声到达模型或 checkpoint 之前将其
剥离。L0 是纯规则（无 LLM）：丢弃空消息、丢弃相邻重复消息、剥离推理
内容、合并相邻同类型消息。

```python
from langcompress import CompressionConfig, CompressionMiddleware

# L0 默认开启。每次 before_model 调用时在内存中运行；仅当 L0 确实
# 改动了内容时才回写 state（无改动时零 checkpoint 开销）。L3 触发时，
# L0 的清理免费搭 L3 全量替换的便车，两者绝不会重复写入。
mw = CompressionMiddleware(CompressionConfig(
    summary_model=...,
    token_threshold=0.8,
    keep_recent=6,
    l0_enabled=True,            # 默认值；设为 False 可完全关闭 L0
))
```

**`drop_reasoning_kwargs` 操作。** 两种形态的推理内容现有专属 L0 操作：

- `drop_reasoning_parts`——从 content-list-of-parts 消息中剥离
  `{"type":"thinking"/"reasoning"}` 条目（Gemini-CLI 风格）。
- `drop_reasoning_kwargs`——移除
  `additional_kwargs["reasoning_content"]` / `["reasoning"]`，即
  GLM-4.6 / GLM-5.2 / DeepSeek-R1 发出的 OpenAI 兼容思考模式载荷。
  两者默认均为 `True`。

```python
from langcompress import L0Filter
# 四个 L0 操作可独立调节（均默认 True）：
L0Filter(drop_reasoning_kwargs=True, drop_reasoning_parts=True,
         drop_empty=True, drop_duplicates=True, merge_adjacent=True)
```

`CompressionConfig.l0_filter` 接受自定义 `L0Filter`（或任意
`CompressionStage`）供宿主调节；`l0_enabled=False`（或环境变量
`LANGCOMPRESS_L0_ENABLED=0`）关闭 L0 并回到仅 L3 的行为。

**缺陷修复：** `_merge_adjacent_same_type` 此前合并两条相邻同类型消息时
只克隆第一条消息的属性，静默丢弃第二条的 `tool_calls` / `tool_call_id`
——破坏 AI/Tool 配对不变量。携带工具元数据的消息现在被排除在合并之外。

## 中间件顺序（设计 §12.3）

压缩中间件可与框架/handoff 中间件组合。加载顺序规则是**生产者先于
压缩者**——`before_model` 按加载顺序运行，因此任何注入/改写消息的组件
都应排在压缩器之前，而 `wrap_tool_call` 源头外部化的运行时机应保证其
大型 `ToolMessage` 输出在下一次 `before_model` 前被外部化：

```python
from copilotkit.langgraph import CopilotKitMiddleware
# from your_handoff_pkg import HandoffMiddleware     # 如使用 handoff

middleware = [
    # dynamic_prompt,           # 1.（宿主）提示词组装——产出 SystemMessage
    # HandoffMiddleware(...),   # 2.（框架）路由/handoff——产出消息
    # CopilotKitMiddleware(...),# 3.（框架）前端同步——产出消息
    CompressionMiddleware(CompressionConfig(...)),          # 4. L0 + L3
    ToolCallExternalizerMiddleware(FilesystemExternalizer(  # 5. L2 源头
        base_dir="./cache",
    )),
]
agent = create_agent(model=..., tools=[...], middleware=middleware)
```

`wrap_tool_call`（L2）拦截任意 `BaseTool`，包括 MCP 加载的工具
（`langchain-mcp-adapters.get_tools()` 返回标准 `BaseTool` 实例），因此
MCP 工具与原生工具的外部化方式相同——无需特殊接线。

每个 agent 独立加载自己的中间件（按 agent 配置），宿主决定确切顺序；
上述顺序是推荐默认。

## 许可证

[MIT](LICENSE)。全部运行时依赖均为宽松许可证（MIT / Apache-2.0 / BSD /
MPL-2.0 / PSF），依赖链中不存在任何 copyleft（GPL/AGPL/LGPL）代码。
