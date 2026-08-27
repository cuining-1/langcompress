# langcompress — Agent 上下文压缩中间件

> 面向 LangGraph / LangChain 架构的开源、可插拔、五级分层 token 压缩中间件。覆盖内容分类、触发策略、摘要管理、状态同步与工程化落地的完整方案。

**版本**: v0.1 · **日期**: 2026.08.19 · **定位**: 独立开源 Python 包（`pip install langcompress`）· **技术栈**: LangChain-Core · LangGraph · Middleware

> 注：§17 路线图中的 v0.1.0–v0.6.0 是开发期的里程碑代号；上述里程碑对应的全部能力均已包含在首个公开发布 v0.1.0（2026-08-27）中。

---

## 目录

1. [项目定位与设计目标](#1-项目定位与设计目标)
2. [行业背景与核心矛盾](#2-行业背景与核心矛盾)
3. [设计哲学](#3-设计哲学)
4. [五级压缩管线](#4-五级压缩管线)
5. [触发策略](#5-触发策略)
6. [内容分类与场景化策略](#6-内容分类与场景化策略)
7. [结构化摘要格式](#7-结构化摘要格式)
8. [保留规则与优雅降级](#8-保留规则与优雅降级)
9. [上下文重构与恢复](#9-上下文重构与恢复)
10. [业界方案对比](#10-业界方案对比)
11. [扩展点设计（四 Hook 接口）](#11-扩展点设计四-hook-接口)
12. [LangGraph 中间件集成](#12-langgraph-中间件集成)
13. [Message Reducer 与状态同步](#13-message-reducer-与状态同步)
14. [包结构与依赖](#14-包结构与依赖)
15. [市面已有组件分析](#15-市面已有组件分析)
16. [开源工程化约束](#16-开源工程化约束)
17. [实施路线图](#17-实施路线图)
18. [外部化内容生命周期管理](#18-外部化内容生命周期管理)

---

## 1 项目定位与设计目标

### 1.1 一句话定位

`langcompress` 是一个为 LangGraph / LangChain agent 提供生产级上下文压缩能力的开源 Python 包，通过五级分层管线渐进压缩对话历史，在不破坏 agent 决策能力的前提下最大化 token 利用率。

### 1.2 核心设计目标

| 目标 | 含义 | 反目标（不做什么） |
|---|---|---|
| **框架中立** | 核心仅依赖 `langchain-core`，不绑定 LangGraph 运行时、不绑定 CopilotKit、不绑定特定 LLM 厂商 | 不做"只为某个项目服务"的隐式假设 |
| **可插拔管线** | 五级压缩每一级是独立 `CompressionStage`，可替换、跳过、自定义 | 不强制用户使用全部五级 |
| **扩展点驱动** | 宿主项目的框架特定适配（流式过滤、前端同步、消息标记等）通过注册 hook 完成，不修改包源码 | 不在核心包内出现任何框架特定耦合 |
| **最小核心 + extras** | `pip install langcompress` 只装骨架；LLM、tiktoken、Redis、向量库按需 extras | 不让用户为核心不需要的能力买单 |
| **延迟导入** | 可选依赖不在模块顶层 import，使用时动态导入 | 不因 optional import 失败导致核心功能不可用 |

### 1.3 明确不做的事

为避免范围蔓延，以下能力**不在本包范围内**，由宿主项目自行实现：

- **前端显示层适配**：CopilotKit 的 `MESSAGES_SNAPSHOT` / `STATE_SNAPSHOT` 同步、摘要卡片定位、流式事件过滤等，属于宿主项目与前端框架的契约，不属于压缩中间件本身
- **业务场景特定分类**：本包提供内容分类的**机制**（按消息类型、大小、来源路由），但不内置"飞书文档""多维表格""A2UI"等业务场景的判断规则——这些由宿主项目通过 hook 注入
- **持久化存储后端的运维**：Redis / 向量库 / 文件系统的部署、备份、监控由宿主负责，本包只提供客户端适配
- **模型厂商特定优化**：不内置 GLM / Claude / GPT 的特殊参数处理，也不创建任何模型实例、不读取任何模型类环境变量（`langchain-openai` 等仅为 extras 可选依赖）；摘要模型由宿主通过 `summary_model` 构造参数显式传入配置好的 `BaseChatModel`（或供 `init_chat_model` 解析的模型标识字符串），见 §11.5

---

## 2 行业背景与核心矛盾

Agent 在多轮对话中持续累积上下文，token 消耗线性增长直至触及模型上下文窗口上限。业界普遍采用基于对话轮次和 token 阈值的压缩机制，但各家做法差异显著，缺乏统一的设计范式。

核心矛盾在于：**Agent 需要完整历史状态来决策下一步，但我们无法预知当前丢弃的某个细节是否在未来关键时刻变得至关重要** [1](#ref-1)。

业界三大流派的策略截然不同：

| 流派 | 代表产品 | 核心策略 | 致命短板 |
|---|---|---|---|
| 永不丢失 | Manus | 文件系统作为终极上下文，用引用替代删除 | 依赖模型学会主动读写文件 |
| 极限压榨 | Claude Code | 92% 阈值触发，八段式结构化摘要 | 紧急压缩质量不稳定 |
| 稳健保守 | Gemini CLI | 70% 阈值，五段式摘要，thought 过滤 | 频繁压缩、开销大 |

单一策略都有短板。**完善的设计必须组合多种策略，分层渐进执行**，这是本项目的设计出发点。

---

## 3 设计哲学

本项目遵循三个核心原则：**分层渐进、内容感知、有损可恢复**。

### 3.1 分层渐进

轻量操作（内容过滤、token 剪枝）高频执行、成本低；重量操作（语义摘要、外部化存储）低频执行、成本高但压缩比大。通过分层让不同类型的内容走不同压缩路径，避免"一刀切"的信息丢失风险。

### 3.2 内容感知

压缩前先分类内容类型，不同类型走不同管线。工具结果用引用替代，推理过程保留结论压缩过程，用户消息以某种形式始终保留。这是 Claude Code 八段式摘要和 Gemini CLI 精选历史的核心思想 [2](#ref-2)。

### 3.3 有损可恢复

压缩操作本身可能是有损的，但通过文件系统引用、向量库检索、摘要链等机制，被压缩的内容在需要时可以恢复。这让"有损"变得"可恢复"，降低信息丢失风险。

> **设计原则**：引用优于保留——能用路径/URL/ID 引用的，不保留完整内容（Manus 哲学）。程序化处理结构化数据——能用 SQL/code 处理的，不让 LLM 读全表（TableZoomer/CoAgt 哲学）。结论优于过程——推理链和思考过程保留结论，压缩过程（Claude Code 哲学）。

---

## 4 五级压缩管线

五个层级从轻到重渐进执行，每一层只在需要时触发下一层。

```
L0 内容过滤 → L1 Token 剪枝 → L2 引用替代 → L3 语义摘要 → L4 外部化存储
(5-15%)      (20-40%)        (60-90%)       (70-95%)       (90-99%)
```

### 4.1 L0：内容过滤

- **成本**：极低，压缩 5-15%，无需 LLM 调用，纯规则过滤
- **操作**：移除空响应、中断的流式输出、重复消息；过滤 thinking/reasoning 类型的 part（Gemini CLI 的做法）[1](#ref-1)；合并相邻的同类内容块
- **本包实现**：纯 Python 启发式，零依赖，可委托 `langchain-collapse`

### 4.2 L1：Token 级剪枝

- **成本**：低，压缩 20-40%
- **操作**：用小模型（如 GPT-2/LLaMA-7B）计算每个 token 的困惑度——困惑度低的 token 信息量小，删除；困惑度高的 token 信息量大，保留。这是 LLMLingua 的思路，不改变语义，只去掉"废话" [3](#ref-3)
- **本包实现**：后置版本，MVP 不含（需要小模型依赖）；可委托 `ContextZip`

### 4.3 L2：引用替代

- **成本**：低，单条压缩 60-90%
- **操作**：大输出（网页内容、PDF、API 响应）替换为元信息引用；保留 URL、文件路径、请求参数，移除完整内容；维护本地缓存，需要时通过引用重新加载
- **这是 Manus 的核心策略**：文件系统作为"终极上下文" [1](#ref-1)
- **本包实现**：可委托 `distil` 的 reversible digest；源头压缩（`wrap_tool_call` hook）由宿主在工具层注入

### 4.4 L3：语义摘要

- **成本**：中，压缩 70-95%
- **操作**：LLM 生成结构化摘要替换历史对话；使用八段式结构化模板（见第 7 节）；保留用户消息、关键决策、错误历史
- **这是 Claude Code 和 Gemini CLI 的核心策略** [4](#ref-4)
- **本包实现**：自建核心能力，八段式模板 + 摘要链管理 + 质量校验；摘要模型由宿主独立配置、按"能力就低、成本就低"选型（§11.5）

### 4.5 L4：外部化存储

- **成本**：中，压缩 90-99%
- **操作**：将摘要卸载到文件系统或向量库；上下文只保留轻量引用；支持即时检索（just-in-time retrieval）
- Anthropic 称之为"结构化笔记"策略 [5](#ref-5)
- **本包实现**：抽象 `Externalizer` 基类 + 文件系统后端（MVP）+ Redis / 向量库后端（后续）

---

## 5 触发策略

业界三种触发方式各有优劣：

| 策略 | 代表 | 阈值 | 优势 | 风险 |
|---|---|---|---|---|
| 极限压榨 | Claude Code | 92% | 最大化上下文利用率 | 紧急压缩质量不稳 |
| 稳健保守 | Gemini CLI | 70% | 压缩从容、质量高 | 频繁压缩、开销大 |
| 可配置 | Claude SDK | 默认 100k | 灵活适配场景 | 需调参经验 |

### 5.1 推荐：多信号融合触发

```
触发条件 = Token阈值(主) OR 轮次阈值(辅) OR 预测触发(前) OR 内容超限(即)
```

- **Token 阈值**：80% 触发完整压缩（L0-L4），70% 触发轻量压缩（L0-L1）。取 Claude Code 和 Gemini CLI 的中间值 [5](#ref-5)
- **轮次阈值**：每 N 轮执行一次 L0-L1 轻量清理，避免积累无效内容
- **预测触发**：根据当前任务剩余步骤估算，如果预测剩余 token 需求 + 当前用量 > 窗口上限，提前触发 L2-L3
- **内容超限即触发**：当单条 observation 超过阈值（如 10k token），立即对该条执行 L2 引用替代 [6](#ref-6)

### 5.2 防重复触发

压缩后保留的最近 N 条消息里若含大体积 ToolMessage（如长文档内容），下一轮 LLM input 仍可能超 token 阈值，导致每轮 `before_model` 都重复触发压缩——而压缩同样内容意义不大。

本包的策略：首条消息已是摘要时，跳过 token/fraction 维度触发，只按消息条数维度触发。等消息条数累积到阈值（默认 50 条）再压缩更合理。

---

## 6 内容分类与场景化策略

不同内容类型的压缩策略不同。根据压缩力度，将 15 种内容类型分为四个层级。

### 6.1 Tier 1：永不压缩

| 场景 | 压缩级别 | 核心策略 |
|---|---|---|
| System Prompt / 人格设定 | NEVER | 永不压缩，定期审查精简。追求"最小但充分"的信息集 [5](#ref-5) |
| 注入的工具定义 | NEVER | 动态裁剪未使用工具，而非压缩。"如果人类工程师无法明确说哪个工具该用，AI Agent 更做不到" [5](#ref-5)。MCP 工具应按任务语义筛选 |
| 当前用户消息 | NEVER | 原样保留，无任何压缩 |
| Few-shot 示例 | L1 | 精选 3-5 个典型样本，按任务阶段替换 [5](#ref-5) |

### 6.2 Tier 2：轻压（L0-L1，最多到 L2）

| 场景 | 压缩级别 | 核心策略 |
|---|---|---|
| 正常语境对话 | L1-L3 | 滑动窗口 + 用户消息保留 + 语义摘要。最近 N 轮原样保留（N=5-10），历史用户消息全部保留，助手回复做 L1-L3 |
| 读取既定步骤/摘要 | L0 | 已是压缩态，仅做 L0 过滤。不二次摘要。用状态标记法：pending/in_progress 保留完整，completed 压缩为一行 |
| 读取本地文件（代码/文本） | L2-L3 | 代码：路径+行号引用 + 按需重载 + 符号表索引。文本：路径引用 + 片段级检索 + 向量索引 |
| 工具返回结果（中小型） | L2-L3 | 字段过滤移除无关字段，JSON→Markdown 表格省 20-50%，列式 JSON 省 35-45%，LLM 生成结构化摘要 [7](#ref-7) |

### 6.3 Tier 3：中压（L2-L3）

| 场景 | 压缩级别 | 核心策略 |
|---|---|---|
| 思考环节 | L1-L4 | 保留结论，压缩推理链。最近 1-2 轮完整保留，3-5 轮前 L3 压缩，5 轮前 L4 外部化。Gemini CLI 直接过滤 thought part [1](#ref-1) |
| RAG 检索片段 | L3 | 保留 chunk ID 引用，多个片段合并为综合摘要 |
| 子 Agent 返回 | L3 | 本已是摘要态，保留结论+引用，不保留完整推理过程 [7](#ref-7) |

### 6.4 Tier 4：重压（L2-L4 激进压缩）

| 场景 | 压缩级别 | 核心策略 |
|---|---|---|
| 读取结构化数据（宽表） | L4 | Schema 替代全表 + 查询感知列裁剪 + Program-of-Thoughts 将查询转 SQL/代码 [8](#ref-8) [9](#ref-9) |
| 联网查询 | L4 | URL 引用 + 简短描述 + 本地缓存。完整网页内容不保存，需要时通过 URL 重载（Manus 模式）[1](#ref-1) |
| 工具返回结果（大型） | L4 | 请求参数引用 + 文件系统卸载 + 渐进式加载。保留 API endpoint+参数，大结果写临时文件 [7](#ref-7) |
| 错误堆栈 / 异常追踪 | L4 | 保留错误码和根因，压缩完整堆栈帧。相同错误去重 [4](#ref-4) |
| 多模态内容 | L4 | 图片/音频保留媒体路径+一句话描述，不保留完整 OCR/转写文本 |
| 会话检查点 | L4 | 外部化到文件系统，上下文只保留引用 |
| 用户偏好 / 个性化记忆 | L2 | 结构化键值对，外部化存储 |

> **关键原则**：用户消息神圣不可侵犯——所有用户输入以某种形式保留（Claude Code 八段式第 6 段）。活跃工具动态裁剪——工具定义不压缩，但未使用工具应移除。一次性内容即焚——渲染后即移除。
>
> **本包提供分类的机制，不内置业务场景规则**：上表中的"飞书文档""A2UI"等业务特定场景由宿主项目通过 `content_classifier` hook 注入分类逻辑，本包只提供路由框架和默认的按消息类型/大小分类。

---

## 7 结构化摘要格式

综合 Claude Code 的八段式和 Gemini CLI 的五段式，本包采用增强版八段结构 [1](#ref-1) [4](#ref-4)：

| # | 段落 | 作用 |
|---|---|---|
| 1 | Primary Request and Intent | 主要请求和意图（永不丢失） |
| 2 | Key Technical Concepts | 关键技术决策和约束 |
| 3 | Files and Code Sections | 文件和代码段引用 |
| 4 | Errors and Fixes | 错误和修复（防重复踩坑） |
| 5 | Problem Solving | 问题解决的关键路径 |
| 6 | All User Messages | 所有用户消息（压缩形式保留） |
| 7 | Pending Tasks | 待处理任务 |
| 8 | Entity State（新增） | 关键实体及其当前状态 |

第 8 段"Entity State"是本项目的增强项。很多 agent 压缩后丢失了实体追踪（如"用户叫张三"），导致重复询问。将实体状态独立成段可避免此问题。

摘要模板支持自定义，宿主项目可通过 `summarizer_factory` hook 注入场景化模板（如工单系统的"已完成工单/进度状态/下一步操作"三段式）：

```python
compaction_control = {
    "enabled": True,
    "context_token_threshold": 80000,  # 80% of 100k
    "summary_template": "请创建摘要，保留：1.已完成工单 2.进度状态 3.下一步操作"
}
```

---

## 8 保留规则与优雅降级

### 8.1 保留规则

以下内容在压缩过程中以某种形式始终保留：

- 所有用户消息（即使其他内容压缩，用户消息以某种形式保留）
- 最近 N 轮对话原文（N=5-10，维持上下文连续性）
- 当前任务状态和待办事项
- 关键决策及其理由
- 错误历史
- 活跃文件引用和工具状态

### 8.2 优雅降级

当压缩质量不达标时的降级链（Claude Code 的"永不放弃"设计）[1](#ref-1)：

```
质量校验失败
  ├→ Plan A: 自适应重压缩（调整参数重试）
  ├→ Plan B: 混合模式（压缩旧内容，完整保留最近交互）
  ├→ Plan C: 保守截断（最坏情况，保证系统继续运行）
  └→ Plan D: 外部化兜底（摘要失败则卸载到文件系统）
```

---

## 9 上下文重构与恢复

压缩后上下文重构为 [10](#ref-10)：

```
重构上下文 = [结构化摘要] + [最近N轮原文] + [当前任务状态] + [活跃引用]
```

恢复机制让"有损"变得"可恢复"：

- **文件系统引用**：外部化内容通过路径随时重载（Manus 模式）
- **向量库检索**：过往摘要支持语义检索
- **摘要链**：对摘要再压缩时保留前序摘要，形成层级上下文
- **即时重载**：当对话引用某个已压缩资源时，自动通过引用拉回完整内容
- **缓存层**：最近访问的压缩内容暂存，避免频繁重载

### 9.1 摘要链管理

每次压缩时面对的是"已压缩的历史 + 新消息"。核心原则是**不生成"摘要的摘要"，而是生成新的综合摘要**——LLM 收到旧摘要的内容（当作上下文）+ 旧摘要之后的消息，输出全新综合摘要替代旧摘要。

```
Turn 5:  [summary_v1, msg_4, msg_5]
         压缩 → [summary_v2, msg_5, msg_6]

Turn 8:  [summary_v2, msg_5..7, msg_8]
         压缩 → summary_v3 整合了 summary_v2 的内容 + msg_5..7
         结果: [summary_v3, msg_7, msg_8]

Turn 12: [summary_v3, msg_7..11, msg_12]
         压缩 → summary_v4 整合了 summary_v3 的内容 + msg_7..11
         结果: [summary_v4, msg_11, msg_12]
```

这样信息不会逐层衰减。

---

## 10 业界方案对比

| 维度 | Manus | Claude Code | Gemini CLI | LLMLingua | langcompress |
|---|---|---|---|---|---|
| 触发阈值 | 不删除 | 92% | 70% | 按需 | 80%+预测 |
| 压缩策略 | 外部化引用 | 结构化摘要 | 结构化摘要 | Token 剪枝 | 五级管线 |
| 摘要结构 | 无(保留引用) | 八段式 | 五段式 | 无 | 增强八段式 |
| 可恢复性 | 完全可恢复 | 有损 | 有损 | 有损 | 分层可恢复 |
| 降级机制 | 无需(不删除) | 四级降级 | 多层过滤 | 无 | 四级降级 |
| 核心哲学 | 永不丢失 | 极限压榨 | 稳健保守 | Token 级精简 | 分层渐进 |
| 框架中立 | — | — | — | — | ✅ 仅依赖 langchain-core |
| 可插拔管线 | — | — | — | — | ✅ 每级可替换 |
| 开源 | — | — | — | ✅ | ✅ |

---

## 11 扩展点设计（四 Hook 接口）

本包与宿主项目的解耦通过四个 hook 接口实现。**这四个 hook 的 API 稳定性是本包的命门**——一旦发布，签名变更即为 breaking change。在发布前必须用真实适配场景压测稳定。

### 11.1 设计原则

1. **hook 是公开 API，私有方法不暴露**：本包内部继承 LangChain `SummarizationMiddleware` 的 `_` 前缀私有方法（如 `_build_new_messages`、`_acreate_summary`、`_should_summarize`）全部翻译为公开 hook 调用点，宿主项目不再继承私有方法
2. **默认实现即开箱可用**：每个 hook 都有合理默认值，不注册任何 hook 也能跑通基础压缩
3. **hook 是纯函数或简单可调用对象**：不要求宿主继承特定基类，注册即可

### 11.2 四个 Hook

#### Hook 1: `build_summary_message(summary: str) -> BaseMessage`

- **作用**：构造承载摘要的消息对象
- **默认实现**：返回 `HumanMessage(content=summary)`（LangChain 父类行为）
- **宿主覆盖场景**：
  - 改用 `SystemMessage` 避免被前端渲染为用户消息
  - 注入标记字段（如 `additional_kwargs["__summarization__"] = True`）供前端识别
  - 注入触发位置标记（如 `triggered_by_user_id`）供前端按用户消息定位摘要卡片
- **签名稳定性约束**：入参为摘要字符串，出参为 `BaseMessage` 子类实例；标记字段通过 `additional_kwargs` 传递，不扩展函数签名

```python
from langchain_core.messages import SystemMessage
from langcompress import CompressionConfig

def my_summary_builder(summary: str) -> SystemMessage:
    return SystemMessage(
        content=f"Here is a summary of the conversation to date:\n\n{summary}",
        additional_kwargs={"__summarization__": True},
    )

config = CompressionConfig(summary_message_builder=my_summary_builder)
```

#### Hook 2: `get_summary_llm_config() -> RunnableConfig`

- **作用**：返回摘要 LLM 调用时附加的 `RunnableConfig`（含 metadata）
- **默认实现**：返回空 dict
- **宿主覆盖场景**：
  - 注入 `emit-messages=False` 阻断摘要 LLM 的流式输出泄漏到前端
  - 注入 tracing 标签用于观测
- **签名稳定性约束**：无入参，出参为 `RunnableConfig`（dict 子类）；metadata 键名约定由宿主自管

```python
def my_llm_config() -> dict:
    return {"metadata": {"emit-messages": False, "lc_source": "summarization"}}

config = CompressionConfig(summary_llm_config_provider=my_llm_config)
```

#### Hook 3: `post_compress(state: dict, result: dict) -> dict`

- **作用**：压缩完成后、返回给 reducer 前的后处理钩子，可修改 result
- **默认实现**：原样返回 result
- **宿主覆盖场景**：
  - 给摘要消息打 `triggered_by_user_id` 标记（从 state 找最后一条 HumanMessage 的 id）
  - 同步外部状态（如通知前端、写审计日志）
- **签名稳定性约束**：入参为 LangGraph state 片段 dict 和压缩结果 dict，出参为处理后的 result dict；不允许抛异常（异常会被本包捕获并降级为原 result）

```python
from langchain_core.messages import HumanMessage

def my_post_compress(state: dict, result: dict) -> dict:
    new_messages = result.get("messages", [])
    for m in new_messages:
        if getattr(m, "additional_kwargs", {}).get("__summarization__"):
            triggered_by = next(
                (x.id for x in reversed(state.get("messages", [])) if isinstance(x, HumanMessage)),
                None,
            )
            if triggered_by:
                m.additional_kwargs["triggered_by_user_id"] = triggered_by
            break
    return result

config = CompressionConfig(post_compress_hook=my_post_compress)
```

#### Hook 4: `should_summarize(messages: list, total_tokens: int, base_decision: bool) -> bool`

- **作用**：覆盖默认触发判断
- **默认实现**：返回 `base_decision`（本包内置的多信号融合判断结果）
- **宿主覆盖场景**：
  - 防重复触发：首条已是摘要时跳过 token 维度，只按条数触发
  - 业务特定触发：如"飞书文档专项压缩"独立阈值
- **签名稳定性约束**：入参为消息列表、当前 token 数、默认判断结果，出参为布尔；`base_decision` 让宿主可在默认逻辑之上做"加强"而非"替换"

```python
def my_should_summarize(messages: list, total_tokens: int, base_decision: bool) -> bool:
    # 防重复触发：首条已是摘要则只按条数判断
    if messages and getattr(messages[0], "additional_kwargs", {}).get("__summarization__"):
        return len(messages) >= 50
    return base_decision

config = CompressionConfig(should_summarize_hook=my_should_summarize)
```

### 11.3 内容分类 Hook（可选，第五个）

除上述四个核心 hook 外，内容分类路由也通过 hook 注入，但它是可选的：

```python
def my_content_classifier(message: BaseMessage) -> str:
    """返回内容类型标签，用于路由到不同压缩管线。"""
    if is_feishu_doc(message):
        return "feishu_doc"
    return "default"

config = CompressionConfig(content_classifier=my_content_classifier)
```

本包内置默认分类器只按消息类型（Human/AI/Tool/System）和大小分类，不识别业务场景。

### 11.4 Hook 注册方式

所有 hook 通过 `CompressionConfig` 传入，构造 `CompressionMiddleware` 时一次性注册，运行期不可变（避免并发问题）：

```python
from langcompress import CompressionMiddleware, CompressionConfig

middleware = CompressionMiddleware(
    config=CompressionConfig(
        token_threshold=0.8,
        keep_recent=6,
        summary_model=summary_llm,  # 专职摘要模型（§11.5），不透传 agent 主模型
        summary_message_builder=my_summary_builder,
        summary_llm_config_provider=my_llm_config,
        post_compress_hook=my_post_compress,
        should_summarize_hook=my_should_summarize,
        content_classifier=my_content_classifier,
    )
)
```

### 11.5 摘要模型独立配置与选型原则

摘要 LLM 与 agent 主对话模型**分开配置**，不透传：`summary_model` 是
`CompressionConfig` 的必填构造参数，由上层应用在初始化时显式传入已配置好的
`BaseChatModel` 实例（或供 `init_chat_model` 解析的模型标识字符串）。依赖注入
机制不变——延续「标量走环境变量、类型与可调用对象走构造参数」的项目约定，
模型实例属于后者，永远不会从环境变量隐式加载。本包零 LLM 配置：不读取模型类
环境变量、不创建模型实例、不绑定任何厂商，`langchain-openai` 等仅为 extras
可选依赖。字段必填、无隐式回退——不配置即报错，杜绝"静默复用主模型"的不确定
行为。

选型原则：**能力就低、成本就低**（mini / flash / haiku 级专职快模型）：

| 动机 | 说明 |
|---|---|
| **成本** | 摘要生成是纯开销路径（输入约 `trim_tokens_to_summarize`，默认 4000 token，单次压缩 0-2 次调用），不产生用户可见价值；八段式结构化模板对模型能力要求低，便宜模型即可通过质量门槛 |
| **延迟** | 压缩发生在 `before_model` 同步关键路径上，专职快模型可将触发压缩那一轮的额外停顿从秒级压至亚秒 |
| **失败域隔离** | 摘要调用拥有独立限流与配额，不挤占主模型的 rate limit；且摘要失败本就有完整兜底（Plan A 降级提示词重试 → Plan B/D/C 结果级降级，§8.2），架构上即按"允许失败"设计 |

对上层宿主的影响：原先 `CompressionConfig(summary_model=agent_llm, ...)` 的写法，
改为构造摘要专职模型传入，例如
`CompressionConfig(summary_model=ChatOpenAI(model="gpt-4o-mini"), ...)`。

---

## 12 LangGraph 中间件集成

LangChain v1.x 提供了完善的中间件系统，这是本包的标准集成点 [11](#ref-11)。LangGraph 内置的 `trim_messages` 只支持基于 token 数量或消息数量的简单截断，不支持内容分类、结构化摘要、引用替代等高级策略。

### 12.1 中间件 Hook 利用

| Hook | 触发时机 | 在压缩管线中的角色 |
|---|---|---|
| `before_model` | 每次 LLM 调用前 | 触发判断 + L0-L1 轻压 |
| `after_model` | 每次 LLM 响应后 | L2-L4 对刚产生的 assistant 消息重压 |
| `wrap_tool_call` | 每次工具调用前后 | 源头压缩：工具执行前注入参数，返回后预处理 |
| `after_agent` | Agent 完整执行后 | 状态同步（由宿主通过 post_compress hook 实现） |

### 12.2 压缩模块核心结构

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import RemoveMessage, HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing import Any, NotRequired

class CompressionState(AgentState):
    compression_count: NotRequired[int]
    external_refs: NotRequired[dict]
    compression_history: NotRequired[list]

class CompressionMiddleware(AgentMiddleware):
    def __init__(self, config: CompressionConfig):
        self.config = config
        # 内部继承 SummarizationMiddleware 但通过 hook 暴露扩展点，
        # 不让宿主直接继承私有方法
        ...

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        token_count = self._count_tokens(messages)
        base_decision = self._evaluate_triggers(messages, token_count)
        # 通过 hook 让宿主覆盖触发判断
        if not self.config.should_summarize_hook(messages, token_count, base_decision):
            return None

        # 分割: 保留最近N条, 压缩其余
        to_keep = messages[-self.config.keep_recent:]
        to_compress = messages[:-self.config.keep_recent]

        # 检测已有摘要, 生成新综合摘要(非摘要套摘要)
        new_summary = self._generate_summary(to_compress)
        count = state.get("compression_count", 0)
        # 通过 hook 让宿主构造消息对象
        summary_msg = self.config.summary_message_builder(new_summary)

        result = {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *to_keep
            ],
            "compression_count": count + 1
        }
        # 通过 hook 让宿主做后处理（打标记、同步外部状态）
        return self.config.post_compress_hook(state, result)
```

### 12.3 源头压缩：wrap_tool_call

混合架构的核心——**中央编排器 + 源头预处理**。工具最了解自己返回的数据结构和可恢复性，因此 L2/L4 由源头处理；中间件拥有全局视野，负责 L0/L1/L3。

| 层级 | 执行位置 | 理由 |
|---|---|---|
| L0 内容过滤 | 中央 (before_model) | 需要全局视野判断哪些消息无效 |
| L1 Token 剪枝 | 中央 (before_model) | 需要完整消息序列计算困惑度 |
| L2 引用替代 | 源头 (wrap_tool_call) | 工具最了解返回的数据结构 |
| L3 语义摘要 | 中央 (after_model) | 需要 LLM 调用 + 全局上下文 |
| L4 外部化存储 | 源头 (wrap_tool_call) | 工具最了解数据的可恢复性 |

**分界线判断法**——一个压缩操作应该放在源头还是中央？用这三个问题判断：

1. 这个压缩需要知道"其他消息"的内容吗？→ 是 → 中央
2. 这个压缩需要知道数据的原始结构/可恢复性吗？→ 是 → 源头
3. 这个压缩需要 LLM 调用吗？→ 是 → 中央（除非工具自带小模型）

> **注意**：`wrap_tool_call` 的源头压缩由宿主项目在工具层实现（宿主最了解自己的工具返回结构），本包只提供 `wrap_tool_call` 中间件骨架和 `Externalizer` 抽象，不内置具体工具的压缩逻辑。

---

## 13 Message Reducer 与状态同步

### 13.1 不重构 reducer 的可行性

`add_messages` reducer 原生支持 `RemoveMessage`。**不需要重构 reducer 就能完成压缩。** LangGraph 论坛官方确认：`RemoveMessage` 被 reducer 解释后，目标消息从累积列表中移除，**reducer 的输出已经排除了被移除的消息** [12](#ref-12)。

### 13.2 不重构的代价

| 代价 | 说明 | 实际影响 |
|---|---|---|
| 存储增长 O(n×m) | 每个 checkpoint 存完整消息列表，旧 checkpoint 仍存旧列表 | 中等规模对话可接受，超长高频压缩时成为瓶颈 |
| 无法批量条件移除 | 只能逐条 RemoveMessage 或 REMOVE_ALL | 用 REMOVE_ALL_MESSAGES 可满足"替换历史"需求 |
| 快照时序延迟 | 压缩发生在 Run 中途，快照在 Run 结束后发送 | 功能无影响，但前端可能在 Run 结束后才看到消息变化 |

### 13.3 REMOVE_ALL_MESSAGES 策略

用 `REMOVE_ALL_MESSAGES` 一次性清空再追加压缩后消息，这是最干净的方案 [13](#ref-13)：

```python
return {"messages": [
    RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 清空全部
    summary_msg,                          # 新摘要在前
    *recent_messages                      # 原样保留最近消息
]}
```

> ⚠️ **风险点**
>
> **ID 不对齐**：摘要消息必须有稳定 ID（如 `compression_summary_{turn_number}`），否则 reducer 把同一条消息当成两条导致重复。本包通过 `summary_message_builder` hook 生成的消息由宿主保证 ID 稳定性。
>
> **状态同步协议**：压缩后状态如何同步到前端属于宿主与前端框架的契约（如 CopilotKit 的 `STATE_SNAPSHOT` / `MESSAGES_SNAPSHOT`、Vercel AI SDK 的消息协议等），**本包不处理前端同步**，只保证 LangGraph state 的正确性。宿主通过 `post_compress` hook 在压缩完成时自行触发前端同步。

### 13.4 已知限制

- **部分状态更新不合并**：如果宿主通过框架特定 API（如 `copilotkit_emit_state`）发送部分状态更新而非完整快照，可能不会被正确合并。建议宿主使用完整的状态快照或通过 `Command(update=...)` 走标准 reducer 路径。本包不对此做适配。

---

## 14 包结构与依赖

### 14.1 仓库形态

**独立 Git 仓库**，不作为任何项目的子目录。理由：

1. 逼出"独立项目"思维：独立 README、独立测试、独立 CI、独立 issue tracker
2. 开源时零代码迁移：`git remote add public && git push` 即可
3. 避免未来 import 路径迁移：从一开始宿主就 `from langcompress import ...`
4. editable install 与子包同等迭代速度：改包代码 → 重启宿主即生效

### 14.2 包结构

```
langcompress/                          # 独立 Git 仓库根
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── src/langcompress/
│   ├── __init__.py                    # 公开 API
│   ├── middleware.py                  # CompressionMiddleware
│   ├── config.py                      # CompressionConfig（含四个 hook 字段）
│   ├── state.py                       # CompressionState
│   ├── pipeline/
│   │   ├── base.py                    # CompressionStage 抽象基类
│   │   ├── l0_filter.py               # 内容过滤
│   │   ├── l1_prune.py                # Token 剪枝
│   │   ├── l2_reference.py            # 引用替代
│   │   ├── l3_summarize.py            # 语义摘要
│   │   └── l4_externalize.py          # 外部化存储
│   ├── summarizer/
│   │   ├── base.py                    # Summarizer 抽象基类
│   │   ├── llm_summarizer.py          # LLM 摘要器
│   │   ├── templates.py               # 八段式摘要模板
│   │   └── quality.py                 # 质量校验 + 降级
│   ├── externalizer/
│   │   ├── base.py                    # Externalizer 抽象基类
│   │   ├── filesystem.py              # 文件系统后端
│   │   ├── redis.py                   # Redis 后端
│   │   └── vectorstore.py             # 向量库后端
│   └── token_counter/
│       └── tiktoken_counter.py        # tiktoken 实现
├── tests/
│   ├── unit/                          # 单元测试
│   ├── integration/                   # 集成测试（含 LangGraph 真实运行）
│   └── scenarios/                     # 假想第二消费者场景测试
│       ├── test_pure_langgraph.py     # 纯 LangGraph（无 CopilotKit）
│       ├── test_langchain_only.py     # 纯 LangChain（无 LangGraph）
│       └── test_different_llms.py     # OpenAI / Claude / 本地模型
└── examples/
    ├── basic_usage.py                 # 最小示例
    ├── custom_hooks.py                # 自定义四个 hook
    ├── full_pipeline.py               # 五级全开
    └── source_compression.py          # 源头压缩（wrap_tool_call）
```

### 14.3 核心依赖最小化

核心包只依赖 `langchain-core`（提供 `AgentMiddleware`、`BaseMessage` 等基础类型），不依赖 `langgraph`。可选依赖通过 extras 分离：

| 安装方式 | 包含能力 |
|---|---|
| `pip install langcompress` | 中间件骨架 + 配置 + 四个 hook 接口 + 抽象基类 |
| `pip install langcompress[tiktoken]` | + 精确 token 计数 |
| `pip install langcompress[llm]` | + LLM 摘要生成（langchain-openai） |
| `pip install langcompress[redis]` | + Redis 外部化存储 |
| `pip install langcompress[vectorstore]` | + 向量库检索 |
| `pip install langcompress[all]` | 全量安装 |
| `pip install langcompress[dev]` | 开发依赖（pytest/ruff/mypy/langgraph 测试用） |

### 14.4 pyproject.toml 核心配置

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langcompress"
version = "0.1.0"
description = "Production-grade context compression middleware for LangGraph/LangChain agents"
requires-python = ">=3.10,<4.0.0"
license = {text = "MIT"}
keywords = ["langgraph", "langchain", "compression", "context", "middleware", "agent"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

# 核心依赖: 最小化, 只依赖 langchain-core
dependencies = [
    "langchain-core>=1.4.0,<2.0.0",
    "pydantic>=2.7.0,<3.0.0",
]

[project.optional-dependencies]
tiktoken = ["tiktoken>=0.7.0"]
llm = ["langchain-openai>=0.3.0,<1.0.0"]
redis = ["redis>=5.0.0"]
vectorstore = ["langchain-vectorstores>=0.3.0", "langchain-openai>=0.3.0,<1.0.0"]
all = ["tiktoken>=0.7.0", "langchain-openai>=0.3.0,<1.0.0", "redis>=5.0.0"]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "ruff>=0.6.0", "mypy>=1.10.0", "langgraph>=1.0.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/langcompress"]
```

### 14.5 延迟导入规范

可选依赖不在模块顶层 import，使用时动态导入，核心安装不需要装 tiktoken/redis/openai：

```python
# src/langcompress/token_counter/tiktoken_counter.py
def _get_tiktoken():
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "tiktoken is required for accurate token counting. "
            "Install with: pip install langcompress[tiktoken]"
        ) from e
    return tiktoken
```

### 14.6 宿主项目集成方式

#### 开发期（开源前 12 个月）

宿主项目通过 editable install 引用本地仓库：

```bash
# 宿主项目根目录
pip install -e ../langcompress[all]
```

或 `requirements.txt` 写本地路径：

```
langcompress[all] @ file:../langcompress
```

#### 开源后

```bash
pip install langcompress[all]
```

`requirements.txt` 改为一行，构建链路完全不变。

---

## 15 市面已有组件分析

在自建之前，全面调研市面已有的成熟压缩组件。

### 15.1 LangChain 官方内置

| 组件 | 压缩方式 | LLM 调用 | 可逆 | 内容分类 |
|---|---|---|---|---|
| `SummarizationMiddleware` | 阈值触发→摘要前缀→替换 | 是 | 否 | 无 |
| `trim_messages` | 按 token/条数截断 | 否 | 否 | 无 |

### 15.2 第三方 LangGraph 中间件包

| 组件 | 安装 | 核心机制 | 特点 |
|---|---|---|---|
| `langchain-collapse` | `pip install langchain-collapse` | 重复工具结果折叠为一行 | 零 LLM、无状态、92% 削减，与摘要组合延迟 4.2x [16](#ref-16) |
| `langmiddle` | `pip install langmiddle` | 摘要 + 语义记忆 + 工具清理 | 功能有重叠但压缩深度不足 [15](#ref-15) |

### 15.3 独立压缩库（语言无关）

| 组件 | 压缩率 | 可逆 | LangGraph 集成 | 核心特点 |
|---|---|---|---|---|
| `distil` | 83.2% | 是(字节精确) | pre_model_hook | SWE-bench 验证 42.0% vs 39.2% 不可区分，纯 Python 启发式比 LLMLingua 快 1000x [17](#ref-17) |
| `headroom-ai` | 60-95% | 是(CCR) | library/proxy/MCP | 内容路由+专用压缩器，25,785 stars，四种部署模式 [18](#ref-18) |
| `ContextZip` | 50-90% | 否 | Claude Code hooks | 零依赖，Protection Aura 保护逻辑词，Code Bypass [19](#ref-19) |
| `LLMLingua` | 20x | 否 | ContextualCompressionRetriever | 微软学术级，但 SWE-bench 仅 2.4% 成功率，有损风险高 [17](#ref-17) [3](#ref-3) |

### 15.4 能力覆盖对比

| 能力 | langcompress | distil | headroom | ContextZip | collapse | SummarizationMW |
|---|---|---|---|---|---|---|
| L0 内容过滤 | ✅ | 部分 | ✅ | ✅ | ✅ | ❌ |
| L1 Token 剪枝 | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| L2 引用替代 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| L3 语义摘要 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| L4 外部化存储 | ✅ | ❌ | 部分 | ❌ | ❌ | ❌ |
| 内容类型感知 | ✅ | ✅ | ✅ | ✅ | 部分 | ❌ |
| 可逆恢复 | 分层可恢复 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 源头压缩 | ✅ | 部分 | ✅ | ❌ | ❌ | ❌ |
| 摘要链管理 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 框架中立 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 可插拔管线 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 扩展点 hook | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

> **关键发现**：没有一个组件覆盖全部需求。最接近的是 distil（可逆+质量契约+LangGraph hook），但它缺 L3 摘要、L4 外部化、摘要链管理、扩展点 hook。langcompress 的差异化在于：**五级完整管线 + 四 hook 扩展点 + 框架中立**。

---

## 16 开源工程化约束

### 16.1 单一消费者陷阱防范

开源前本项目可能是唯一消费者。单一消费者时 API 会不知不觉向其需求收敛，导致开源后其他场景的用户发现 API 拐角处都是隐含假设。

**强制约束**：每设计一个 hook，必须通过以下三个"假想第二消费者"压测：

1. **纯 LangGraph（不用 CopilotKit）** 的项目能用吗？
2. **纯 LangChain（不用 LangGraph）** 的项目能用吗？
3. **非 OpenAI 模型（Claude / 本地模型 / GLM）** 能用吗？

如果某个 hook 让上述任一场景必须 fork 包，说明 hook 设计偏了。`tests/scenarios/` 目录下必须为每个场景维护端到端测试。

### 16.2 API 稳定性窗口

| 时段 | API 政策 |
|---|---|
| 0-9 月（开发期） | hook 签名可自由变更，只影响本项目一个消费者 |
| 9-12 月（冻结期） | hook 签名冻结，只修 bug 不加 breaking change |
| 12 月+（开源后） | breaking change 走 semver，major 版本升级 |

### 16.3 源码保护与混淆隔离

宿主项目（qianqian2.0）的 backend 采用 PyArmor 混淆构建。`langcompress` 作为开源包**不应被混淆**——混淆开源包无意义且破坏可调试性。

宿主项目 Dockerfile 约束：

- obfuscator 阶段：`pyarmor gen --recursive` 时排除 `langcompress`（通过 `--exclude langcompress` 或在 obfuscator 阶段不安装 langcompress）
- runner 阶段：以普通 pip 包安装 `langcompress`，保持明文 .py

### 16.4 Docker 构建层

宿主项目 docker 构建需将 `langcompress` 作为独立构建上下文：

- 开发期：`requirements.txt` 写 `langcompress[all] @ file:../langcompress`，或 docker-compose 用 volume 挂载源码 + 容器内 `pip install -e`
- 开源后：`requirements.txt` 写 `langcompress[all]>=0.1.0`，构建链路完全不变

### 16.5 开源前最小 checklist

- [ ] `langcompress` 核心零 CopilotKit 依赖、零 LangGraph 运行时依赖（仅 langchain-core）
- [ ] 核心依赖只 `langchain-core` + `pydantic`，LLM/tiktoken/redis/向量库全走 extras
- [ ] 四个 hook 接口签名稳定 3 个月未变
- [ ] 至少一个非 CopilotKit 场景的端到端测试通过（`tests/scenarios/`）
- [ ] 至少一个非本项目的真实消费者（内部项目或友好试用者）
- [ ] PyArmor 排除路径在宿主 Dockerfile 验证通过
- [ ] LICENSE (MIT) / README / CONTRIBUTING / CHANGELOG 完整
- [ ] CI 通过（pytest + ruff + mypy）
- [ ] examples 可独立运行

---

## 17 实施路线图

### 17.1 推荐策略：组合 + 自建

不从零自建全部五级，而是**编排层 + L3/L4/分类/扩展点自建，L0/L1/L2 委托给成熟组件**。自研量减少 50%，但差异化能力完整保留。

| 层级 | 实现方式 | 理由 |
|---|---|---|
| L0 内容过滤 | 用 `langchain-collapse` | 极专注此层，92% 削减，零 LLM [16](#ref-16) |
| L1 Token 剪枝 | 用 `ContextZip` 或自建 | 零依赖、语义保留、code bypass [19](#ref-19) |
| L2 引用替代 | 用 `distil` 的 reversible digest | SWE-bench 级质量验证，可逆恢复 [17](#ref-17) |
| L3 语义摘要 | 自建（八段式+摘要链管理） | 差异化核心，市面无成熟方案 |
| L4 外部化存储 | 自建 | 需要与特定文件系统/向量库集成 |
| 内容分类路由 | 自建（机制）+ 宿主 hook（业务规则） | 本包提供路由框架，业务场景由宿主注入 |
| 扩展点 hook | 自建 | 差异化核心，市面无人做此层 |
| 编排层 | 自建 middleware | 统一编排以上各层 |

### 17.2 MVP 范围（v0.1.0）

| 包含 | 不包含（后续版本） |
|---|---|
| CompressionMiddleware（before_model + wrap_tool_call 骨架） | L1 token 剪枝（需要小模型） |
| 四个 hook 接口（含默认实现） | L2 引用替代（需要外部存储协议） |
| L0 内容过滤 | 向量库外部化 |
| L3 语义摘要 + 八段模板 | Redis 后端 |
| L4 文件系统外部化 | 自定义摘要模板引擎 |
| tiktoken token 计数 | 摘要质量校验 |
| REMOVE_ALL_MESSAGES 策略 | 降级机制 |
| 摘要链管理 | |
| 延迟导入 + optional extras | |
| 假想第二消费者场景测试 | |

### 17.3 12 个月实施计划

| 时段 | 目标 | 完成标志 |
|---|---|---|
| **0-3 月** | 独立仓库搭建 + L3 摘要从宿主项目迁入 + 四个 hook 接口定稿 + 宿主项目切换为消费者 | 宿主用 `pip install -e langcompress` 跑通，功能与当前完全等价 |
| **3-6 月** | 补 L0 内容过滤、L2 引用替代（用 distil/collapse 委托） + 编写"假想第二消费者"测试 | 至少有一个非 CopilotKit 场景的单元测试通过 |
| **6-9 月** | 补 L4 文件系统外部化 + 摘要链管理（不生成"摘要的摘要"）+ 质量校验 | 五级管线 L0/L2/L3/L4 落地（L1 token 剪枝可后置） |
| **9-12 月** | API 冻结期 + 文档 + examples + 找 1-2 个真实外部试用者 | 至少一个非本项目的真实消费者跑通 |
| **12 月+** | 开源：换公共 PyPI、写 README/CONTRIBUTING/LICENSE、宣传 | 通过开源前 checklist |

### 17.4 关键里程碑

1. **M1（第 3 月末）— 独立仓库可用**：宿主项目完全切换为 langcompress 消费者，自身不再有压缩逻辑代码（仅保留 CopilotKit 适配层通过 hook 注册）。这是整个方案的奠基里程碑——验证四 hook 接口足以承载宿主全部适配需求。
2. **M2（第 6 月末）— 多场景验证**：`tests/scenarios/` 下三个假想消费者场景测试全部通过，证明 API 未偏科。
3. **M3（第 9 月末）— 五级管线完整**：L0/L2/L3/L4 落地，L1 可委托或后置。
4. **M4（第 12 月末）— 开源就绪**：通过开源前 checklist，至少一个外部真实消费者。

### 17.5 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 四 hook 接口不够，宿主适配需求溢出 | 中 | M1 前用宿主现有 170 行适配代码做完整压测，若发现不够立即扩展为五 hook 或六 hook |
| 单消费者导致 API 偏科 | 高 | 强制 `tests/scenarios/` 三场景测试，每加一个 hook 必须过三场景 |
| LangChain 父类私有方法签名变更 | 低 | 通过 hook 翻译层隔离，父类升级只影响本包内部，不影响宿主 |
| L1 token 剪枝依赖小模型过重 | 中 | MVP 不含 L1，委托 ContextZip，保持核心包零小模型依赖 |
| 开源后无人使用 | 中 | M4 前找 1-2 个真实外部试用者验证需求 |

---

## 18 外部化内容生命周期管理

### 18.1 问题

L2/L4 把原始内容卸载到外部存储后，**只增不减**：`externalize` 只写、`retrieve` 只读，没有任何 TTL、引用计数或淘汰机制。长期运行的 agent 会在 `.langcompress_cache/`（或任何后端）无限累积文件。

但清理不能"定期一删了之"——它和 §3.3 有损可恢复契约天然冲突：

- **删早了**：ref 还在 `state["external_refs"]` 里、摘要消息还引用着它，`retrieve` 却失败 = 静默破坏契约；
- **不删**：磁盘无限增长（现状）。

且**包无法单方面判定 ref 是否已死**——只有宿主知道"这段对话已终结、没人会再召回"。这决定了体系的根本分工（延续 §12.3 的边界哲学）：**机制在包，策略时机在宿主**。

### 18.2 设计不变量

任何实现不得违反以下五条：

1. **可达 ⇒ 可恢复**：凡在宿主提供的可达性根集（root set）里的 ref，永不被物理删除；已进入宽限期的会被自动恢复。
2. **删除分两阶段**：物理删除前必经"软删除 + 宽限期"（Stale 态），窗口内可逆（restore）。
3. **默认零行为变化**：`LANGCOMPRESS_RETENTION_ENABLED` 默认 `false`；不启用时一切与现状逐字节一致。
4. **后端可选实现**：`Externalizer` ABC 的新方法带 no-op 默认实现，第三方后端（Redis/S3）按能力实现，不实现则生命周期管理对它静默降级。
5. **全程可观测**：每次清理产出 `PurgeReport` + INFO 日志（对齐 v0.4 观测面），失败的清理绝不中断 agent。

### 18.3 生命周期状态机

```
                 externalize()
                      │
                      ▼
                ┌──────────┐   ref 进入根集（state["external_refs"] 或活消息）
                │  Active  │ ◄────────────────────────────┐
                └────┬─────┘                              │
                     │ 判定不可达（根集外）                 │
                     │   且 policy.should_evict == True   │
                     ▼                                    │
                ┌──────────┐                              │
                │  Stale   │ ──── restore() ──────────────┘
                │ （宽限期）│      （宽限期内可逆，防误删）
                └────┬─────┘
                     │ 宽限期（grace period）到期
                     ▼
                ┌──────────┐
                │  Purged  │  物理删除，不可逆
                └──────────┘
```

两阶段删除是安全核心：Stale 态的内容**仍可 retrieve**（只是被标记为待删），只有宽限期（默认 24h）过后才物理删除。误判（ref 被错误判定死亡，随后又被召回）在窗口内可自愈。

### 18.4 可达性模型（root set）

一个 ref 是**可达的**，当且仅当它出现在以下任一处：

1. `state["external_refs"]` 的键（v0.4 的 dict-merge reducer 保证跨压缩累积——它同时是观测面和 GC 根集来源）；
2. 任意活消息的 `additional_kwargs["external_ref"]`（`aggregate_external_refs` 已实现的扫描）；
3. 宿主显式 pin 的集合（如归档对话想永久保留的资源）。

包提供工具函数 `collect_live_refs(state, messages, extra) -> set[str]` 聚合三者；**何时调用、以什么频率调用由宿主决定**（对话终结钩子、每日 cron、§5.1 式轮次阈值均可）。

### 18.5 组件设计

#### 18.5.1 `Externalizer` ABC 增量扩展（向后兼容）

```python
class Externalizer(ABC):
    # 现有（不变）
    def externalize(self, blob, *, key=None) -> str: ...
    def retrieve(self, ref) -> str: ...

    # 新增（全部带默认实现，现有子类零改动）
    def list_refs(self) -> list[ExternalRefRecord]: ...   # 默认 []
    def purge(self, keep_refs: set[str], *,
              evict_refs: set[str],      # Active → Stale（软删除）
              purge_refs: set[str],      # Stale → 物理删除（不可逆）
              restore_refs: set[str]) -> PurgeReport: ...  # 默认空报告
    def restore(self, ref: str) -> bool: ...              # 默认 False（手动单条恢复）
    # 异步版本默认 asyncio.to_thread 委派（对齐 v0.4）
```

状态机决策集中在 `RetentionManager` 一处（三集合怎么算出来），后端只执行批量迁移——这是"机制在包、后端只做存储操作"的落点。**ref 字符串在全生命周期稳定不变**：软删除只改物理位置（移入 `.trash/`），上下文里已写死的 ref 不需要任何更新。

`ExternalRefRecord`（frozen dataclass）：`ref / created_at / size / state`（active|stale）。
`PurgeReport`（frozen dataclass）：`staled / purged / restored / kept / errors` 五个列表——完整的清理审计面。

#### 18.5.2 `RetentionPolicy` ABC（策略对象）

```python
class RetentionPolicy(ABC):
    def should_evict(self, record: ExternalRefRecord) -> bool: ...
    @property
    def grace_period(self) -> timedelta: ...

class NullPolicy(RetentionPolicy): ...    # 永不清理（默认 = 现状）
class TTLPolicy(RetentionPolicy): ...      # 超龄淘汰（created_at 或 last_accessed 超过 TTL）
class MaxSizePolicy(RetentionPolicy): ...  # 存储总量超限时驱逐最老（v1.1）
```

策略是纯函数式判定（输入 record，输出 bool），可组合；`grace_period` 归策略管——不同策略可以有不同宽限强度。

#### 18.5.3 `RetentionManager`（编排器，纯核心包）

```python
class RetentionManager:
    def __init__(self, externalizer: Externalizer, policy: RetentionPolicy): ...

    def run(self, keep_refs: set[str]) -> PurgeReport:
        # 1. list_refs() 枚举全部记录
        # 2. 在根集 → keep；若处于 Stale 态则先 restore（不变量 1）
        # 3. 不在根集 且 policy.should_evict → 软删除（进 Stale）
        # 4. Stale 超过 grace_period → 物理删除
        # 每步 try/except，错误进 report.errors，绝不抛出（不变量 5）
```

#### 18.5.4 `FilesystemExternalizer` 两阶段实现

- Active 文件：`base_dir/<key>.md`（现状不变）；
- 软删除：`os.rename` 移入 `base_dir/.trash/<key>.md`——同目录 rename，零拷贝、跨平台原子操作；
- `retrieve` 对 `.trash/` 里的 ref 仍可读（Stale 可恢复）；物理删除即 unlink；
- 元数据零数据库：文件 `mtime` 即 `created_at`、文件大小即 `size`（对齐"生产环境不运行时建表"的工程约束）。

### 18.6 配置（环境变量只提供标量，开关决策在宿主代码）

```
LANGCOMPRESS_RETENTION_TTL_HOURS=        # TTLPolicy.from_env() 读；未设置 → None（= 不启用）
LANGCOMPRESS_RETENTION_GRACE_HOURS=24    # 宽限期，from_env 的默认
LANGCOMPRESS_EXTERNALIZER_DIR=           # （已有）存储目录
```

关键决策（v0.5 实现定稿）：**环境变量永远不会单独开启清理**。`TTLPolicy.from_env()` 在 TTL 未设置时返回 `None`，宿主显式 fallback 到 `NullPolicy`：

```python
policy = TTLPolicy.from_env() or NullPolicy()   # 未配置 → 永不清理（= 现状）
manager = RetentionManager(externalizer, policy)
```

即"是否启用 retention"是宿主代码里的显式判断，环境变量只调标量参数——延续 v0.3 的约定（标量走环境变量，类型/可调用走构造参数），并把不变量 3 落到实处：包里不存在任何能隐式触发删除的环境变量。

### 18.7 触发时机（宿主掌握）

包不内置调度。宿主的三种典型接入：

```python
# A. 对话终结钩子（最精准）
manager = RetentionManager(externalizer, TTLPolicy(ttl=timedelta(days=7)))
manager.run(keep_refs=collect_live_refs(state, messages))

# B. post_compress_hook 内嵌（对齐 §5.1 轮次阈值思想，每 N 次压缩跑一次）
def post_compress(state, result):
    refs = aggregate_external_refs(result)
    if state.get("compression_count", 0) % 10 == 0:      # 每 10 次压缩
        manager.run(keep_refs=set(state["external_refs"]))
    return {**result, "external_refs": refs}

# C. 独立 cron / 运维脚本（与 agent 进程解耦）
```

### 18.8 分期

- **v0.5.0**：ABC 扩展（含 no-op 默认）+ `FilesystemExternalizer` 两阶段删除 + `NullPolicy`/`TTLPolicy` + `RetentionManager` + `collect_live_refs` 工具 + 测试（状态机各迁移、根集保护、宽限期到期、误删自愈、默认关闭零行为变化）。
- **v1.1**：`MaxSizePolicy`、`CompositePolicy`、post_compress 内嵌触发的官方配方、Redis/S3 后端的 purge/restore 实现。

---

## 参考来源

<a id="ref-1"></a>[1] Juejin, *Manus vs Claude Code vs Gemini CLI 上下文压缩深度对比*. 三种流派的策略差异和核心哲学. https://juejin.cn/post/7546507803332411443

<a id="ref-2"></a>[2] Anthropic, *Effective Context Engineering for AI Agents*. 工具设计哲学和最小但充分原则. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

<a id="ref-3"></a>[3] Microsoft, *LLMLingua Series Documentation*. Token 级困惑度剪枝原理与 LangChain/LlamaIndex 集成. https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md

<a id="ref-4"></a>[4] Claude Platform, *Tool Use Automatic Context Compaction*. Claude SDK 的 compaction_control 和自定义 summary_prompt. https://platform.claude.com/cookbook/tool-use-automatic-context-compaction

<a id="ref-5"></a>[5] Anthropic, *Effective Context Engineering for AI Agents*. 结构化笔记策略和上下文工程最佳实践. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

<a id="ref-6"></a>[6] arxiv.org, *Context Window Management for LLM Agents*. 预测触发和内容超限即触发策略. https://www.arxiv.org/pdf/2510.00615

<a id="ref-7"></a>[7] ReinforcementCoding, *Context Compression: Efficient Data Formats*. 字段过滤、JSON→Markdown、列式 JSON 格式优化. https://www.reinforcementcoding.com/blog/context-compression-efficient-data-formats

<a id="ref-8"></a>[8] arxiv.org, *TableZoomer*. Schema 替代全表与查询感知列裁剪框架. https://arxiv.org/abs/2509.01312

<a id="ref-9"></a>[9] PeerJ, *CoAgt*. 多 Agent 分工处理结构化数据。Collector + Synthesizer 架构. https://peerj.com/articles/cs-3423/

<a id="ref-10"></a>[10] Claude Platform, *Context Editing*. 压缩后上下文重构的官方文档. https://platform.claude.com/docs/en/build-with-claude/context-editing

<a id="ref-11"></a>[11] LangChain Docs, *Custom Middleware*. AgentMiddleware 的 6 个 hook 点和自定义中间件开发指南. https://docs.langchain.com/oss/python/langchain/middleware/custom.md

<a id="ref-12"></a>[12] LangChain Forum, *DeltaChannel + RemoveMessage + Message Compression*. add_messages reducer 和 RemoveMessage 的精确行为. https://forum.langchain.com/t/deltachannel-removemessage-message-compression-how-to-express-deletions-with-incremental-storage/4156

<a id="ref-13"></a>[13] LangChain Docs CN, *Short-term Memory*. REMOVE_ALL_MESSAGES 和 RemoveMessage 的使用方式. https://langchain-doc.cn/v1/python/langchain/short-term-memory.html

<a id="ref-14"></a>[14] CopilotKit Skills, *AG-UI Protocol Spec*. MESSAGES_SNAPSHOT 和 STATE_SNAPSHOT 事件协议规范（参考，本包不直接依赖）. https://github.com/copilotkit/skills/blob/main/skills/copilotkit-agui/references/protocol-spec.md

<a id="ref-15"></a>[15] GitHub, *LangMiddle — Production Middleware for LangGraph*. 独立中间件包的 ChatSaver/ContextEngineer/ToolRemover. https://github.com/alpha-x-one/langmiddle

<a id="ref-16"></a>[16] GitHub, *langchain-collapse — Preventive Context Management*. CollapseMiddleware 重复工具结果折叠，92% 削减. https://github.com/johanity/langchain-collapse

<a id="ref-17"></a>[17] GitHub, *distil — Compression with a Quality Contract*. SWE-bench 验证可逆压缩，42.0% vs 39.2% 不可区分. https://github.com/dshakes/distil

<a id="ref-18"></a>[18] EveryDev, *Headroom — Context Compression Layer*. 60-95% token reduction，内容路由+专用压缩器，四种部署模式. https://www.everydev.ai/tools/headroom

<a id="ref-19"></a>[19] GitHub, *ContextZip — Semantic Context Compression*. 零依赖，50-90% 压缩，Protection Aura + Code Bypass. https://github.com/luislozanogmia/contextzip
