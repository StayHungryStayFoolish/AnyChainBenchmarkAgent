# AnyChain Agent Google Search Task Plan

本文档约束本次 `google_search` 接入任务，避免再次偏离 AnyChain Agent 主线。
本文件是开发任务文档，不是用户快速入门。

## 执行约束

- 本次任务必须遵循本地 `CLAUDE.md`：先读现有代码，定义假设和验收标准，保持最小改动，验证后再收口。
- 不允许把业务意图识别移动到 terminal、runner bridge、shell wrapper 或正则/关键词列表。
- 不允许通过用户话术关键词触发 unknown chain、custom RPC 或 benchmark onboarding。
- ADK 和配置模型负责理解用户意图、选择 sub-agent、决定是否调用工具。
- Deterministic tools 只负责事实读取、配置校验、计划生成、执行 gate 和产物分析。
- 搜索结果只能作为 evidence，不能作为“已支持”的承诺。

## 目标

当用户请求测试当前 36 条链之外的新链、增加自定义 RPC method，或要求评估一个属于现有 6 family 但尚未配置的链时，AnyChain Agent 可以在 Gemini 模式下使用 ADK `google_search` 辅助调研官方资料。

## 非目标

- 不为 DeepSeek、OpenAI、Claude API Key、Claude on Vertex 默认启用 ADK `google_search`。
- 不开发 provider-neutral 搜索 API。
- 不在本地无 Gemini/ADC 环境时声称 live Google Search 已经通过。
- 不让搜索结果绕过 endpoint、request/response sample、fixture recording、template validation 或 fake-node smoke。

## 启用条件

只有同时满足以下条件，才启用 ADK `google_search`：

- `LLM_PROVIDER=gemini`
- `LLM_MODEL` 是 Gemini family
- `LLM_AUTH_MODE` 和 Gemini/Google 认证配置通过本地校验
- 当前 ADK runtime 可以 import `google.adk.tools.google_search`

Claude 即使使用 Google Cloud/Vertex/Agent Platform 认证，也不启用 ADK `google_search`。

## 业务流程边界

`google_search` 只能挂载在 Chain/RPC Onboarding Agent：

1. 先读取本地 framework capabilities / 36 chain template / RPC method inventory。
2. 如果本地不支持，进入 onboarding。
3. 如果 Gemini `google_search` 可用，由 ADK onboarding agent 自己判断是否搜索。
4. 搜索优先级：
   - 官方 RPC 文档
   - 官方节点运维文档
   - 官方 GitHub / API examples
   - 高质量社区资料仅作为辅助线索，不能覆盖官方文档
5. 输出 onboarding handoff：
   - evidence URL / source summary
   - adapter family hypothesis 和 confidence
   - 需要用户提供的 endpoint / request sample / response sample
   - chain template 草案要求
   - fixture 录制计划
   - validation commands
   - fake-node smoke 要求

## 验收标准

本地可验收：

- 非 Gemini provider 启动时展示 `Web research: unavailable for current provider`。
- Gemini 配置不完整时不启用搜索，并展示原因。
- Gemini 配置完整但本机 ADK 无 `google_search` import 时不启用搜索，并展示原因。
- Chain/RPC Onboarding Agent 只有在启用条件满足时才挂载 `google_search`。
- DeepSeek live matrix、单测、boundary check 不回归。
- 代码中不新增业务关键词/模糊匹配意图路由。

工作电脑可验收：

- Gemini/ADC 或 Gemini API Key 下启动展示 `Web research: enabled via ADK google_search`。
- unknown chain / custom RPC 对话中，ADK onboarding agent 可调用 `google_search` 获取官方 evidence。
- evidence 进入 onboarding plan，但不会跳过人工确认、fixture、validation、smoke。
