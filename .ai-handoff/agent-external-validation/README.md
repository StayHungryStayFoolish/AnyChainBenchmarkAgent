# AnyChain Agent 跨电脑验证交接

状态：跨 AI、跨电脑的开发验证资料。它位于专用 `.ai-handoff/` 路径，不属于
产品文档，也不得被 AnyChain Agent 作为运行时知识读取。

协调分支：`docs/agent-handoff-chaos`

目标产品 revision：`a22108e53401cf7d9314917b870b71ccf338619c`

运行目标：仅 Linux/Docker。

## 工作电脑读取顺序

只读取以下 4 份交接文件：

1. `README.md`
2. `VALIDATION_MANIFEST.yaml`
3. `ADC_GEMINI_WORKSTATION_RUNBOOK.md`
4. `FINDING_TEMPLATE.md`

不要依据日期型 task plan、旧 transcript、旧 finding 或历史 `.agent/` 输出扩大
任务。需要确认产品预期行为时，只读取目标 revision 中的长期权威文档：

- `docs/zh/agent-handoff-product-verification.md`
- `docs/zh/agent-cli-verification-guide.md`
- `agent/README.md`

## 双 AI 角色与证据

- AnyChain Agent：通过 Vertex AI ADC 使用 Gemini，产生真实 Agent 回复和工具调用。
- Claude：扮演 response-driven 用户。每一轮必须完整读取 Gemini Agent 的实际回复，
  再根据该回复决定下一次用户输入；禁止照着预生成 transcript 播放。
- Claude 同时负责记录发现，但不能修改 tracked 产品代码、测试、verifier、期望输出
  或运行时 checkpoint。
- Gemini 不是独立的“问题记录员”。它在 AnyChain Agent 内产生的原始回复、工具
  receipt、checkpoint、job 和日志必须原样进入证据包；不得只保存 Claude 的总结。
- Claude discovery 不能标记为 `Codex as user`，也不能伪造正式 Codex attestation。
- 当前个人电脑上的 Codex：验证证据包，在原 revision 复现问题，确定 root owner，
  实施框架级确定性修复并生成新 revision。
- 工作电脑：clean checkout 新 revision，重跑原失败路径和至少两个相邻变体；全部
  通过后才可关闭 finding。

## 修复闭环

工作电脑发现的问题必须记录到 `FINDING_TEMPLATE.md`，至少包含：

1. 完整 source revision、模型、ADC 项目/区域和 Linux/Docker 环境；
2. 失败前完整 Agent 回复、用户原始输入、失败后完整 Agent 回复；
3. checkpoint、runtime event、tool receipt、job、日志及其 SHA-256；
4. 期望契约、实际行为、首次失败 turn、是否能在 fresh session 复现；
5. 脱敏后的 evidence bundle 及 SHA-256。

Codex 不直接采信 Claude 或 Gemini 对根因的猜测。Codex 必须先校验证据，再在同一
revision 复现，然后将修复落在唯一权威 owner 上。禁止关键字补丁、transcript 特判、
terminal 文案拦截、放宽 verifier 或恢复旧 checkpoint 字段来制造通过结果。

## 当前基线

- Agent 静态测试：`1553/1553`，另有 `620` 个子测试。
- 全仓库：`2336/2336`；该数字包含 Agent 测试，不能相加。
- Linux required shell gates：`50/50`。
- Product Chaos catalog：`510` 个可达 obligation；只代表目录已生成，不代表已经
  全部实时执行。
- 当前 revision 正式 Case 2：`0/1` qualifying pass。第一次运行在 turn 0 前因
  DeepSeek readiness HTTP 402 结束，属于 infrastructure interruption。
- Gemini/ADC、真实 google_search 和工作电脑 Claude discovery 尚待执行。

## 安全边界

完整 transcript、checkpoint、runtime event、job、日志、CSV 和 HTML 只放在被测
worktree 的 `.agent/external-validation/<revision>/`，不进入 Git。

任何 API key、ADC JSON、OAuth refresh token、带凭据 endpoint、客户数据或未脱敏
payload 都禁止进入 finding、transcript、Git 和传输包。
