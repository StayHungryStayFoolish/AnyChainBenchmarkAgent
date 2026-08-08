# AnyChain Agent External Validation Record

每个问题使用一份；没有发现问题但形成 observed pass 的 scenario 也使用精简副本。
工作电脑只记录事实，不修改代码。

## 身份

- Record ID: `EXT-YYYYMMDD-NNN | PASS-<scenario-id>`
- Finding present: `yes | no`
- Status: `observed_failure | observed_pass | reproduced | repaired | work-verified | closed`
- Acceptance class: `discovery | formal`
- Source branch:
- Full source revision:
- Worktree hash:
- Tracked worktree clean: `yes | no`
- Runtime: `Linux distribution + Docker image/container ID`
- Host: `OS/architecture，仅供参考`

## AI 与认证

- Agent provider/model/auth:
- External user AI/model:
- External user role: `response_driven_discovery | codex_formal`
- ADC project/location，不包含凭据:
- `llm-config` evidence:
- `doctor` evidence:
- `llm-smoke` evidence:
- `adk-status` evidence:
- 实际 google_search typed receipt/citations:

不得把 Claude 标记为 Codex。不得包含 API key、ADC JSON、OAuth refresh token、
带凭据 endpoint、客户 payload 或未脱敏 secret。

## 起始状态

- Manifest scenario ID:
- Journey/task ID:
- Session/checkpoint ID:
- Starting groups and states:
- Pending question ID/owner:
- Chain case: `supported | case1 | case2 | case3 | none`
- Target/workflow/RPC/QPS modes:
- Prior job ID/status:

## 精确观察

- Previous complete Gemini Agent response path + SHA-256:
- Exact Claude user input path + SHA-256:
- Next complete Gemini Agent response path + SHA-256:
- Tool receipt path + SHA-256:
- Runtime event/checkpoint path + SHA-256:
- Job/log/CSV/HTML path + SHA-256:
- First failing turn index:
- Expected contract/postcondition:
- Observed behavior:
- User impact:
- Reproduction count:
- Fresh isolated session reproduced: `yes | no`

不能省略决定用户下一步行为的 Agent 回复。引用 transcript 必须保留原始语言、多行
边界和 turn 顺序。Claude 的总结不能替代 Gemini Agent 原始输出。

## 分类

- Classification: `none | product_failure | verifier_failure | provider_blocker | auth_blocker | infrastructure_interrupted | test_harness_failure`
- Suspected root owner: `unknown | terminal | coordinator | semantic_planning | admission | domain:<name> | response_authority | checkpoint | runner | verifier | documentation`
- Supporting evidence:
- Evidence against stale session/data contamination:
- Not tested because:

工作电脑可将 root owner 写为 `unknown`。不得修改源码、测试、verifier、期望输出、
checkpoint 或 transcript 来制造通过结果。

## 安全传输

- Redaction review completed by:
- Bundle path:
- Bundle SHA-256:
- Internal `SHA256SUMS` verified on personal workstation: `yes | no`
- Missing evidence:

## 个人电脑 Codex 确定性修复

- Evidence bundle integrity verified: `yes | no`
- Reproduced on exact source revision: `yes | no`
- Root cause:
- Violated architecture authority:
- Single authoritative repair owner:
- Framework-level repair design:
- Files changed:
- Repair revision:
- Focused deterministic tests:
- Adjacent regression tests:
- Complete Docker/Linux gates:
- Why this is not a phrase/keyword/transcript/terminal/verifier patch:

Codex 必须用原始证据和可重复状态转移定位根因；不能只依据 Claude 或 Gemini 的
自然语言诊断修改代码。

## 工作电脑回验

- Clean checkout of repair revision: `yes | no`
- Original failed journey result:
- Adjacent variant 1:
- Adjacent variant 2:
- Gemini/ADC/google_search result:
- New evidence bundle + SHA-256:
- Remaining gap:
- Closure decision and reason:

只有原失败 journey 和两个相邻变体在 repair revision 上都通过，finding 才能标记
`closed`。provider/auth/infrastructure blocker 不能转写为产品通过。
