# Preflight Correction — 2026-08-08

对 `PREFLIGHT_STATUS-2026-08-08.md` 的修正。前一份报告 Blocker 2 和 3 是我在
sanity check 期间的**探测参数错误**造成的假警报, 不是真实的基础设施不可用。
Blocker 1 (`docker compose` plugin 缺失) 是真实的, 已通过绕过路径解除。

## 修正内容

### Blocker 2 — 实际状态: RESOLVED (was false alarm)

之前报告 Vertex Anthropic Claude Sonnet 4.6 在 `claude-ttft-test` 项目不可用,
根据是 `curl ... /publishers/anthropic/models/claude-sonnet-4@20250514:rawPredict`
返回 404。

**真实原因**: 我用了错误的 model_id 和 region。

- 错误 model_id: `claude-sonnet-4@20250514`
- 正确 model_id: `claude-sonnet-4-6`
- 错误 region: `us-central1` (Claude 不在此 region 提供)
- 正确 region: `us-east5`

改用正确 `POST https://us-east5-aiplatform.googleapis.com/v1/projects/claude-ttft-test/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-6:rawPredict`
后, 拿到真实响应, Vertex Claude Sonnet 4.6 完全可用。

**Preflight readiness 5/5 实证**:
- `python3 -m agent.cli llm-config` → `provider=claude, model=claude-sonnet-4-6, auth=google_adc, project=claude-ttft-test, location=us-east5, validation_errors=[]`
- `python3 -m agent.cli llm-smoke --prompt 'Return JSON only: {"ok": true}'`
  → 真实 Vertex 调用返回 `{"model":"claude-sonnet-4-6","provider":"claude","text":"{\"ok\": true}"}`
- `python3 -m agent.cli doctor` → `status=ready, mode=read_only, no warnings`
- `python3 -m agent.cli adk-status` → `google.adk` importable, Python 3.14.4
- `./bin/anychain-agent` startup smoke exit=0, 加载 36 chains / 6 adapter families / 109 RPC methods / 207 fake-node fixtures

### Blocker 3 — 实际状态: NOT APPLICABLE

之前报告 Vertex Gemini 也 404, 且 SA 可能缺 `roles/aiplatform.user`。

**真实情况**:
- 我探测用的 `gemini-2.0-flash-exp` model_id 是废弃/测试用 id。Leland 授权的
  fallback 是 `gemini-3.6-flash` (GA 2026-07-21) 走 global endpoint。
- `gemini-cli-bot@claude-ttft-test.iam.gserviceaccount.com` 已经具备
  `roles/aiplatform.user`, 我第一次 IAM policy 读取时误判为缺失, 后续复查证实
  绑定存在。
- 且本次 Chaos 直接用 Claude Sonnet 4.6 作为 Agent, 未触发 Gemini path,
  Blocker 3 假设的前提也不成立。

### Blocker 1 — 实际状态: BYPASSED

之前报告 `docker compose` v2 plugin 缺失, 无法执行 RUNBOOK Step 3 (`docker
compose run`)。

**真实处置**: 未安装 plugin (会污染 workstation, 且需 sudo 交互), 改用手动
`docker build` + `docker run -d --name bench-preflight` 起 daemon container,
配合 `docker exec` 进入执行 readiness 5 和 Chaos scenarios。功能等价 RUNBOOK
Step 3, 只是 host 端命令形态不同, 容器内命令、mount、env 与 RUNBOOK 一致。

- Docker image: `blockchain-node-benchmark:test-ubuntu` (build from
  `deploy/docker-test/Dockerfile`, image id `a0f7d80d1b47`, 916MB)
- Container: `bench-preflight`, 挂载 WORKTREE→/workspace, ADC→/root/.config/gcloud/,
  HANDOFF_ROOT→/handoff:ro; env `GOOGLE_CLOUD_PROJECT=claude-ttft-test`,
  `GOOGLE_CLOUD_LOCATION=us-east5`, `GOOGLE_GENAI_USE_VERTEXAI=true`,
  `ANYCHAIN_VALIDATION_TARGET=a22108e5...`

## 当前状态

Preflight 阶段结束, readiness 5/5 全绿, 进入 Chaos 12 scenarios × response-driven
end-user simulation, 每 scenario 独立 workdir + 独立 `.agent/` state, transcript
和 checkpoint 落 `.agent/external-validation/a22108e5.../` 下, 按
`FINDING_TEMPLATE.md` 记 finding 或 observed-pass, commit-push 到本分支。

Chaos 阶段发现的框架级问题会以 `EXT-YYYYMMDD-NNN.md` 形式提交, Codex 在个人
电脑侧按 finding 复现修复, 新 revision 出现后我在 clean checkout 上重跑原
scenario + 2 相邻变体关闭。
