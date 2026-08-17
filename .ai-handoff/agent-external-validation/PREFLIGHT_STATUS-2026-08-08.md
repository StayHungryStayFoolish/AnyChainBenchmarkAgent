# Workstation Preflight Status — 2026-08-08

工作电脑在启动 dual-AI Chaos 测试**之前**的 sanity check 报告。
本次未进入被测 revision, 未产生任何 CLI journey, 未生成 product finding。

按 `.ai-handoff/agent-external-validation/README.md` 和
`ADC_GEMINI_WORKSTATION_RUNBOOK.md` 的定义, 本状态归类为
`infrastructure_interrupted` + `auth_blocker`, **不能**转写成产品失败
或产品通过。

## 身份与环境

- Record ID: `PREFLIGHT-20260808-001`
- Finding present: `no`
- Status: `preflight_blocked`
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Coordinator HEAD: `8123d3f8d3eadc31e6bfbbdc071a22f4179c2bd6`
- Target revision (未进入): `a22108e53401cf7d9314917b870b71ccf338619c`
- Worktree hash: N/A (未创建 detached worktree)
- Tracked worktree clean: N/A
- Runtime: Linux `6.12.90+deb13.1-cloud-amd64`, GCE VM
- Host: GCE, project `claude-ttft-test`
- Wall clock: 2026-08-08 22:00 – 22:10 HKT (Asia/Hong_Kong)

## AI 与认证 (计划 vs 实际)

- **计划 Agent provider/model**: Claude Sonnet 4.6 via Vertex AI (Anthropic partner) — 由 Leland 指定
- **计划 End user AI/model**: Claude Sonnet 4.6, 32 concurrent terminals
- **计划 auth mode**: Vertex ADC on `claude-ttft-test`, no API key
- **实际 ADC principal**: `gemini-cli-bot@claude-ttft-test.iam.gserviceaccount.com` (service account, 非 user credential)
- **`llm-config` evidence**: not executed (未进入 detached worktree)
- **`doctor` evidence**: not executed
- **`llm-smoke` evidence**: not executed
- **`adk-status` evidence**: not executed
- **google_search typed receipt**: not executed

## Blocker 详情 (按发现顺序)

### Blocker 1 · `docker compose` 子命令缺失

Evidence:
```
$ docker --version
Docker version 26.1.5+dfsg1, build a72d7cd
$ docker compose version
docker: 'compose' is not a docker command.
$ which docker-compose
(empty)
```

Target revision 的 `docker-compose.yml` 明确使用 `docker compose build/up/exec` 语法
(v2 plugin), 老 `docker-compose` v1 binary 也未安装。

结论: 无法在容器里跑 `bench` 服务, 无法执行 RUNBOOK Step 3。

Required action (需 Leland / sudo 权限):
```
sudo apt-get install -y docker-compose-plugin
```

### Blocker 2 · Vertex Anthropic Claude Sonnet 4.6 未在 project 可用

Evidence:
```
$ curl POST https://us-east5-aiplatform.googleapis.com/v1/projects/claude-ttft-test/\
   locations/us-east5/publishers/anthropic/models/claude-sonnet-4@20250514:rawPredict
{
  "error": {
    "code": 404,
    "message": "Publisher model `.../publishers/anthropic/models/claude-sonnet-4@20250514`
                was not found or your project does not have access to it.",
    "status": "NOT_FOUND"
  }
}

$ gcloud ai models list --region=us-east5 --project=claude-ttft-test
Listed 0 items.
```

Vertex Model Garden 里 Anthropic Claude partner 需要用户在 Console UI 或
`gcloud ai model-garden models accept-eula` 层面显式接受 T&C 并 enable, SA
credential (`gemini-cli-bot`) 无法代替 user 做这一步。

结论: 无法配置 AnyChain Agent `LLM_PROVIDER=claude` + Vertex auth。

Required action (需 Leland UI 操作):
- Google Cloud Console → Vertex AI → Model Garden → Anthropic Claude Sonnet 4.6 → Enable
- 或提供 `ANTHROPIC_API_KEY` 走 direct Anthropic Messages API (次选, 引入新 secret)

### Blocker 3 · Vertex Gemini 也 404 (fallback provider 一并不可用)

Evidence:
```
$ curl POST .../publishers/google/models/gemini-2.0-flash-exp:generateContent
{
  "error": {
    "code": 404,
    "message": "Publisher model `.../gemini-2.0-flash-exp`
                was not found or your project does not have access to it.",
    "status": "NOT_FOUND"
  }
}
```

即使降级用 Leland 提到的 "Gemini 3.5 Flash" 作为 Agent provider, 也需要先
enable `aiplatform.googleapis.com` **且** SA 有 `roles/aiplatform.user` **且**
project 允许调用该 publisher model。当前状态两者都可能欠缺。

Required action (需 Leland):
- `gcloud services enable aiplatform.googleapis.com --project=claude-ttft-test`
- Grant `roles/aiplatform.user` to `gemini-cli-bot@claude-ttft-test.iam.gserviceaccount.com`
- 或改用 `application_default_credentials.json` 里 **user credential** (`gcloud auth application-default login`) 而非 SA key

## 未测试的路径 (Not tested because of blockers)

以下 scenario 全部未开始, 不能视为 pass 也不能视为 fail:

- **`WW-STARTUP-001`** ~ **`WW-RESTART-001`** (12 项 pending scenario, 全部未启动)
- **`FORMAL-CASE2-001`** (frozen journey `g4-chaos-636156190aa2cbbc0ce43df2`, 仍然 blocked, 且换 provider 也不能关 formal 证据 —— `model_substitution_closes_formal_codex_evidence: false`)
- **`FINAL-G0-G6-001`** (pending, 需先有 live evidence)

## 分类

- Classification: `infrastructure_interrupted` + `auth_blocker`
- Suspected root owner: `unknown` (workstation setup layer, 不属于 target revision code)
- Supporting evidence: 上文 3 段 curl / docker CLI 输出
- Evidence against stale session/data contamination: 全新 GCE session, 无历史 `.agent/` 或 checkpoint
- Not tested because: docker-compose plugin absent + Vertex partner model access absent

## 安全传输

- Redaction review completed by: CE-Project Bot (Claude Opus 4.7, GCE workstation)
- Bundle path: N/A (无 raw evidence 生成, 未创建 `.agent/external-validation/<revision>/`)
- Bundle SHA-256: N/A
- Internal `SHA256SUMS`: N/A
- Missing evidence: 全部 (未执行任何 CLI journey)

## Codex 无法在个人电脑修复本 blocker

- 本 blocker 位于**工作电脑 host 环境**层 (docker plugin + Vertex model garden access), **不在** target revision `a22108e5...` 的 tracked repo 里。
- Codex 修任何代码都无法解除本 blocker。
- Required unblock 完全在 Leland 侧: `sudo apt` + Vertex Console Enable + IAM grant。

## 建议 Leland 醒来后的行动 (按优先级)

1. **P0** — 装 docker-compose-plugin:
   ```
   sudo apt-get update && sudo apt-get install -y docker-compose-plugin
   ```
2. **P0** — Vertex Model Garden UI enable Anthropic Claude Sonnet 4.6
   (或提供 `ANTHROPIC_API_KEY` 走 direct)
3. **P1** — 确认 `claude-ttft-test` 已启用 `aiplatform.googleapis.com` 且
   `gemini-cli-bot` SA 有 `roles/aiplatform.user`
4. **P1** — 决定 Chaos test 用 Vertex Claude 还是 direct Anthropic API
   (Vertex 复用 ADC 更安全, direct API key 更快但引入 secret)
5. **P2** — 让当前 GCE 用户加入 `docker` 组 (`sudo usermod -aG docker $USER`
   + relogin), 或允许我 `sudo -n docker` 继续用

3 项 P0 解除后, 我立刻按 RUNBOOK Step 1 → 5 全流程开跑 12 WW-* scenario
+ Case 2, 32 并发, response-driven, 每 scenario 输出 finding 或 observed pass
到 `.ai-handoff/agent-external-validation/findings/`。
