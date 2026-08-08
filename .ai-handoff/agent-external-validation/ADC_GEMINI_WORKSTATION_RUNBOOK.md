# 工作电脑 Gemini ADC 验证手册

本手册只授权工作电脑发现、复现和记录问题，不授权修改产品代码。

## 1. 获取协调分支并锁定被测 revision

```bash
git fetch origin docs/agent-handoff-chaos
git switch docs/agent-handoff-chaos
git pull --ff-only origin docs/agent-handoff-chaos

HANDOFF_ROOT="$PWD/.ai-handoff/agent-external-validation"
TARGET="$(python3 - <<'PY'
from pathlib import Path

for line in Path('.ai-handoff/agent-external-validation/VALIDATION_MANIFEST.yaml').read_text().splitlines():
    if line.startswith('target_revision:'):
        print(line.split(':', 1)[1].strip())
        break
PY
)"

git cat-file -e "$TARGET^{commit}"
git worktree add --detach ../anychain-agent-external-validation "$TARGET"
cd ../anychain-agent-external-validation
test "$(git rev-parse HEAD)" = "$TARGET"
test -z "$(git status --short)"
```

协调分支用于传递交接资料；detached worktree 才是被测产品。revision 不匹配或
tracked worktree 非空时立即停止，不能自行更新 manifest。

## 2. 配置 Google Cloud ADC

宿主机执行：

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project PROJECT_ID
gcloud services enable aiplatform.googleapis.com --project PROJECT_ID
```

用户身份需要目标项目的 Vertex AI 调用权限。无法启用 API 时记录 auth blocker，
不要自行提升权限或切换到未批准项目。

官方参考：

- https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart

在 detached test worktree 创建 gitignored 的 `config/agent_config.local.sh`：

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro-preview"
LLM_AUTH_MODE="google_adc"
GOOGLE_CLOUD_PROJECT="PROJECT_ID"
GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

不要修改 `config/agent_config.sh`。标准本地 ADC 使用 well-known credential 位置，
不设置 `GOOGLE_APPLICATION_CREDENTIALS`。

## 3. 启动 Linux/Docker

当前 compose 不会自动挂载宿主机 ADC。回到协调 checkout 取得
`HANDOFF_ROOT` 的绝对路径，再从 detached worktree 执行：

```bash
ADC="$HOME/.config/gcloud/application_default_credentials.json"
HANDOFF_ROOT="/absolute/path/to/coordinator/.ai-handoff/agent-external-validation"
test -r "$ADC"

docker compose run --rm -it \
  -v "$ADC:/root/.config/gcloud/application_default_credentials.json:ro" \
  -v "$HANDOFF_ROOT:/handoff:ro" \
  -e CLOUDSDK_CONFIG=/root/.config/gcloud \
  -e ANYCHAIN_VALIDATION_TARGET="$TARGET" \
  bench bash
```

容器内：

```bash
bash scripts/install_agent_deps.sh --yes --with-google-search
python3 -m agent.cli llm-config
python3 -m agent.cli doctor
python3 -m agent.cli llm-smoke --prompt 'Return JSON only: {"ok": true}'
python3 -m agent.cli adk-status
./bin/anychain-agent
```

`llm-config` 必须显示 Gemini、`google_adc`、正确 project 和 `global`；
`llm-smoke` 必须完成真实调用。`adk-status` 只证明 google_search 可配置和可导入，
不等于实际搜索已经发生。真实 CLI journey 还必须提供 typed receipt：

- `google_search_invoked=true`
- `google_search_available=true`
- 至少一个官方 citation

readiness 失败时停止下游测试，记录 provider/auth/infrastructure blocker，不得把
未执行路径记成产品失败或产品通过。

## 4. 初始化并记录 evidence

容器内执行：

```bash
TARGET="${ANYCHAIN_VALIDATION_TARGET:?missing validation target}"
EVIDENCE_ROOT=".agent/external-validation/$TARGET"
mkdir -p "$EVIDENCE_ROOT"/{findings,transcripts,evidence}
git rev-parse HEAD > "$EVIDENCE_ROOT/revision.txt"
{
  uname -a
  printf 'container='; hostname
  printf 'image='; sed -n 's/^PRETTY_NAME=//p' /etc/os-release
} > "$EVIDENCE_ROOT/environment.txt"
python3 -m agent.cli llm-config > "$EVIDENCE_ROOT/provider-status.json"
python3 -m agent.cli doctor > "$EVIDENCE_ROOT/doctor.json"
python3 -m agent.cli llm-smoke --prompt 'Return JSON only: {"ok": true}' \
  > "$EVIDENCE_ROOT/llm-smoke.json"
python3 -m agent.cli adk-status > "$EVIDENCE_ROOT/adk-status.json"
cp /handoff/FINDING_TEMPLATE.md \
  "$EVIDENCE_ROOT/findings/EXT-YYYYMMDD-NNN.md"
```

每条 journey 保存完整 PTY transcript：

```bash
script -q -c './bin/anychain-agent' \
  "$EVIDENCE_ROOT/transcripts/journey-ID.typescript"
```

Claude 每轮必须先读取 Agent 最新完整回复，再决定输入。每个 manifest scenario 都
必须有 finding 或 observed-pass 记录；未记录不能视为通过。

## 5. 打包并传回个人电脑

逐文件脱敏后执行：

```bash
BUNDLE="$PWD/.agent/anychain-external-validation-$TARGET.tar.gz"
(
  cd "$EVIDENCE_ROOT"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
tar -C ".agent/external-validation" -czf "$BUNDLE" "$TARGET"
sha256sum "$BUNDLE" > "$BUNDLE.sha256"
```

个人电脑上的 Codex 先验证压缩包 hash 和内部 `SHA256SUMS`，再在 finding 的 source
revision 上复现。修复形成新 revision 后，更新 manifest；工作电脑重新建立 clean
detached worktree，回测原路径和两个相邻变体。
