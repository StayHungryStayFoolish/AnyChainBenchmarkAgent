#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - <<'PY'
from pathlib import Path

entry = Path("blockchain_node_benchmark.sh").read_text()
user_config = Path("config/user_config.sh").read_text()

required_entry_tokens = [
    "start_observability_stack()",
    "stop_observability_stack()",
    "OBSERVABILITY_STACK_ENABLED",
    "OBSERVABILITY_STACK_AUTO_STOP",
    "start_observability_stack",
    "stop_observability_stack || true",
    "BENCHMARK_DATA_DIR",
    "BENCHMARK_MEMORY_DIR",
]

missing = [token for token in required_entry_tokens if token not in entry]
if missing:
    raise SystemExit(f"entrypoint observability lifecycle tokens missing: {missing}")

start_call = entry.index("start_observability_stack\n")
phase1 = entry.index('echo "📋 Phase 1: Start RPC proxy"')
if start_call > phase1:
    raise SystemExit("observability stack must start before Phase 1 traffic/proxy setup")

cleanup = entry.index("cleanup_framework()")
stop_call = entry.index("stop_observability_stack || true", cleanup)
cleanup_temp = entry.index("cleanup_temp_files", cleanup)
if stop_call > cleanup_temp:
    raise SystemExit("observability stack should stop before temp runtime cleanup")

for token in [
    'OBSERVABILITY_STACK_ENABLED="${OBSERVABILITY_STACK_ENABLED:-false}"',
    'OBSERVABILITY_STACK_AUTO_STOP="${OBSERVABILITY_STACK_AUTO_STOP:-true}"',
    "export OBSERVABILITY_STACK_ENABLED OBSERVABILITY_STACK_AUTO_STOP",
]:
    if token not in user_config:
        raise SystemExit(f"user_config observability token missing: {token}")

print("observability lifecycle contract ok")
PY
