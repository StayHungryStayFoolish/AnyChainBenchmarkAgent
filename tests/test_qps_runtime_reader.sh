#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

# shellcheck source=/dev/null
source monitoring/lib/qps_runtime_reader.sh

status_file="$TEST_ROOT/qps_test_status"
vegeta_dir="$TEST_ROOT/vegeta"
mkdir -p "$vegeta_dir"

out="$(get_qps_runtime_fields "$status_file" "$vegeta_dir")"
[[ "$out" == "0,0.0,false" ]] || {
    echo "Missing status output mismatch: $out"
    exit 1
}

QPS_STATUS_MODE=sync_observe
out="$(get_qps_runtime_fields "$status_file" "$vegeta_dir")"
unset QPS_STATUS_MODE
[[ "$out" == "0,0.0,false" ]] || {
    echo "sync-observe output mismatch: $out"
    exit 1
}

echo "running qps:42" > "$status_file"
out="$(get_qps_runtime_fields "$status_file" "$vegeta_dir")"
[[ "$out" == "42,0.0,true" ]] || {
    echo "Status-only output mismatch: $out"
    exit 1
}

cat > "$vegeta_dir/vegeta_42qps_20260611_120000.json" <<'JSON'
{
  "latencies": {
    "mean": 123000000
  }
}
JSON

out="$(get_qps_runtime_fields "$status_file" "$vegeta_dir")"
[[ "$out" == "42,123,true" || "$out" == "42,123.0,true" ]] || {
    echo "Status + vegeta output mismatch: $out"
    exit 1
}

echo "✅ qps_runtime_reader returns expected QPS runtime fields"
