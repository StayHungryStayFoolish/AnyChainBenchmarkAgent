#!/usr/bin/env bash
# =====================================================================
# QPS Runtime Reader for Unified Monitor
# =====================================================================
# Reads the framework lifecycle marker and latest Vegeta JSON result, returning
# the three CSV fields embedded in performance_latest.csv:
#   current_qps,rpc_latency_ms,qps_data_available
# =====================================================================

get_latest_vegeta_latency_ms() {
    local vegeta_dir="${1:-${VEGETA_RESULTS_DIR:-}}"
    [[ -n "$vegeta_dir" ]] || { echo "0.0"; return 0; }

    local latest_vegeta_file
    latest_vegeta_file=$(ls -t "${vegeta_dir}"/vegeta_*qps_*.json 2>/dev/null | head -1 || true)

    if [[ -z "$latest_vegeta_file" || ! -f "$latest_vegeta_file" ]]; then
        echo "0.0"
        return 0
    fi

    if [[ ! -s "$latest_vegeta_file" ]] || ! grep -q "}" "$latest_vegeta_file" 2>/dev/null; then
        echo "0.0"
        return 0
    fi

    python3 - "$latest_vegeta_file" <<'PY' 2>/dev/null | tr -d '\n\r' || echo "0.0"
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    latency_ns = data.get("latencies", {}).get("mean", 0)
    print(latency_ns / 1_000_000)
except Exception:
    print(0.0)
PY
}

get_qps_runtime_fields() {
    local status_file="${1:-${TMP_DIR:-}/qps_test_status}"
    local vegeta_dir="${2:-${VEGETA_RESULTS_DIR:-}}"

    local current_qps="0"
    local rpc_latency_ms="0.0"
    local qps_data_available="false"

    if [[ "${QPS_STATUS_MODE:-}" == "sync_observe" || "${SYNC_OBSERVE_MODE:-false}" == "true" ]]; then
        echo "0,0.0,false"
        return 0
    fi

    if [[ -f "$status_file" ]]; then
        local qps_status_content
        qps_status_content=$(cat "$status_file" 2>/dev/null || true)

        if [[ -n "$qps_status_content" ]]; then
            current_qps=$(echo "$qps_status_content" | grep -o "qps:[0-9]*" | cut -d: -f2 || echo "0")
            [[ -z "$current_qps" ]] && current_qps="0"
            qps_data_available="true"
            rpc_latency_ms=$(get_latest_vegeta_latency_ms "$vegeta_dir")
            [[ -z "$rpc_latency_ms" ]] && rpc_latency_ms="0.0"
        fi
    fi

    current_qps=$(echo "$current_qps" | tr -d '\n\r' | head -c 20)
    rpc_latency_ms=$(echo "$rpc_latency_ms" | tr -d '\n\r' | head -c 20)
    qps_data_available=$(echo "$qps_data_available" | tr -d '\n\r' | head -c 10)

    echo "${current_qps},${rpc_latency_ms},${qps_data_available}"
}
