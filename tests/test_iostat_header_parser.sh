#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

IOSTAT_COLLECTOR_SKIP_CONFIG_LOAD=1 source monitoring/iostat_collector.sh >/dev/null 2>&1

header='Device r/s rkB/s rrqm/s %rrqm r_await rareq-sz w/s wkB/s wrqm/s %wrqm w_await wareq-sz d/s dkB/s drqm/s %drqm d_await dareq-sz f/s f_await aqu-sz %util'
row='vda 0.00 0.02 0.00 0.00 0.29 12.90 3.19 1350.46 2.41 43.00 1.34 423.40 2.57 17793.59 0.00 0.00 0.12 6923.07 0.45 1.89 0.01 0.30'

assert_field() {
    local field="$1"
    local expected="$2"
    local actual
    actual="$(_iostat_field_by_header "$header" "$row" "$field")"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $field expected $expected, got $actual" >&2
        exit 1
    fi
}

assert_field "w/s" "3.19"
assert_field "wkB/s" "1350.46"
assert_field "w_await" "1.34"
assert_field "aqu-sz" "0.01"
assert_field "%util" "0.30"
assert_field "rareq-sz" "12.90"
assert_field "wareq-sz" "423.40"

missing="$(_iostat_field_by_header "$header" "$row" "missing_field" "NA")"
if [[ "$missing" != "NA" ]]; then
    echo "FAIL: missing field fallback expected NA, got $missing" >&2
    exit 1
fi

echo "PASS: iostat parser reads metrics by header name"
