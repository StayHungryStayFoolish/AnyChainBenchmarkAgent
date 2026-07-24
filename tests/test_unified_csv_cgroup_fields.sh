#!/usr/bin/env bash
# tests/test_unified_csv_cgroup_fields.sh
# Proves cgroup_collector.py is wired into unified_monitor.sh main pipeline.
# This test verifies cgroup_collector is part of the active monitoring pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIFIED="${REPO_ROOT}/monitoring/unified_monitor.sh"
COLLECTOR="${REPO_ROOT}/monitoring/cgroup_collector.py"
WRAPPER="${REPO_ROOT}/monitoring/lib/cgroup_collector_wrapper.sh"
LINE_BUILDER="${REPO_ROOT}/monitoring/lib/performance_data_line_builder.sh"

PASS=0
FAIL=0
assert_pass() { PASS=$((PASS+1)); echo "  ✓ $1"; }
assert_fail() { FAIL=$((FAIL+1)); echo "  ✗ $1"; }
extract_function() {
  local file="$1"
  local name="$2"
  awk -v signature="${name}()" '
    index($0, signature) == 1 { inside=1 }
    inside { print }
    inside && $0 == "}" { exit }
  ' "$file"
}

# --- Test 1: source files exist ---
echo "Test 1: source files exist"
for f in "$UNIFIED" "$COLLECTOR" "$WRAPPER" "$LINE_BUILDER"; do
  [[ -f "$f" ]] && assert_pass "exists: ${f##*/}" || assert_fail "missing: $f"
done

# --- Test 2: wrapper is loaded and owns the cgroup helpers ---
echo "Test 2: cgroup wrapper owns helpers and is loaded by unified monitor"
if grep -q 'lib/cgroup_collector_wrapper.sh' "$UNIFIED"; then
  assert_pass "unified_monitor loads cgroup_collector_wrapper"
else
  assert_fail "unified_monitor does not load cgroup_collector_wrapper"
fi
for fn in get_cgroup_header get_cgroup_data; do
  if grep -qE "^${fn}\\(\\)" "$WRAPPER"; then
    assert_pass "wrapper defines $fn"
  else
    assert_fail "wrapper does not define $fn"
  fi
done

# --- Test 3: generate_csv_header calls get_cgroup_header ---
echo "Test 3: generate_csv_header wires cgroup_header"
if extract_function "$UNIFIED" generate_csv_header | grep -q 'cgroup_header=$(get_cgroup_header)'; then
  assert_pass "generate_csv_header calls get_cgroup_header"
else
  assert_fail "generate_csv_header does NOT call get_cgroup_header"
fi

# --- Test 4: collection and row builder preserve cgroup data ---
echo "Test 4: log_performance_data passes cgroup data to the row builder"
if extract_function "$UNIFIED" log_performance_data | grep -q 'cgroup_data=$(get_cgroup_data)' \
    && extract_function "$UNIFIED" log_performance_data | grep -q '"$cgroup_data"'; then
  assert_pass "log_performance_data collects and passes cgroup_data"
else
  assert_fail "log_performance_data does not pass collected cgroup_data"
fi
builder_refs=$(extract_function "$LINE_BUILDER" build_performance_data_line | grep -c '$cgroup_data' || true)
if [[ "$builder_refs" -eq 2 ]]; then
  assert_pass "row builder includes cgroup_data in ENA and non-ENA rows"
else
  assert_fail "row builder does not preserve cgroup_data in both branches"
fi

# --- Test 5: collector --header/--data produce 19 fields ---
echo "Test 5: cgroup_collector.py contract (19 fields)"
header_n=$(python3 "$COLLECTOR" --header 2>/dev/null | tr ',' '\n' | wc -l)
if [[ "$header_n" -eq 19 ]]; then
  assert_pass "--header → 19 fields"
else
  assert_fail "--header → $header_n fields (expected 19)"
fi
data_n=$(python3 "$COLLECTOR" --data 2>/dev/null | tr ',' '\n' | wc -l)
if [[ "$data_n" -eq 19 ]]; then
  assert_pass "--data → 19 fields"
else
  assert_fail "--data → $data_n fields (expected 19)"
fi

# --- Test 6: disabled flag honored ---
echo "Test 6: CGROUP_COLLECTOR_ENABLED=false produces 19 placeholder fields"
disabled_out=$(bash -c "
  CGROUP_COLLECTOR_ENABLED=false
  source '$WRAPPER'
  get_cgroup_data
")
fields_n=$(echo "$disabled_out" | tr ',' '\n' | wc -l)
if [[ "$fields_n" -eq 19 ]]; then
  assert_pass "disabled mode emits 19 placeholder fields"
else
  assert_fail "disabled mode emitted $fields_n fields"
fi
if echo "$disabled_out" | grep -q "disabled"; then
  assert_pass "disabled mode meta_source='disabled' sentinel"
else
  assert_fail "disabled mode missing 'disabled' sentinel"
fi

# --- Test 7: missing-collector fallback ---
echo "Test 7: missing collector path produces 19 placeholder fields"
missing_out=$(bash -c "
  CGROUP_COLLECTOR_PATH='/definitely/missing/cgroup_collector.py'
  source '$WRAPPER'
  get_cgroup_data
" 2>/dev/null || echo "")
fields_n=$(echo "$missing_out" | tr ',' '\n' | wc -l)
if [[ "$fields_n" -eq 19 ]]; then
  assert_pass "missing collector → 19 placeholder fields (fail-soft)"
else
  assert_fail "missing collector emitted $fields_n fields"
fi
if echo "$missing_out" | grep -q "unavailable"; then
  assert_pass "missing collector meta_source='unavailable' sentinel"
else
  assert_fail "missing collector missing 'unavailable' sentinel"
fi

# --- Summary ---
echo ""
echo "=================================="
echo "cgroup_in_unified_csv: $PASS pass, $FAIL fail"
echo "=================================="
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
