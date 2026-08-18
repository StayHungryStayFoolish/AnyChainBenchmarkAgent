# Local Closed-Loop Testing and fake-node Guide

[中文](../zh/local-closed-loop-testing.md) | [English](local-closed-loop-testing.md)

This guide explains how to use fake-node for local Linux / Docker closed-loop testing. For adding chains or RPC methods, see [How to add a blockchain or RPC method](how-to-add-chain.md).

## Goal

A local closed-loop test verifies that:

- Chain templates can be loaded.
- Vegeta targets can be generated from `single` or `mixed_weighted`.
- Requests go through the proxy and are recorded in `proxy_method.csv`.
- fake-node returns real recorded responses by `chain + method + fixture`.
- Block height / sync health is written to monitoring CSV.
- Analysis, visualization, and HTML reports are generated.
- Results are archived under `archives/run_*`.

fake-node is not a performance benchmark target. It does not represent real node throughput, latency, database cost, or consensus behavior. Its job is deterministic framework regression testing when 36 real nodes are not available.

## Recommended Environment

Run this in Docker, Ubuntu, or a Linux VM.

The framework does not target macOS as a production-compatible runtime. macOS is fine for editing code, but full monitoring, disk-device fields, process metrics, and cgroup behavior should be verified in Linux/Docker.

## Test Levels

Run these layers from fastest to most complete:

1. Fixture authenticity and coverage.
2. fake-node runtime probe.
3. block height / sync health runtime probe.
4. Full `blockchain_node_benchmark.sh --fake-node` entrypoint test.

## 1. Check Committed Fixture Coverage

First make sure every configured workload RPC method has a committed fake-node fixture.

```bash
python3 tools/fake-node/check_fixture_coverage.py --json
```

Expected:

- 184 workload RPC methods are covered.
- 0 missing.

For local authenticity checks after recording fresh request/response evidence, run:

```bash
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/validate_fixture_authenticity.py --json
python3 tools/fake-node/check_fixture_coverage.py --json --strict
```
- 0 JSON-RPC semantic error.

If this fails, do not run the full E2E yet. Re-record fixtures first.

## 2. Start fake-node Manually

Build:

```bash
cd tools/fake-node
go build -o /tmp/fake-node-v2 .
```

Start one chain:

```bash
BLOCKCHAIN_NODE=solana /tmp/fake-node-v2 -port 19101
```

Or override with a flag:

```bash
/tmp/fake-node-v2 -chain ethereum -port 19102
```

Common flags:

- `-chain`: chain name, overrides `BLOCKCHAIN_NODE`.
- `-chains-dir`: chain template directory, default `../../config/chains`.
- `-configs-dir`: fake-node family YAML directory, default `configs`.
- `-fixtures-dir`: fixture root, default `./fixtures`.
- `-port`: listen port, default `19000`.

## 3. fake-node Runtime Probe

The runtime probe starts fake-node per chain, generates requests through the production adapter, and performs real HTTP calls against fake-node.

```bash
python3 tools/fake-node/runtime_probe.py
```

Check options:

```bash
python3 tools/fake-node/runtime_probe.py --help
```

Expected:

- 184/184 workload calls ok.
- Every target URL points to the fake-node endpoint.
- No method returns non-200 HTTP.

This is stricter than coverage. Coverage says files exist; runtime probe proves that production adapter-generated requests hit the expected fake-node fixtures.

### After Changing `mixed_weighted`

If you add, remove, or reweight mixed-mode RPC methods, verify the change before
running a full benchmark:

`weight` must be a positive integer. Missing, invalid, or zero weights are
normalized to `1`; remove a method from `mixed_weighted` when you do not want it
to appear in generated Vegeta targets.

```bash
# 1. Confirm the chain template can build all mixed targets.
python3 tests/test_chain_adapters.py

# 2. Confirm weighted target generation still follows the configured ratio.
bash tests/test_target_generator_mixed_weighted.sh

# 3. Record fixtures for the changed chain or chain list.
tools/fake-node/record_rpc_fixtures.sh <chain>

# 4. Check that every configured method has a fake-node fixture.
python3 tools/fake-node/check_fixture_coverage.py --json

# 5. Exercise real adapter-generated HTTP calls against fake-node.
python3 tools/fake-node/runtime_probe.py
```

Acceptance criteria:

- The changed chain has no missing fixtures.
- The runtime probe returns HTTP 200 for every configured mixed method.
- `proxy_method.csv` from a quick entrypoint run shows the workload methods you
  expect, not only sync-health or status methods.
- The generated HTML report includes per-method attribution for those workload
  methods.

## 4. Block Height / Sync Health Runtime Probe

This probe exercises the production bash entrypoint `core/common_functions.sh:get_block_height()`:

```bash
python3 tools/fake-node/runtime_probe_block_height.py
```

Expected:

- 36/36 chains ok.
- Chain templates, adapter health probes, fake-node fixtures, parse-height logic, and the bash entrypoint are connected.

Also run registry audits:

```bash
python3 tools/audit_sync_health_registry.py --write --json
python3 tools/fake-node/audit_health_probe_fixtures.py
```

## 5. Prepare the Full Entrypoint E2E

The full entrypoint uses:

- fake-node as the local blockchain node.
- proxy to record methods.
- vegeta for load generation.
- monitoring for resources and chain health.
- analysis/report generation.
- archiver for final results.

Prepare a small active account file so the test does not depend on a real node for account discovery:

```bash
mkdir -p //blockchain-node-benchmark-result/current/tmp
cat > //blockchain-node-benchmark-result/current/tmp/active_accounts.txt <<EOF
11111111111111111111111111111111
TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
SysvarRent111111111111111111111111111111111
ComputeBudget111111111111111111111111111111
EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
EOF
```

This is a Solana example. Other chains need addresses or samples in that chain's real format.

## 6. Run a Quick Closed-Loop Test

Use a 3-second quick smoke first to avoid default warmup/cooldown wait time.

```bash
export BLOCKCHAIN_NODE=solana
export RPC_MODE=mixed
export QUICK_INITIAL_QPS=1
export QUICK_MAX_QPS=1
export QUICK_QPS_STEP=1
export QUICK_DURATION=3
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0
export BLOCK_HEIGHT_MONITOR_RATE=1
export BLOCK_HEIGHT_CURL_TIMEOUT=2

./blockchain_node_benchmark.sh \
  --quick \
  --mixed \
  --fake-node \
  --initial-qps 1 \
  --max-qps 1 \
  --step-qps 1 \
  --duration 3
```

Notes:

- `--fake-node` builds and starts fake-node automatically.
- The framework starts the proxy before generating vegeta targets, so traffic goes through the proxy.
- `QPS_WARMUP_DURATION=0` and `QPS_COOLDOWN=0` keep local regression fast.
- `RPC_MODE=mixed` uses `rpc_methods.mixed_weighted`.

## 7. Inspect Results

Completed runs are archived under:

```text
//blockchain-node-benchmark-result/archives/run_<N>_<timestamp>
```

Find the latest archive:

```bash
ls -lt //blockchain-node-benchmark-result/archives | head
```

Important files:

```text
logs/proxy_method.csv
logs/performance_*.csv
logs/block_height_monitor_*.csv
vegeta_results/vegeta_*qps_*.json
reports/performance_report_zh_*.html
reports/performance_report_en_*.html
reports/per_method_charts/
stats/qps_status.json
test_summary.json
```

### Acceptance Checks

`proxy_method.csv` should contain workload methods, not only health probes.

Example:

```bash
awk -F, 'NR>1 {count[$2]++} END {for (m in count) print m,count[m]}' \
  //blockchain-node-benchmark-result/archives/<run-id>/logs/proxy_method.csv
```

`block_height_monitor_*.csv` should contain:

- `sync_mode`
- `sync_status`
- `lag_value`
- `lag_unit`
- `probe_error`

Both English and Chinese HTML reports should be generated.

If Docker has no real DATA disk device, disk professional charts may be skipped. This is not a fake-node problem. Test disk chart rendering with synthetic data:

```bash
python3 tests/test_disk_visualization_synthetic.py
```

Example chart output:

```text
docs/audit/disk-visualization-synthetic/
```

## 8. Single Mode Closed-Loop Test

To verify single mode:

```bash
export BLOCKCHAIN_NODE=solana
export RPC_MODE=single
export QUICK_INITIAL_QPS=1
export QUICK_MAX_QPS=1
export QUICK_QPS_STEP=1
export QUICK_DURATION=3
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0

./blockchain_node_benchmark.sh \
  --quick \
  --single \
  --fake-node \
  --initial-qps 1 \
  --max-qps 1 \
  --step-qps 1 \
  --duration 3
```

Single mode uses the method configured in `rpc_methods.single`.

## 9. Common Regression Commands

```bash
bash tests/test_config_env_overrides.sh
bash tests/test_csv_registry_symmetry.sh
python3 tests/test_disk_visualization_synthetic.py
python3 tests/test_per_method_charts.py
python3 tests/test_degraded_report.py
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py
python3 tools/fake-node/runtime_probe_block_height.py
python3 tools/audit_sync_health_registry.py --write --json
python3 tools/fake-node/audit_health_probe_fixtures.py
```

## 10. Troubleshooting

### proxy_method.csv Only Contains getHealth

This usually means vegeta targets bypassed the proxy and went directly to fake-node or the local node.

Check:

- Target URLs should use `http://localhost:18545`.
- `blockchain_node_benchmark.sh` should start the proxy before target generation.
- `LOCAL_RPC_URL` should be rewritten after proxy startup.

### All vegeta Requests Fail

Check:

- fake-node is running.
- `BLOCKCHAIN_NODE` matches a chain template file.
- fake-node fixture coverage passes.
- Active account samples match that chain's address format.

### Coverage Passes but Runtime Probe Fails

Coverage only says files exist. Runtime probe failures usually mean:

- Adapter-generated requests do not match fixture mapping.
- REST path rendering drifted.
- fake-node family YAML maps the method to the wrong fixture file.

### Block Height Runtime Probe Fails

Check:

- `_meta.sync_health` exists.
- The adapter can generate a health probe request.
- fake-node has the health probe fixture.
- parse-height supports the response structure.

### No Disk Charts in Docker

Docker may not expose production DATA device metrics, such as `/dev/sda` or matching iostat fields. The framework should skip disk professional charts without blocking report generation.

Use synthetic data to test chart functionality:

```bash
python3 tests/test_disk_visualization_synthetic.py
```

### Quick Test Still Waits 60s Warmup or 30s Cooldown

Export these before running:

```bash
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0
```

Then run:

```bash
bash tests/test_config_env_overrides.sh
```

## Minimum Closed-Loop Acceptance Checklist

A local fake-node closed loop should satisfy:

- [ ] Fixture authenticity passes.
- [ ] Committed fixture coverage passes.
- [ ] `tools/fake-node/runtime_probe.py` passes.
- [ ] `tools/fake-node/runtime_probe_block_height.py` passes.
- [ ] `blockchain_node_benchmark.sh --fake-node` exits 0.
- [ ] `proxy_method.csv` contains workload methods.
- [ ] vegeta result success > 0.
- [ ] Block height monitor CSV contains `sync_mode` / `sync_status`.
- [ ] HTML report is generated.
- [ ] Results are archived under `archives/run_*`.
