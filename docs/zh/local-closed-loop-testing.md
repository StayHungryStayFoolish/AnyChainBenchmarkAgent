# 本地闭环测试与 fake-node 使用指南

[中文](local-closed-loop-testing.md) | [English](../en/local-closed-loop-testing.md)

本文档说明如何在本地 Linux / Docker 环境用 fake-node 验证框架闭环。新增链和新增 RPC method 的流程见 [如何新增区块链或 RPC Method](how-to-add-chain.md)。

## 目标

本地闭环测试要验证的是：

- chain template 能被加载。
- vegeta targets 能按 `single` 或 `mixed_weighted` 生成。
- 请求会经过 proxy，并记录到 `proxy_method.csv`。
- fake-node 能按 `chain + method + fixture` 返回真实录制响应。
- block height / sync health 能写入监控 CSV。
- analysis / visualization / HTML report 能生成。
- 结果能归档到 `archives/run_*`。

fake-node 不是真实性能基准。它不能代表真实节点的吞吐、延迟、数据库查询成本或共识行为。它的职责是让框架在没有 36 个真实节点时仍能做确定性回归测试。

## 推荐环境

推荐在 Docker / Ubuntu / Linux VM 中运行。

本框架不以 macOS 作为生产兼容目标。macOS 可以编辑代码，但完整监控链路、磁盘设备字段、进程/cgroup 行为应以 Linux/Docker 测试结果为准。

## 测试层级

建议按 4 层跑，越往下越接近完整框架：

1. Fixture 真实性与覆盖率检查。
2. fake-node runtime probe。
3. block height / sync health runtime probe。
4. `blockchain_node_benchmark.sh --fake-node` 完整入口闭环。

## 1. 检查已提交 Fixture 覆盖率

先确认每个已配置的 workload RPC method 都有已提交的 fake-node fixture。

```bash
python3 tools/fake-node/check_fixture_coverage.py --json
```

期望：

- 184 个 workload RPC method 都有 fixture 覆盖。
- 0 missing。

如果本地重新录制了完整 request/response evidence，可以再运行真实性检查：

```bash
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/validate_fixture_authenticity.py --json
python3 tools/fake-node/check_fixture_coverage.py --json --strict
```
- 0 JSON-RPC semantic error。

如果这里失败，不要继续跑完整 E2E。先重新录制 fixture。

## 2. 单独启动 fake-node

手动构建：

```bash
cd tools/fake-node
go build -o /tmp/fake-node-v2 .
```

启动某条链：

```bash
BLOCKCHAIN_NODE=solana /tmp/fake-node-v2 -port 19101
```

也可以用 flag 覆盖：

```bash
/tmp/fake-node-v2 -chain ethereum -port 19102
```

常用参数：

- `-chain`：链名，覆盖 `BLOCKCHAIN_NODE`。
- `-chains-dir`：chain template 目录，默认 `../../config/chains`。
- `-configs-dir`：fake-node family YAML 目录，默认 `configs`。
- `-fixtures-dir`：fixture 根目录，默认 `./fixtures`。
- `-port`：监听端口，默认 `19000`。

## 3. fake-node Runtime Probe

runtime probe 会逐链启动 fake-node，用生产 adapter 生成请求，然后真实 HTTP 请求 fake-node。

```bash
python3 tools/fake-node/runtime_probe.py
```

查看参数：

```bash
python3 tools/fake-node/runtime_probe.py --help
```

期望：

- 184/184 workload calls ok。
- 所有 target URL 都指向 fake-node endpoint。
- 没有 method 返回 HTTP 非 200。

这个测试比 coverage 更严格。coverage 只说明文件存在，runtime probe 会验证 adapter 生成的真实请求能打到 fake-node fixture。

### 修改 `mixed_weighted` 后的验证

如果新增、删除或调整 mixed-mode RPC method 权重，先完成下面验证，再跑完整 benchmark：

`weight` 必须是正整数。缺失、非法或为 0 的权重都会被归一化为 `1`；如果不希望某个
method 出现在生成的 Vegeta targets 中，需要从 `mixed_weighted` 中移除。

```bash
# 1. 确认 chain template 可以构造所有 mixed targets。
python3 tests/test_chain_adapters.py

# 2. 确认 weighted target generation 仍按配置比例生成。
bash tests/test_target_generator_mixed_weighted.sh

# 3. 为变更的链或链列表重新录制 fixture。
tools/fake-node/record_rpc_fixtures.sh <chain>

# 4. 确认每个已配置 method 都有 fake-node fixture。
python3 tools/fake-node/check_fixture_coverage.py --json

# 5. 用真实 adapter 生成 HTTP 请求并打到 fake-node。
python3 tools/fake-node/runtime_probe.py
```

验收标准：

- 变更链没有 missing fixtures。
- runtime probe 中每个 mixed method 都返回 HTTP 200。
- quick entrypoint run 生成的 `proxy_method.csv` 中能看到预期 workload method，
  而不是只有 sync-health 或 status method。
- 生成的 HTML report 中包含这些 workload method 的 per-method attribution。

## 4. Block Height / Sync Health Runtime Probe

这个 probe 会测试生产 bash 入口 `core/common_functions.sh:get_block_height()`：

```bash
python3 tools/fake-node/runtime_probe_block_height.py
```

期望：

- 36/36 chain ok。
- chain template、adapter health probe、fake-node fixture、parse-height、bash 入口没有断裂。

同时建议跑 registry 审计：

```bash
python3 tools/audit_sync_health_registry.py --write --json
python3 tools/fake-node/audit_health_probe_fixtures.py
```

## 5. 准备完整入口 E2E

完整入口会用：

- fake-node 模拟本地区块链节点。
- proxy 统计 method。
- vegeta 产生请求。
- monitoring 采集资源和区块健康。
- analysis/report 生成图表和 HTML。
- archiver 归档结果。

先准备少量 active accounts，避免测试时依赖真实节点抓账户：

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

上面是 Solana 示例。其他链应放该链真实格式的地址或样本值。

## 6. 运行 Quick 闭环

推荐先用 3 秒 quick smoke，避免每次等待默认 warmup/cooldown。

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

说明：

- `--fake-node` 会自动构建并启动 fake-node。
- 框架会先启动 proxy，再生成 vegeta targets，确保请求经过 proxy。
- `QPS_WARMUP_DURATION=0` 和 `QPS_COOLDOWN=0` 用于缩短本地回归时间。
- `RPC_MODE=mixed` 会走 `rpc_methods.mixed_weighted`。

## 7. 检查结果

运行完成后会归档到：

```text
//blockchain-node-benchmark-result/archives/run_<N>_<timestamp>
```

查看最新归档：

```bash
ls -lt //blockchain-node-benchmark-result/archives | head
```

重点检查这些文件：

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

### 关键验收点

`proxy_method.csv` 应该包含 workload method，而不是只有 health probe。

示例：

```bash
awk -F, 'NR>1 {count[$2]++} END {for (m in count) print m,count[m]}' \
  //blockchain-node-benchmark-result/archives/<run-id>/logs/proxy_method.csv
```

`block_height_monitor_*.csv` 应该包含：

- `sync_mode`
- `sync_status`
- `lag_value`
- `lag_unit`
- `probe_error`

HTML 报告应生成中英文版本。

如果 Docker 环境没有真实 DATA disk 设备，disk 专业图可能跳过。这不是 fake-node 问题。可以用合成数据测试 disk 图表：

```bash
python3 tests/test_disk_visualization_synthetic.py
```

样例图位置：

```text
docs/audit/disk-visualization-synthetic/
```

## 8. Single 模式闭环

如果要验证 single mode：

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

single mode 使用 chain template 中 `rpc_methods.single` 指定的 method。

## 9. 常用回归命令

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

## 10. 常见问题

### proxy_method.csv 只有 getHealth

通常说明 vegeta target 绕过了 proxy，直接打到了 fake-node 或本地节点。

检查：

- target 文件中的 URL 是否是 `http://localhost:18545`。
- `blockchain_node_benchmark.sh` 是否先启动 proxy，再生成 targets。
- `LOCAL_RPC_URL` 是否在 proxy 启动后被替换。

### vegeta 全部失败

优先检查：

- fake-node 是否启动。
- `BLOCKCHAIN_NODE` 是否和 chain template 文件名一致。
- fake-node fixture 覆盖率是否通过。
- active account 样本是否符合该链地址格式。

### fixture coverage ok，但 runtime probe 失败

coverage 只说明文件存在。runtime probe 失败通常表示：

- adapter 生成的请求和 fixture mapping 不一致。
- REST path 渲染不一致。
- fake-node family YAML 中 method 到 fixture 文件名映射错误。

### block height runtime probe 失败

检查：

- `_meta.sync_health` 是否存在。
- adapter health probe 是否能生成请求。
- fake-node 是否有 health probe fixture。
- parse-height 是否支持该响应结构。

### Docker 中没有 disk 图

Docker 可能没有生产 DATA device 指标，例如没有 `/dev/sda` 或对应 iostat 字段。完整框架会跳过 disk 专业图，但不应阻断报告生成。

用合成数据验证图表功能：

```bash
python3 tests/test_disk_visualization_synthetic.py
```

### quick 测试仍然等待 60s warmup 或 30s cooldown

确认环境变量在运行脚本前 export：

```bash
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0
```

并运行：

```bash
bash tests/test_config_env_overrides.sh
```

## 最小闭环验收清单

一次本地 fake-node 闭环至少满足：

- [ ] fixture authenticity 通过。
- [ ] fixture coverage strict 通过。
- [ ] `tools/fake-node/runtime_probe.py` 通过。
- [ ] `tools/fake-node/runtime_probe_block_height.py` 通过。
- [ ] `blockchain_node_benchmark.sh --fake-node` 退出码为 0。
- [ ] `proxy_method.csv` 中有 workload method。
- [ ] vegeta result success > 0。
- [ ] block height monitor CSV 有 `sync_mode` / `sync_status`。
- [ ] HTML 报告生成。
- [ ] 结果归档到 `archives/run_*`。
