# 如何新增区块链或 RPC Method

[中文](how-to-add-chain.md) | [English](../en/how-to-add-chain.md)

本文档是当前框架的扩展实操手册。Chain template 字段以 `config/chains/*.json` 的现有示例和下文步骤为准，fake-node 细节见 [fake-node README](../../tools/fake-node/README.md)。

## 核心原则

新增链时要先区分两件事：

- **请求构造 schema 可以复用**：如果新链属于现有 family，可以复用 adapter、proxy extractor、target generator 和 fake-node handler。
- **响应 fixture 不能默认复用**：即使两个 method 都传 `address` 或 `tx_hash`，也必须录制该 `chain + method` 自己的真实响应。

fake-node 的匹配粒度是：

```text
chain + rpc method + family handler + fixture
```

所以新增链或新增 method 后，都要录制真实 request/response fixture，不能用参数名相同的旧 fixture 代替。

## 现有 6 个 Family 如何判断

Family 不是按品牌、token 或生态划分，而是按真实 RPC 请求和解析方式划分。

| Family | 适用情况 | 当前示例 |
|---|---|---|
| `jsonrpc` | 标准 JSON-RPC POST，请求 body 中有 `method` | Ethereum、Solana、Sui、Tron、Avalanche C |
| `bitcoin_jsonrpc` | Bitcoin Core / UTXO 风格 JSON-RPC，可能带 Basic Auth 或 REST workaround | Bitcoin、Litecoin、Dogecoin、BCH |
| `rest` | HTTP path/body 是主要接口，method 需要映射到 `_meta.rest_paths` | Aptos、Algorand、Cardano、Tezos、TON |
| `substrate` | Polkadot SDK / Substrate RPC，如 `chain_*`、`state_*`、`system_*` | Polkadot、Kusama、Acala |
| `tendermint` | Cosmos SDK / Tendermint / CometBFT REST-RPC | Cosmos Hub、Osmosis、Celestia |
| `hedera_dual` | Hedera Mirror REST + Hashio JSON-RPC Relay 双协议 | Hedera |

如果新链的请求 envelope、参数结构、endpoint 路由、认证/header、响应 envelope、区块高度解析方式都能归入其中之一，通常就是配置扩展。

如果这些能力现有 family 表达不了，才需要新增或扩展 adapter/fake-node handler。

## 新增一条现有 Family 的链

### 1. 新建 Chain Template

新增文件：

```text
config/chains/<chain>.json
```

建议先复制同 family 的相近链，例如：

- 新 EVM 链：复制 `config/chains/ethereum.json`、`arbitrum.json` 或 `base.json`
- 新 Cosmos 链：复制 `config/chains/cosmos-hub.json`
- 新 Substrate 链：复制 `config/chains/polkadot.json`
- 新 REST 链：复制 `config/chains/aptos.json` 或 `algorand.json`

### 2. 配置基础字段

必须确认这些字段：

```json
{
  "chain_type": "<chain>",
  "rpc_url": "LOCAL_RPC_URL",
  "params": {},
  "rpc_methods": {},
  "param_formats": {},
  "system_addresses": [],
  "_meta": {
    "adapter_family": "<family>"
  }
}
```

其中：

- `chain_type` 应与文件名一致。
- `rpc_url` 生产框架中通常保持 `LOCAL_RPC_URL`。
- `_meta.adapter_family` 必须是现有 6 个 family 之一。

### 3. 配置 RPC Methods

`single` 是用户选择 single 模式时默认测试的 method。

`mixed` 是 method 列表，用于兼容旧逻辑和可读性。

`mixed_weighted` 是当前 mixed 模式优先使用的权重配置。

示例：

```json
{
  "rpc_methods": {
    "single": "eth_getBalance",
    "mixed": "eth_getBalance,eth_blockNumber,eth_getBlockByNumber,eth_call",
    "mixed_weighted": [
      {"method": "eth_getBalance", "weight": 40},
      {"method": "eth_blockNumber", "weight": 30},
      {"method": "eth_getBlockByNumber", "weight": 20},
      {"method": "eth_call", "weight": 10}
    ]
  }
}
```

建议权重总和写成 100，便于审计。框架会按比例生成 vegeta targets。

### Mixed Workload 真实度

默认模板优先保证框架回归测试稳定，不代表所有生产环境的真实流量模型。如果希望
mixed 模式更接近真实 end-user 请求，需要按自己的链和业务调整
`mixed_weighted`。

建议 mixed 中包含不同请求形态：

| 流量类型 | 示例 | 作用 |
|---|---|---|
| 链状态 / tip | `eth_blockNumber`、`getBlockHeight`、`system_chain` | 低成本基线流量和类似健康检查的读请求。 |
| 账户 / 余额 / 对象读取 | `eth_getBalance`、`getAccountInfo`、REST account paths | 钱包、浏览器、dApp 常见读取。 |
| 交易 / receipt 查询 | `eth_getTransactionReceipt`、`getrawtransaction`、transaction-by-hash paths | 钱包、浏览器、用户支持常见查询。 |
| 区块读取 | `eth_getBlockByNumber`、`getblock`、block-by-height paths | 响应更大，通常涉及更多存储读取。 |
| 合约 / view call | `eth_call`、`POST /v1/view`、`runGetMethod` | 比纯状态查询更接近 dApp 生产流量。 |
| logs / events / indexer 查询 | `eth_getLogs`、account transaction paths、validator/pool queries | 通常更贵，更接近 explorer/analytics 流量。 |

如果 mixed workload 大部分都是 `no_params` method，它适合测试框架链路，
但不适合代表真实生产业务。真实 profile 至少应该在链支持的前提下覆盖一些
address、transaction、block、contract/view 类型 method。

调整 `mixed_weighted` 时按下面顺序做：

1. 在 `rpc_methods.mixed_weighted` 增加或替换 method。
2. 增加对应的 `param_formats.<method>`。
3. 在 `params` 中配置真实样本，使用 `${TARGET_*:-measured-default}`。
4. REST/path method 需要增加 `_meta.rest_paths.<method>`。
5. 如果 method 需要显式位置参数、对象字段、query 参数或 body 字段，增加 `param_spec.<method>`。
6. 使用 `python3 tools/chain_adapters/cli.py validate-template --chain <chain>` 验证请求构造。
7. 在 `tools/fake-node/configs/` 对应 family YAML 中增加 fixture 映射。
8. 录制该 method 自己的 fixture，并运行 coverage/runtime probes。

`weight` 必须是正整数。target generator 会把缺失、非法或为 0 的权重视为 `1`，
确保已配置的 method 在 mixed-mode smoke 和 benchmark 中仍会被覆盖。如果要禁用某个
method，需要从 `mixed_weighted` 中移除，不要把 `weight` 设置为 `0`。

不要因为某个 method 出现在官方文档中就直接加入 mixed。只有当框架能构造合法
请求、录制真实响应、通过 fake-node 回放，并进入 proxy/HTML 归因链路时，才应
加入 mixed workload。

自定义 RPC 支持在验证完成前应保持为 job-local。Agent 可以为某次 benchmark 起草
workload override，但不能因为用户要求添加 method 就直接修改 `config/chains/*.json`。
只有 endpoint validation、fixture 录制、fake-node coverage、smoke 和文档更新都通过后，
才应该把 method 提升到 canonical chain template。

对于已有链的自定义 method，仅有 method 名称不够。需要收集并验证：

1. 链或 API 产品的可访问 endpoint。
2. 精确 method 或 route。
3. 完整 request 结构，包括 params、body、path variables、query values，以及必要 header。
4. 每个 `TARGET_*` placeholder 的真实样本值。
5. 成功 response，必要时还包括有代表性的错误 response。
6. single/mixed workload 参与方式和 mixed 权重。
7. 能回放验证响应的 fake-node fixture。

如果用户希望某个 method 权重为 `0`，应从 `mixed_weighted` 移除该 method。当前
target generator 会归一化 0 权重，已配置 method 仍可能被执行。

### 4. 配置参数格式

每个会进入 `single` 或 `mixed_weighted` 的 method 都必须能被所选 adapter 构造。
常见 JSON-RPC method 使用 `param_formats`。

示例：

```json
{
  "param_formats": {
    "eth_getBalance": "address_latest",
    "eth_blockNumber": "no_params",
    "eth_getBlockByNumber": "block_number",
    "eth_call": "eth_call_object_latest"
  }
}
```

简单 method 优先使用 `param_formats`，这样模板更紧凑。如果 method 需要显式位置参数、
对象字段、REST path 绑定、query 参数或 body 模板，可以使用可选的 `param_spec`：

```json
{
  "param_spec": {
    "eth_getBalance": {
      "transport": "jsonrpc_list",
      "params": [
        {"source": "address"},
        {"literal": "latest"}
      ]
    },
    "GET_ACCOUNT_TXS": {
      "transport": "rest_query",
      "bindings": {
        "address": {"source": "address"}
      },
      "query": {
        "limit": {"literal": 10}
      }
    }
  }
}
```

支持的 transport 包括 `jsonrpc_list`、`jsonrpc_dict`、`rest_path`、
`rest_query`、`rest_body`。编辑模板后运行：

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
```

如果该 method 仍无法用 `param_formats`、`_meta.rest_paths` 或 `param_spec`
表达，再扩展 `tools/chain_adapters/<family>.py`。

#### 完整示例：3 个参数的 JSON-RPC Method

`eth_getStorageAt(address, storageSlot, blockTag)` 需要 3 个有顺序的参数。
做法是：把 method 加入 workload，配置真实样本 slot，定义 `param_spec`，
并确保 proxy extraction 和 fake-node fixture 使用同一个 method 名称。

Chain template：

```json
{
  "params": {
    "target_address": "${TARGET_ADDRESS:-0x0000000000000000000000000000000000000000}",
    "target_storage_slot": "${TARGET_STORAGE_SLOT:-0x0}"
  },
  "rpc_methods": {
    "single": "eth_getBalance",
    "mixed_weighted": [
      {"method": "eth_getBalance", "weight": 40},
      {"method": "eth_blockNumber", "weight": 30},
      {"method": "eth_getStorageAt", "weight": 30}
    ]
  },
  "param_spec": {
    "eth_getStorageAt": {
      "transport": "jsonrpc_list",
      "params": [
        {"source": "address"},
        {"source": "target_storage_slot"},
        {"literal": "latest"}
      ]
    }
  },
  "proxy_extraction": {
    "protocol": "jsonrpc",
    "method_path": "method",
    "fallback_method": "unknown"
  }
}
```

fake-node family YAML：

```yaml
methods:
  eth_getStorageAt:
    fixture: eth_getStorageAt.json
    tier: expensive
```

验证：

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/check_fixture_coverage.py
```

这些步骤通过后，`eth_getStorageAt` 会参与 mixed weighted target 生成，
并且会用同一个 method name 出现在 per-method 报告图表中。

### 5. 配置真实样本参数

`params` 中要放真实可查的样本值。常见字段：

```json
{
  "params": {
    "target_address": "0x0000000000000000000000000000000000000000",
    "target_tx_hash": "0x...",
    "target_height": 123456,
    "target_block_hash": "0x..."
  }
}
```

注意：

- `tx_hash` 必须是该链真实存在且 endpoint 可查询的交易。
- `address` 最好是真实存在、有余额或有活动记录的地址。
- REST 链经常需要 `asset_id`、`token_id`、`validator_address`、`denom` 等链特定样本。

如果样本不真实，fixture 录制可能得到 400/404/空对象，fake-node 就无法模拟真实业务。

新增链和自定义 method 都必须做 endpoint validation。用户提供的 request/response 样本
只是有用证据，不是事实本身；只有框架 probe 用户提供的 endpoint，并观察到同样行为后，
才可以认为样本可信。如果 endpoint 属于某个节点平台的其他 API 产品，不要假设它与该链
常见 RPC family 兼容。例如 EVM endpoint 可以走 JSON-RPC 路径，但 provider 的 billing
或 management REST API 可能需要 REST adapter，甚至需要新增 adapter family。

### 6. 配置 REST Path 或 Sidecar Path

REST family 或带 sidecar 的 family 需要 `_meta.rest_paths`。

示例：

```json
{
  "_meta": {
    "adapter_family": "rest",
    "rest_paths": {
      "GET /v1/accounts/{addr}/transactions": {
        "method": "GET",
        "path": "/v1/accounts/{address}/transactions"
      }
    }
  }
}
```

逻辑 method 名可以和 HTTP path 相同，也可以是更易读的 key，但必须能被 adapter 和 fake-node fixture 映射一致。

### 7. 配置 Sync Health

`_meta.sync_health` 决定 block height monitor 如何判断节点健康。

支持模式：

- `absolute_gap`：本地和 target/mainnet 都返回数值高度，比较 `target - local`。
- `conditional_gap`：例如 EVM `eth_syncing`，同步中返回对象，不同步返回 `false`。
- `reported_lag`：本地节点直接返回 lag，例如 Solana `getHealth`。
- `freshness_only`：没有可靠主网高度，只能看 probe 是否成功或本地进度是否新鲜。
- `health_only`：只有 boolean 或粗粒度健康状态。

示例：

```json
{
  "_meta": {
    "sync_health": {
      "mode": "conditional_gap",
      "local_probe": "adapter.sync_status_request(local_rpc_url)",
      "comparison": "local_reported_gap",
      "threshold_env": "BLOCK_HEIGHT_DIFF_THRESHOLD",
      "time_threshold_env": "BLOCK_HEIGHT_TIME_THRESHOLD",
      "threshold_unit": "block",
      "notes": "eth_syncing returns false when node is not syncing; otherwise highestBlock-currentBlock is used."
    }
  }
}
```

优先复用现有阈值：

- `BLOCK_HEIGHT_DIFF_THRESHOLD`
- `BLOCK_HEIGHT_TIME_THRESHOLD`

只有新链暴露了无法映射到现有 diff/time 语义的新单位时，才新增配置变量。

## 录制 Fixture

录制命令：

```bash
tools/fake-node/record_rpc_fixtures.sh <chain>
```

产物位置：

```text
tools/fake-node/fixtures/<chain>/*.json
docs/audit/rpc-fixtures/<chain>/<method>/{request.json,response.json,meta.json}  # 本地审计证据，通常不提交
```

如果某个 method 需要真实 `tx_hash`、`address`、`height`、`asset_id`，必须先补 `params`。不要用 placeholder 录制。

## 验证

### Fixture 覆盖率

```bash
python3 tools/fake-node/check_fixture_coverage.py --json
```

每个已配置的 `single` 和 `mixed_weighted` method 都必须有已提交的 fake-node fixture。

如果本地录制了完整 request/response evidence，可以再执行真实性审计：

```bash
python3 tools/fake-node/validate_fixture_authenticity.py --json
python3 tools/fake-node/check_fixture_coverage.py --json --strict
```

### Runtime Probe

```bash
python3 tools/fake-node/runtime_probe.py
python3 tools/fake-node/runtime_probe_block_height.py
```

如果只验证某条链，可以先查看工具参数：

```bash
python3 tools/fake-node/runtime_probe.py --help
python3 tools/fake-node/runtime_probe_block_height.py --help
```

### Sync Health Registry

```bash
python3 tools/audit_sync_health_registry.py --write --json
```

新增链必须出现在 registry 审计结果中，且不能有 errors。

## 本地闭环测试

推荐在 Docker/Linux 环境执行，不考虑 macOS 兼容。

准备少量 active accounts：

```bash
mkdir -p //blockchain-node-benchmark-result/current/tmp
cat > //blockchain-node-benchmark-result/current/tmp/active_accounts.txt <<EOF
<address-1>
<address-2>
<address-3>
EOF
```

运行 fake-node + proxy + mixed quick smoke：

```bash
export BLOCKCHAIN_NODE=<chain>
export RPC_MODE=mixed
export QUICK_INITIAL_QPS=1
export QUICK_MAX_QPS=1
export QUICK_QPS_STEP=1
export QUICK_DURATION=3
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0
export BLOCK_HEIGHT_MONITOR_RATE=1

./blockchain_node_benchmark.sh \
  --quick \
  --mixed \
  --fake-node \
  --initial-qps 1 \
  --max-qps 1 \
  --step-qps 1 \
  --duration 3
```

检查结果：

```bash
ls -lh //blockchain-node-benchmark-result/archives/
```

重点看：

- vegeta 请求是否成功。
- `logs/proxy_method.csv` 是否有业务 method，而不是只有 health probe。
- `logs/block_height_monitor_*.csv` 是否有 `sync_mode` / `sync_status`。
- `reports/performance_report_zh_*.html` 是否生成。
- per-method charts 是否生成。

## 新增 RPC Method

如果只是给现有链增加 method：

1. 在 `config/chains/<chain>.json` 增加 `param_formats.<method>`；如果请求结构无法用内置格式清晰表达，增加 `param_spec.<method>`。
2. 如果 method 要参与 mixed，加入 `rpc_methods.mixed` 和 `mixed_weighted`。
3. REST method 还要增加 `_meta.rest_paths.<method>`。
4. 使用 `python3 tools/chain_adapters/cli.py validate-template --chain <chain>` 验证请求构造。
5. 在 fake-node family YAML 增加 method 到 fixture 文件名的映射。
6. 准备真实样本参数。
7. 重新录制 fixture。
8. 跑真实性、覆盖率和 runtime probe。

JSON-RPC 示例：

```json
{
  "param_formats": {
    "eth_getTransactionReceipt": "transaction_hash"
  },
  "rpc_methods": {
    "mixed_weighted": [
      {"method": "eth_getTransactionReceipt", "weight": 10}
    ]
  },
  "params": {
    "target_tx_hash": "0x..."
  }
}
```

fake-node YAML 示例：

```yaml
methods:
  eth_getTransactionReceipt:
    fixture: eth_getTransactionReceipt.json
    tier: expensive
```

## 何时需要改代码

新增现有 family 的链，理想情况下不需要改 Python/Go。

以下情况才需要改代码：

- 新请求 envelope 不是现有 JSON-RPC 或 REST/sidecar 可表达。
- 新认证/header 规则无法用现有配置表达。
- 参数需要链特有编码，现有 `param_formats` 或 `param_spec` 都不支持。
- 返回高度或健康状态的结构无法被现有 sync health parser 解析。
- fake-node 现有 family handler 无法路由或回放该请求。
- 新 chain 内按 method 路由到多个 endpoint，而现有 family 不支持这种 routing。

如果要新增 family，需要同步增加：

- `tools/chain_adapters/<family>.py`
- `tools/fake-node/handlers/<family>.go`
- `tools/fake-node/configs/<family>.yaml`
- chain template `_meta.adapter_family`
- proxy extraction DSL
- fixture 录制与覆盖率测试
- 文档和审计记录

## 常见错误

### 错误 1：复用相同参数名的响应

不允许。`address` 或 `tx_hash` 相同不代表响应结构相同。必须录制 `chain + method` 自己的 fixture。

### 错误 2：`mixed_weighted` 里有 method，但 adapter 无法构造

target generator 无法构造请求。每个 method 都必须按 family 和请求结构由
`param_formats`、`_meta.rest_paths` 或 `param_spec` 覆盖。

### 错误 3：REST method 只写了 method 名，没有 `_meta.rest_paths`

REST adapter 不知道真实 HTTP path/body，会导致请求生成失败或 404。

### 错误 4：fixture 是 403/429/404 也当成功

不允许。换 public endpoint 或补真实参数后重新录制。

### 错误 5：sync health 没有配置

block height monitor 无法判断节点健康，bottleneck detector 和报告链路也会缺失健康信号。

### 错误 6：只跑 coverage，不跑 authenticity

coverage 只能说明“有文件”，authenticity 才能说明“不是占位符或错误响应”。两者都要跑。

## 最小验收清单

新增链完成前，至少满足：

- [ ] `config/chains/<chain>.json` 存在且 `_meta.adapter_family` 正确。
- [ ] `single` 和 `mixed_weighted` 中所有 method 都能由 `param_formats`、`_meta.rest_paths` 或 `param_spec` 构造。
- [ ] REST/sidecar method 都有 `_meta.rest_paths`。
- [ ] `python3 tools/chain_adapters/cli.py validate-template --chain <chain>` 通过。
- [ ] `_meta.sync_health` 已配置。
- [ ] `params` 中样本值真实可查。
- [ ] fake-node fixture 已录制到 `tools/fake-node/fixtures`。
- [ ] 可选的本地审计证据 `docs/audit/rpc-fixtures/<chain>/<method>` 不包含 placeholder。
- [ ] `tools/fake-node/check_fixture_coverage.py --json` 通过。
- [ ] 可选的本地真实性审计在录制 request/response evidence 后通过。
- [ ] `tools/fake-node/runtime_probe.py` 通过。
- [ ] `tools/fake-node/runtime_probe_block_height.py` 或 sync health registry 审计通过。
- [ ] fake-node quick smoke 能生成报告和 proxy method CSV。
