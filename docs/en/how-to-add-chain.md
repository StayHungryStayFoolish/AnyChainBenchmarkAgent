# How to Add a Blockchain or RPC Method

[中文](../zh/how-to-add-chain.md) | [English](how-to-add-chain.md)

This is the operational guide for extending the current framework. Chain template fields are documented through the examples in `config/chains/*.json` and the steps below. For fake-node details, see the [fake-node README](../../tools/fake-node/README.md).

## Core Principle

When adding a chain, separate two concerns:

- **Request-building schema can be reused**: if the new chain belongs to an existing family, it can reuse the adapter, proxy extractor, target generator, and fake-node handler.
- **Response fixtures cannot be reused by default**: even if two methods both accept `address` or `tx_hash`, each `chain + method` must have its own real recorded response.

fake-node matches responses at this granularity:

```text
chain + rpc method + family handler + fixture
```

So after adding a chain or method, record real request/response fixtures. Do not reuse an older fixture only because the parameter name looks similar.

## How to Choose One of the 6 Families

Family is not based on brand, token, or ecosystem. It is based on the actual RPC request and parsing shape.

| Family | When to use it | Examples |
|---|---|---|
| `jsonrpc` | Standard JSON-RPC POST with `method` in the request body | Ethereum, Solana, Sui, Tron, Avalanche C |
| `bitcoin_jsonrpc` | Bitcoin Core / UTXO-style JSON-RPC, possibly with Basic Auth or REST workarounds | Bitcoin, Litecoin, Dogecoin, BCH |
| `rest` | HTTP path/body is the primary interface; logical methods map through `_meta.rest_paths` | Aptos, Algorand, Cardano, Tezos, TON |
| `substrate` | Polkadot SDK / Substrate RPC, such as `chain_*`, `state_*`, `system_*` | Polkadot, Kusama, Acala |
| `tendermint` | Cosmos SDK / Tendermint / CometBFT REST-RPC | Cosmos Hub, Osmosis, Celestia |
| `hedera_dual` | Hedera Mirror REST plus Hashio JSON-RPC Relay | Hedera |

If the request envelope, parameter structure, endpoint routing, authentication/header behavior, response envelope, and block-height parsing fit one of these families, this is usually a configuration-only extension.

If the existing families cannot express those behaviors, extend or add an adapter and fake-node handler.

## Add a Chain in an Existing Family

### 1. Create the Chain Template

Add:

```text
config/chains/<chain>.json
```

Start by copying a similar chain in the same family:

- New EVM chain: copy `config/chains/ethereum.json`, `arbitrum.json`, or `base.json`
- New Cosmos chain: copy `config/chains/cosmos-hub.json`
- New Substrate chain: copy `config/chains/polkadot.json`
- New REST chain: copy `config/chains/aptos.json` or `algorand.json`

### 2. Configure the Base Fields

Confirm these fields:

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

Rules:

- `chain_type` should match the file name.
- `rpc_url` should usually remain `LOCAL_RPC_URL` in the production framework.
- `_meta.adapter_family` must be one of the existing 6 families.

### 3. Configure RPC Methods

`single` is the default method when the user chooses single mode.

`mixed` is a method list used for compatibility and readability.

`mixed_weighted` is the current preferred source for mixed-mode traffic generation.

Example:

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

Enabled weights must be positive integers totaling exactly 100. Chain-template
and runtime-workload validation rejects any other total before execution. The
framework generates Vegeta targets proportionally from the validated weights.

### Mixed Workload Realism

The default templates are designed to be stable for framework regression tests.
They are not a universal production workload model. If you want mixed mode to
simulate real end-user traffic, tune `mixed_weighted` for your chain and
application.

Prefer a mix of method shapes:

| Traffic type | Examples | Why it matters |
|---|---|---|
| Chain status / tip | `eth_blockNumber`, `getBlockHeight`, `system_chain` | Low-cost baseline traffic and health-like reads. |
| Account / balance / object reads | `eth_getBalance`, `getAccountInfo`, REST account paths | Common wallet, explorer, and dApp reads. |
| Transaction / receipt reads | `eth_getTransactionReceipt`, `getrawtransaction`, transaction-by-hash paths | Common user support, wallet, and explorer lookups. |
| Block reads | `eth_getBlockByNumber`, `getblock`, block-by-height paths | Higher response volume and storage access. |
| Contract / view calls | `eth_call`, `POST /v1/view`, `runGetMethod` | Closer to dApp production behavior than status-only traffic. |
| Logs / events / indexer queries | `eth_getLogs`, account transaction paths, validator/pool queries | Often more expensive and closer to analytics/explorer traffic. |

Avoid a mixed workload made mostly of `no_params` methods unless you are only
testing the framework plumbing. A realistic production profile should include
some address, transaction, block, and contract/view methods when the chain
supports them.

When changing `mixed_weighted`:

1. Add or select the method in `rpc_methods.mixed_weighted`.
2. Add the matching `param_formats.<method>`.
3. Add real sample values in `params` using `${TARGET_*:-measured-default}`.
4. For REST/path methods, add `_meta.rest_paths.<method>`.
5. If the method needs explicit positional params, object fields, query values,
   or body fields, add `param_spec.<method>`.
6. Validate request construction with `python3 tools/chain_adapters/cli.py validate-template --chain <chain>`.
7. Add the fake-node family YAML mapping under `tools/fake-node/configs/`.
8. Record the method's own fixture and run coverage/runtime probes.

`weight` must be a positive integer, and all enabled weights must total exactly
100. The target generator still normalizes a missing, invalid, or zero legacy
value to `1` defensively, but that is not an acceptance path: template/workload
validation must reject the configuration before execution. To disable a method,
remove it from `mixed_weighted`; do not set `weight` to `0`.

Do not add a method only because it is listed in official docs. Add it when the
framework can build a valid request, record a real response, replay it through
fake-node, and include it in the proxy/HTML attribution path.

Custom RPC support is job-local until validation is complete. The Agent may
draft a workload override for a single benchmark job, but it must not mutate
`config/chains/*.json` just because a user asked for a custom method. Promote a
method into the canonical chain template only after endpoint validation,
fixture recording, fake-node coverage, smoke, and documentation updates all
pass.

For existing-chain custom methods, a method name is not enough. Collect and
validate:

1. A reachable endpoint for the chain or API product.
2. The exact method or route.
3. The full request shape, including params, body, path variables, query
   values, and required headers when applicable.
4. Real sample values for every `TARGET_*` placeholder.
5. A successful response and, when useful, a representative error response.
6. The desired single/mixed workload participation and mixed weight.
7. A fake-node fixture that replays the verified response.

If the user wants a method weight of `0`, remove that method from
`mixed_weighted` instead. A configured zero weight is normalized by the target
generator and the method can still be exercised.

### 4. Configure Parameter Formats

Every method used by `single` or `mixed_weighted` must be buildable by the
selected adapter. For simple JSON-RPC methods, use a `param_formats` entry.

Example:

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

For simple methods, prefer `param_formats` because it keeps templates compact.
For methods that need explicit positional params, object fields, REST path
bindings, query parameters, or body templates, use optional `param_spec`:

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

Supported transports are `jsonrpc_list`, `jsonrpc_dict`, `rest_path`,
`rest_query`, and `rest_body`. After editing the template, run:

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
```

If the method is still not expressible by `param_formats`, `_meta.rest_paths`,
or `param_spec`, extend `tools/chain_adapters/<family>.py`.

#### Complete Example: Three-Argument JSON-RPC Method

`eth_getStorageAt(address, storageSlot, blockTag)` needs three ordered
arguments. Add the method to the workload, provide a real sample slot, define
`param_spec`, then make sure proxy extraction and fake-node fixtures use the
same method name.

Chain template:

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

fake-node family YAML:

```yaml
methods:
  eth_getStorageAt:
    fixture: eth_getStorageAt.json
    tier: expensive
```

Validation:

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/check_fixture_coverage.py
```

After these pass, `eth_getStorageAt` participates in mixed weighted target
generation and appears in the per-method report charts by the same method name.

### 5. Configure Real Sample Parameters

Put real queryable sample values in `params`. Common examples:

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

Notes:

- `tx_hash` must exist on that chain and be queryable from the endpoint.
- `address` should preferably exist and have balance or activity.
- REST chains often require chain-specific samples such as `asset_id`, `token_id`, `validator_address`, or `denom`.

If the sample is not real, fixture recording may produce 400/404/empty responses, and fake-node will not simulate production behavior.

Endpoint validation is mandatory for new chains and custom methods. User
request/response samples are useful evidence, but they are not trusted as truth
until the framework probes the supplied endpoint and observes the same behavior.
If the endpoint belongs to a platform API with different protocol semantics,
do not assume it is compatible with the chain's usual RPC family. For example,
an EVM endpoint can follow the JSON-RPC path, while a provider's billing or
management REST API may need the REST adapter or a new adapter family.

### 6. Configure REST Paths or Sidecar Paths

REST family, or families using sidecar paths, need `_meta.rest_paths`.

Example:

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

The logical method name may be identical to the HTTP path, or it may be a more readable key. It must still map consistently through the adapter and fake-node fixture.

### 7. Configure Sync Health

`_meta.sync_health` tells the block height monitor how to judge node health.

Supported modes:

- `absolute_gap`: local and target/mainnet endpoints return numeric heights; compare `target - local`.
- `conditional_gap`: for example EVM `eth_syncing`, where syncing returns an object and not syncing returns `false`.
- `reported_lag`: the local node reports its own lag, such as Solana `getHealth`.
- `freshness_only`: no reliable mainnet height is available; use probe success or local progress freshness.
- `health_only`: only a boolean or coarse health result is available.

Example:

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

Prefer the existing thresholds:

- `BLOCK_HEIGHT_DIFF_THRESHOLD`
- `BLOCK_HEIGHT_TIME_THRESHOLD`

Add a new config variable only when the chain exposes a genuinely different unit that cannot be mapped to the existing diff/time model.

## Record Fixtures

Run:

```bash
tools/fake-node/record_rpc_fixtures.sh <chain>
```

Outputs:

```text
tools/fake-node/fixtures/<chain>/*.json
docs/audit/rpc-fixtures/<chain>/<method>/{request.json,response.json,meta.json}  # local audit evidence, not normally committed
```

If a method requires a real `tx_hash`, `address`, `height`, or `asset_id`, add it to `params` before recording. Do not record placeholders.

## Verify

### Fixture Coverage

```bash
python3 tools/fake-node/check_fixture_coverage.py --json
```

Every configured `single` and `mixed_weighted` method must be covered by a committed fake-node fixture.

For local authenticity checks after recording request/response evidence, run:

```bash
python3 tools/fake-node/validate_fixture_authenticity.py --json
python3 tools/fake-node/check_fixture_coverage.py --json --strict
```

### Runtime Probe

```bash
python3 tools/fake-node/runtime_probe.py
python3 tools/fake-node/runtime_probe_block_height.py
```

To check whether a tool supports single-chain filtering:

```bash
python3 tools/fake-node/runtime_probe.py --help
python3 tools/fake-node/runtime_probe_block_height.py --help
```

### Sync Health Registry

```bash
python3 tools/audit_sync_health_registry.py --write --json
```

The new chain must appear in the registry audit with no errors.

## Local End-to-End Smoke Test

Run this in Docker/Linux. The framework does not target macOS compatibility.

Prepare a small account file:

```bash
mkdir -p //blockchain-node-benchmark-result/current/tmp
cat > //blockchain-node-benchmark-result/current/tmp/active_accounts.txt <<EOF
<address-1>
<address-2>
<address-3>
EOF
```

Run fake-node + proxy + mixed quick smoke:

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

Check:

```bash
ls -lh //blockchain-node-benchmark-result/archives/
```

Look for:

- vegeta requests succeeded.
- `logs/proxy_method.csv` contains workload methods, not only health probes.
- `logs/block_height_monitor_*.csv` contains `sync_mode` / `sync_status`.
- `reports/performance_report_en_*.html` or `reports/performance_report_zh_*.html` was generated.
- per-method charts were generated.

## Add an RPC Method

If you are only adding a method to an existing chain:

1. Add `param_formats.<method>` to `config/chains/<chain>.json`, or add `param_spec.<method>` if the request shape is too specific for a built-in format.
2. If the method participates in mixed mode, add it to `rpc_methods.mixed` and `mixed_weighted`.
3. If it is a REST method, add `_meta.rest_paths.<method>`.
4. Validate request construction with `python3 tools/chain_adapters/cli.py validate-template --chain <chain>`.
5. Add the method-to-fixture mapping in the fake-node family YAML.
6. Prepare real sample parameters.
7. Record fixtures again.
8. Run authenticity, coverage, and runtime probes.

JSON-RPC example:

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

fake-node YAML example:

```yaml
methods:
  eth_getTransactionReceipt:
    fixture: eth_getTransactionReceipt.json
    tier: expensive
```

## When Code Changes Are Required

Adding a chain in an existing family ideally requires no Python or Go changes.

Code changes are required when:

- The new request envelope is not expressible as existing JSON-RPC or REST/sidecar behavior.
- Authentication or header rules cannot be expressed by current configuration.
- Parameters need chain-specific encoding not supported by existing `param_formats` or `param_spec`.
- Block height or health status responses cannot be parsed by existing sync health logic.
- The current fake-node family handler cannot route or replay the request.
- A chain routes methods to multiple endpoints in a way that the existing family cannot express.

If you add a new family, update all of these:

- `tools/chain_adapters/<family>.py`
- `tools/fake-node/handlers/<family>.go`
- `tools/fake-node/configs/<family>.yaml`
- chain template `_meta.adapter_family`
- proxy extraction DSL
- fixture recording and coverage tests
- docs and audit records

## Common Mistakes

### Mistake 1: Reusing a Response Because the Parameter Name Matches

Do not do this. `address` or `tx_hash` being the same does not mean the response structure is the same. Record the fixture for that exact `chain + method`.

### Mistake 2: `mixed_weighted` Has a Method the Adapter Cannot Build

The target generator cannot build the request. Every method must be covered by
`param_formats`, `_meta.rest_paths`, or `param_spec`, depending on its family
and request shape.

### Mistake 3: A REST Method Has No `_meta.rest_paths`

The REST adapter will not know the real HTTP path/body, which can cause request generation failures or 404s.

### Mistake 4: Treating 403/429/404 Fixtures as Success

Do not accept those fixtures. Use another public endpoint or fix the sample params and record again.

### Mistake 5: Missing Sync Health

The block height monitor cannot judge node health, and bottleneck detection/reporting will lose the health signal.

### Mistake 6: Running Coverage Without Authenticity

Coverage only says files exist. Authenticity says they are not placeholders or error responses. Run both.

## Minimum Acceptance Checklist

Before considering the chain complete:

- [ ] `config/chains/<chain>.json` exists and `_meta.adapter_family` is correct.
- [ ] Every method in `single` and `mixed_weighted` is buildable by `param_formats`, `_meta.rest_paths`, or `param_spec`.
- [ ] REST/sidecar methods have `_meta.rest_paths`.
- [ ] `python3 tools/chain_adapters/cli.py validate-template --chain <chain>` passes.
- [ ] `_meta.sync_health` is configured.
- [ ] `params` contains real queryable sample values.
- [ ] fake-node fixtures are recorded under `tools/fake-node/fixtures`.
- [ ] Optional local audit evidence under `docs/audit/rpc-fixtures/<chain>/<method>` has no placeholders.
- [ ] `tools/fake-node/check_fixture_coverage.py --json` passes.
- [ ] Optional local authenticity audit passes after recording request/response evidence.
- [ ] `tools/fake-node/runtime_probe.py` passes.
- [ ] `tools/fake-node/runtime_probe_block_height.py` or sync health registry audit passes.
- [ ] fake-node quick smoke generates a report and proxy method CSV.
