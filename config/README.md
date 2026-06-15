# Configuration Guide

This directory is split into one user-facing configuration file and several
advanced/default layers.

For a normal benchmark run, start with only:

```bash
config/user_config.sh
```

Do not edit `config/config_loader.sh` for normal usage. It loads all layers,
detects runtime paths, resolves cloud/provider details, validates the selected
chain template, and exports derived variables for child processes.

## Configuration Layers

| File or directory | Audience | Purpose |
| --- | --- | --- |
| `user_config.sh` | All users | Required benchmark inputs: chain name, local RPC endpoint, RPC mode, node process names, cloud/machine metadata, disk baselines, network bandwidth, QPS profile defaults. |
| `chains/*.json` | Chain integrators | Chain templates for the 36 supported blockchains. Add or modify these only when adding a chain or changing RPC method coverage. |
| `system_config.sh` | Advanced users | Deployment platform override, logging behavior, monitoring overhead process list, and generic runtime behavior. Defaults are suitable for most runs. |
| `provider_disk_config.sh` | Framework maintainers | Provider-specific post-processing for disk baseline values after `user_config.sh` is loaded. |
| `internal_config.sh` | Framework maintainers | Bottleneck thresholds, block-height health thresholds, and internal monitor defaults. Change only when you intentionally tune framework behavior. |
| `cloud_provider.sh` | Framework maintainers | Provider and NIC detection. It detects GCP, AWS IMDSv2, or other environments and loads provider contracts. |
| `providers/*.sh` | Framework maintainers | Provider-specific metric and disk baseline helpers. |
| `deployment_mode_detector.sh` | Framework maintainers | Detects VM, systemd, Docker, and Kubernetes runtime modes. |
| `runtime_paths.sh` | Framework maintainers | Runtime host path and cgroup path resolution for VM, Docker, and Kubernetes modes. |
| `csv_schema_registry.sh` | Framework maintainers | Central CSV field registry used by monitoring and report generation. |

## First-Run Required Settings

Edit these values in `config/user_config.sh` before a production benchmark:

| Variable | Required | Description |
| --- | --- | --- |
| `BLOCKCHAIN_NODE` | Yes | Chain template name. Must match `config/chains/<chain>.json`, for example `solana`, `ethereum`, `bitcoin`, or `cosmos-hub`. |
| `LOCAL_RPC_URL` | Yes | Local blockchain node RPC endpoint. In fake-node tests this may be rewritten by the framework. |
| `MAINNET_RPC_URL` | Optional | Reference/mainnet endpoint override for block-height health comparison. Leave empty to use the selected chain template's default public endpoint. |
| `RPC_MODE` | Yes | `single` or `mixed`. `single` uses one configured method; `mixed` uses the chain template's `rpc_methods.mixed_weighted` list. |
| `BLOCKCHAIN_PROCESS_NAMES` | Yes for monitoring | Process-name or command-line keywords used to attribute CPU, memory, and IO to the blockchain node. The deployment-mode detector also reuses these values as systemd unit-name prefixes when auto-detecting `vm_systemd`. Use `ps aux`, `systemctl status`, or your container runtime to confirm the real names. |
| `CLOUD_PROVIDER` | Yes for reports/baselines | `gcp`, `aws`, `azure`, or `other`. The framework is GCP-first but keeps AWS compatibility. |
| `CLOUD_REGION` / `CLOUD_ZONE` | Recommended | Region and zone shown in reports. `CLOUD_ZONE` can be empty outside GCP. |
| `MACHINE_TYPE` | Recommended | Instance type shown in reports, for example `c3-standard-22` or `m7i.4xlarge`. |
| `LEDGER_DEVICE` | Yes for disk charts | Data/ledger disk device name from `lsblk`, for example `sdb`. |
| `DATA_VOL_TYPE` | Yes for disk baselines | Data disk type, for example `hyperdisk-extreme`, `hyperdisk-balanced`, `pd-ssd`, `gp3`, `io2`, or `instance-store`. |
| `DATA_VOL_MAX_IOPS` | Yes for disk baselines | Provisioned data disk IOPS or equivalent baseline. |
| `DATA_VOL_MAX_THROUGHPUT` | Yes for disk baselines | Provisioned data disk throughput in MiB/s. |
| `ACCOUNTS_DEVICE` | Optional | Account/state disk device from `lsblk`, for example `sdc`. Leave empty when the node uses a single disk. |
| `ACCOUNTS_VOL_*` | Optional | Account/state disk type, size, IOPS, and throughput baselines. Used only when `ACCOUNTS_DEVICE` is set. |
| `NETWORK_MAX_BANDWIDTH_GBPS` | Yes for network charts | Instance network bandwidth in Gbps. |
| `NETWORK_INTERFACE` | Optional | Network interface name. Leave empty to auto-detect. |
| `OBSERVABILITY_STACK_ENABLED` | Optional | When `true`, the benchmark entry script starts the read-only Prometheus/Grafana stack before benchmark traffic. Defaults to `false`. |
| `OBSERVABILITY_STACK_AUTO_STOP` | Optional | When `true`, a stack started by the entry script is stopped during framework cleanup. Defaults to `true`. |

## Optional Chain Sample Overrides

Chain templates include measured sample values for addresses, transactions,
blocks, assets, and other method-specific parameters. They are written as
`${TARGET_*:-measured-default}` so the framework works out of the box, while
users can override samples in `config/user_config.sh` without editing 36 chain
JSON files.

Set these variables when the embedded sample is not available on your own node,
when a method needs a chain-specific real value, or when recording fresh
fake-node fixtures:

| Variable | Used for |
| --- | --- |
| `TARGET_ADDRESS` | Account, address, contract, or object sample used by address-based methods. |
| `TARGET_TX_HASH` | Transaction hash sample for chains/methods named around `tx_hash`. |
| `TARGET_TXID` | Transaction id sample for UTXO, Tron, Algorand, or REST-style methods. |
| `TARGET_BLOCK_HASH` | Block hash sample. |
| `TARGET_BLOCK` / `TARGET_HEIGHT` / `TARGET_ROUND` | Block identifier, height, or round sample. |
| `TARGET_ASSET_ID` / `TARGET_ASSET` | Asset id or asset tuple. `TARGET_ASSET` may be a JSON array string. |
| `TARGET_EPOCH` / `TARGET_VP` / `TARGET_POOL_ID` | Method-specific epoch, validator, or pool samples. |
| `TARGET_TOKEN_ACCOUNT` / `TARGET_TOKEN_MINT` | Token account/mint samples, currently useful for Solana-style methods. |
| `TARGET_CONTRACT_ADDRESS` | Contract address sample for contract-call style methods. |
| `TARGET_EVM_ADDRESS` | EVM-form address sample for dual-protocol chains. |
| `TARGET_SIGNER_ID` | Signer id sample for NEAR-style transaction lookup. |

Leave these variables empty for default smoke tests and fake-node closed-loop
tests unless you are intentionally replacing the measured sample set.

## Process Names And systemd Detection

`BLOCKCHAIN_PROCESS_NAMES` is the single user-facing list for blockchain node
identity. It is used by:

- Monitoring collectors for runtime process attribution via `pgrep -f`.
- `deployment_mode_detector.sh` as systemd unit-name prefixes for detecting
  whether the host is `vm_systemd` instead of `vm_bare`.

In most deployments these names overlap: `geth.service` runs `geth`,
`agave-validator.service` runs `agave-validator`, and custom units usually keep
the binary name in the unit prefix. If your systemd unit name is completely
different from the process name, either add that unit prefix to
`BLOCKCHAIN_PROCESS_NAMES` or set `DEPLOYMENT_MODE=vm_systemd` explicitly.

## Optional Chain Endpoint Overrides

Most users only need `LOCAL_RPC_URL`. Some chains have methods that require an
auxiliary API surface, such as an indexer, REST/LCD endpoint, Substrate Sidecar,
Hedera mirror node, or companion EVM RPC. The chain templates include public
defaults for these fields so the framework can run smoke tests, but production
benchmarks should usually point them at your own local or trusted endpoints.

Set these optional variables in `config/user_config.sh` when needed:

| Variable | Used for |
| --- | --- |
| `CHAIN_REST_URL` | REST/LCD/HTTP API override for REST-family or Tendermint-family chains. |
| `CHAIN_INDEXER_URL` | Indexer or explorer API override, for example Algorand indexer or Bitcoin-like REST indexers. |
| `CHAIN_SIDECAR_URL` | Substrate Sidecar override for Polkadot/Kusama-style REST sidecar methods. |
| `CHAIN_EVM_RPC_URL` | EVM companion RPC override for dual-protocol chains. |
| `CHAIN_JSON_RPC_URL` | JSON-RPC companion endpoint override, currently useful for Hedera-style dual routing. |
| `CHAIN_MIRROR_URL` | Mirror-node endpoint override, currently useful for Hedera mirror REST. |
| `RPC_API_KEY` | Optional API key for public gateways or indexers that require `x-api-key`. |

If these variables are empty, adapters use the selected chain template defaults.
For fake-node closed-loop tests, leave them empty so all traffic stays on the
fake-node/proxy path.

## Benchmark Defaults

`user_config.sh` also contains default QPS profiles:

- `QUICK_*`: short sanity runs.
- `STANDARD_*`: normal benchmark runs.
- `INTENSIVE_*`: bottleneck discovery runs.

These values can be tuned, but they are not required for a first run.

## Advanced Settings

The following files are intentionally not part of the first-run path:

- Use `system_config.sh` only when you need to override deployment detection,
  logging, or monitoring overhead process attribution.
- Use `provider_disk_config.sh` only when changing provider-specific disk
  baseline derivation logic.
- Use `internal_config.sh` only when you intentionally tune bottleneck or
  block-height health semantics.
- Use `cloud_provider.sh` and `providers/*.sh` only when changing provider
  detection or provider-specific metric contracts.

## Chain Template Consistency

All 36 current chain templates share the same top-level schema:

- `chain_type`
- `rpc_url`
- `params`
- `system_addresses`
- `rpc_methods`
- `param_formats`
- `_meta`
- `proxy_extraction`

All current templates also define:

- `rpc_methods.single`
- `rpc_methods.mixed`
- `rpc_methods.mixed_weighted`

Do not add a top-level `methods` field, and do not add
`_meta.account_discovery` aliases. The 36-chain path is template driven:
`tools/fetch_active_accounts.py` writes deterministic target seed files from
`params.target_address` and optional `TARGET_ADDRESS` overrides.

The current 36-chain benchmark path builds Vegeta targets from:

- `rpc_methods` to select the workload methods.
- `param_formats` to build JSON-RPC parameters.
- `_meta.rest_paths` to build REST, sidecar, indexer, and path-style methods.
- `_meta.adapter_family` to select the adapter implementation.
- `params.target_address` as the default seed consumed by target generation.

The templates are grouped by `_meta.adapter_family`:

| Family | Count | Purpose |
| --- | ---: | --- |
| `jsonrpc` | 16 | JSON-RPC chains such as EVM chains, Solana, NEAR, Sui, Starknet, and Tron. |
| `rest` | 5 | REST-oriented chains such as Algorand, Aptos, Cardano, Tezos, and TON. |
| `bitcoin_jsonrpc` | 4 | Bitcoin-like JSON-RPC chains with `getblockchaininfo` style sync health. |
| `substrate` | 5 | Substrate chains using methods such as `chain_getHeader` and `system_syncState`. |
| `tendermint` | 5 | Tendermint/Cosmos-style chains using REST/RPC status and block endpoints. |
| `hedera_dual` | 1 | Hedera, which uses both mirror REST and JSON-RPC style endpoints. |

Use `config/chain_template.json.bak` as the starting point when adding a new
chain template.
