# Tools Inventory

This directory contains runtime helpers, local test tools, fake-node tooling,
and maintenance scripts. Files here should fit one of the categories below.

## Commit to Git

Runtime and production-path helpers:

- `target_generator.sh`: builds Vegeta targets from chain templates and selected RPC mode.
- `fetch_active_accounts.py`: fetches active addresses or account-like inputs for target generation.
- `chain_adapters/`: production request-building and sync-health adapters for the 6 RPC families.
- `proxy/`: per-method RPC proxy source code and tests. Commit source, `go.mod`, and tests; do not commit the built `proxy` binary.
- `benchmark_archiver.sh`: archives benchmark outputs.
- `disk_analyzer.sh`: offline disk analysis invoked after benchmark runs.
- `disk_bottleneck_detector.sh`: real-time disk bottleneck detector used by the coordinator.

Fake-node closed-loop test assets:

- `fake-node/fake_node.go`, `fake-node/handlers/`, `fake-node/configs/`: fake-node runtime source and family mappings.
- `fake-node/scripts/ci_smoke.sh`: fake-node smoke test runner.
- `fake-node/fixtures/<chain>/*.json`: deterministic runtime fixtures used by fake-node. These should be committed after they are sanitized and coverage checks pass.
- `fake-node/check_fixture_coverage.py`: verifies configured workload methods have fake-node fixtures.
- `fake-node/runtime_probe.py`: starts fake-node and verifies adapter-generated requests against it.
- `fake-node/runtime_probe_block_height.py`: verifies sync-health and block-height paths against fake-node fixtures.
- `fake-node/record_rpc_fixtures.sh`: central human-facing entrypoint for recording all chains or a selected chain list.
- `fake-node/record_rpc_fixtures.py`: records real RPC request/response fixtures.
- `fake-node/validate_fixture_authenticity.py`: validates local recording evidence before trusting fixture updates.

Monitoring and public-repo quality gates:

- `audit_monitoring_lifecycle.sh`: verifies monitor lifecycle cleanup and generated files.
- `audit_monitoring_runtime.sh`: verifies runtime CSV/header/data contracts.
- `audit_sync_health_registry.py`: audits chain sync-health registry coverage.
- `fake-node/audit_health_probe_fixtures.py`: audits health-probe fixture coverage.
- `framework_data_quality_checker.sh`: validates generated monitoring data shape.
- `check_public_repo_markers.py`: blocks process-artifact markers and non-English runtime output.
- `check_agent_boundaries.py`: blocks Agent regressions back to terminal keyword routing,
  legacy Agent files, or natural-language sanitizer patches.

Local smoke and workload tools:

- `legacy_mock_rpc_e2e_smoke.sh`: legacy local smoke harness for the old mock RPC path.
- `legacy_mock_rpc_server.py`: lightweight legacy mock server used by compatibility tests.
- `single_disk_workload_profile.sh`: synthetic disk workload helper for local monitor tests.

Maintenance scripts that can stay if documented:

- `fill_mixed_weighted.py`: one-shot or maintenance helper for chain template `mixed_weighted`.
- `fill_proxy_extraction.py`: one-shot or maintenance helper for chain template `proxy_extraction`.
- `normalize_chain_templates.py`: chain template normalization helper referenced by current templates.

These scripts are not part of the normal benchmark runtime. Keep them because
they document repeatable migrations and can be rerun on future chain-template
updates.

## Current 36-Chain Validation Entry Points

Use these tools for the current framework:

- `fake-node/check_fixture_coverage.py`: fixture file coverage for configured workload RPC methods.
- `fake-node/runtime_probe.py`: adapter-generated requests against fake-node.
- `fake-node/runtime_probe_block_height.py`: sync-health and block-height runtime probe against fake-node.
- `audit_sync_health_registry.py`: local sync-health registry audit.
- `fake-node/audit_health_probe_fixtures.py`: health-probe fixture audit.
- `fake-node/record_rpc_fixtures.sh`: record fixtures for `all`, one chain, or a comma-separated chain list.
- `fake-node/record_rpc_fixtures.py`: real endpoint request/response recording.
- `fake-node/validate_fixture_authenticity.py`: authenticity check for local recording evidence.

The legacy mock RPC path is compatibility-only. It does not validate the full
36-chain fixture replay path.

## Do Not Commit

Generated local artifacts:

- `__pycache__/`
- `*.pyc`
- `.DS_Store`
- `proxy/proxy` or any other built binary
- benchmark result directories
- large local request/response evidence under `docs/audit/rpc-fixtures/<chain>/<method>/`

## Notes

The framework supports 36 chain templates through 6 adapter families:

- `bitcoin_jsonrpc`
- `hedera_dual`
- `jsonrpc`
- `rest`
- `substrate`
- `tendermint`

Use `fake-node/record_rpc_fixtures.sh` as the fixture-recording entrypoint for
all chains, single chains, or comma-separated chain lists.
