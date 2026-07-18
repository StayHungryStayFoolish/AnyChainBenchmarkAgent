# Test Inventory

This directory contains framework regression tests for the 36-chain benchmark
runtime. The tests are intentionally small and mostly dependency-light; many can
run directly with `bash` or `python3`.

## Recommended Test Sets

Use these sets when validating changes:

```bash
# Complete ordinary Python regression suite. This runner permits local test
# servers but fails if a test attempts to contact an external service or LLM.
python3 tests/run_offline_python_suite.py

# Chain template, RPC target generation, fake-node coverage, and sync health.
python3 tests/test_chain_template_proxy_extraction.py
python3 tests/test_chain_adapters.py
python3 tests/test_param_spec.py
python3 tools/chain_adapters/cli.py validate-template --chain all
bash tests/test_target_generator_mixed_weighted.sh
python3 tests/test_sync_health_audit.py
python3 tools/fake-node/check_fixture_coverage.py --json

# Runtime file registry, startup cleanup, and monitor lifecycle contracts.
bash tests/test_runtime_path_registry.sh
bash tests/test_runtime_startup_cleanup.sh
bash tests/test_monitoring_lifecycle_smoke.sh
bash tests/test_monitoring_runtime_contract.sh
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh

# CSV/schema and report-generation contracts.
bash tests/test_csv_registry_symmetry.sh
bash tests/test_csv_header_data_alignment.sh
python3 tests/test_per_method_attribution.py
python3 tests/test_per_method_charts.py
python3 tests/test_per_method_report.py
python3 tests/test_disk_visualization_synthetic.py
```

For Linux-specific monitoring tests, run them in the Docker test environment
rather than on macOS.

## Current Tests

### Chain, RPC, and Fake Node

- `test_chain_template_proxy_extraction.py`: verifies all 36 chain templates can
  be parsed for proxy method extraction.
- `test_chain_adapters.py`: verifies the chain adapter factory, family routing,
  request generation, and 36-chain CLI target generation.
- `test_param_spec.py`: verifies custom RPC parameter specs, including
  three-argument JSON-RPC methods and REST query/body construction.
- `test_target_generator_mixed_weighted.sh`: verifies mixed mode honors
  `rpc_methods.mixed_weighted`.
- `test_sync_health_audit.py`: verifies the 36-chain sync-health registry is
  complete and parseable.
- `test_node_sync_health_state_machine.sh`: verifies sync-health state
  transitions.
- `test_block_height_csv_reader.sh`: verifies bottleneck detection can consume
  block-height/sync-health CSV fields.
- `test_node_health_cache_path.sh`: verifies node-health cache paths use the
  runtime registry.
- `test_proxy_phase.sh`: verifies proxy lifecycle and `proxy_method.csv` output.

### Runtime Files, Startup Cleanup, and Lifecycle

- `test_runtime_path_registry.sh`: verifies generated runtime paths are exported
  through the central path registry.
- `test_runtime_startup_cleanup.sh`: verifies stale runtime files, shared memory
  files, and PID markers are cleaned before a new run.
- `test_monitoring_lifecycle_audit.sh`: static lifecycle contract audit.
- `test_monitoring_lifecycle_smoke.sh`: dynamic lifecycle smoke test.
- `test_monitoring_runtime_contract.sh`: verifies performance CSV, memory-share
  JSON, block-height cache, archive output, and lifecycle markers.
- `test_full_entrypoint_fake_node_lifecycle_smoke.sh`: full local fake-node
  entrypoint smoke.
- `test_legacy_mock_rpc_smoke_harness.sh`: validates the legacy mock RPC smoke
  harness structure.

### Monitoring Collectors

- `test_cgroup_collector.py`: cgroup collector unit tests.
- `test_cgroup_collector_regression.py`: regression coverage for cgroup v1/v2
  edge cases.
- `test_cgroup_collector_wrapper.sh`: fail-soft wrapper contract.
- `test_unified_csv_cgroup_fields.sh`: verifies cgroup fields are wired into the
  unified performance CSV.
- `test_cgroup_kubelet_stats_fallback.py`: Kubernetes kubelet stats fallback mode.
- `test_system_collectors.sh`: CPU, memory, disk, and network collector
  contracts.
- `test_process_collectors.sh`: blockchain process discovery and resource
  collector contracts.
- `test_performance_data_line_builder.sh`: unified performance CSV row order.
- `test_sample_count_tracker.sh`: monitor sample counter persistence.
- `test_qps_runtime_reader.sh`: QPS runtime status and vegeta latency reader.
- `test_metrics_json_writer.sh`: latest/unified metrics JSON writer.
- `test_monitor_utils.sh`: common monitor utility and cleanup helpers.
- `test_monitor_error_recovery.sh`: fail-soft monitor execution helpers.
- `test_monitor_performance_advisor.sh`: performance advisor report contract.
- `test_monitoring_overhead.sh`: monitoring-overhead lifecycle contract.
- `test_monitoring_overhead_csv.sh`: monitoring-overhead CSV write contract.

### Provider and Device Handling

- `test_provider_contract.sh`: provider-aware metric contract.
- `test_cloud_provider_detect.sh`: cloud/provider/NIC driver detection.
- `test_cloud_provider_resolver.sh`: provider resolver helper.
- `test_iops_conversion.sh`: provider-adjusted IOPS conversion rules.
- `test_ena_data_normalizer.sh`: AWS ENA counter normalization.
- `test_aws_ena_parity.sh`: parity test for the AWS ENA provider collector.
- `test_validate_devices_degraded.sh`: device validation degraded-mode behavior.
- `test_single_disk_workload.sh`: single-disk workload generator.
- `test_provider_csv_end_to_end.sh`: provider-aware CSV smoke path.

Provider-specific tests are still current when the corresponding provider
collector is still supported. AWS ENA tests are not a framework default; they
protect the optional AWS ENA collector.

### CSV, Reporting, and Observability

- `test_csv_registry_symmetry.sh`: shell/Python CSV schema registry symmetry.
- `test_csv_header_data_alignment.sh`: performance CSV header/data alignment.
- `test_config_env_overrides.sh`: environment override contract.
- `test_per_method_attribution.py`: proxy-method attribution logic.
- `test_per_method_charts.py`: per-method chart generation.
- `test_per_method_report.py`: report HTML per-method section.
- `test_degraded_report.py`: degraded report generation.
- `test_disk_visualization_synthetic.py`: disk chart generation using synthetic
  data.
- `test_prometheus_exporter.py`: Prometheus text-format exporter output.

### Deployment and Kubernetes

- `test_deployment_mode_detector.sh`: deployment mode detection and env override
  behavior.
- `test_dockerfile_structure.py`: Dockerfile structure and healthcheck
  contract.
- `test_k8s_manifests.py`: Kubernetes manifest static validation.
- `test_k8s_api_client_regression.py`: Kubernetes API client regressions.
- `test_k8s_socket_timeout.py`: socket timeout handling.
- `test_k8s_monitoring_stack.py`: Kubernetes helper stack unit tests.
- `test_monitoring_k8s_diagnostics.sh`: monitoring coordinator Kubernetes diagnostics command.
- `test_rbac_endpoints.py`: Kubernetes RBAC endpoint access.
- `integration_k8s_cgroup_config_chain.sh`: monitoring and cgroup integration
  check.
- `test_monitoring_k8s_import_chain.py`: import-chain check for
  monitoring/Kubernetes helpers.
- `run_monitoring_k8s_tests.sh`: convenience runner for the monitoring and
  Kubernetes group.
- `smoke_config_loader_deployment_env.sh`: config loader smoke.
- `smoke_k8s_helpers_import.py`: Kubernetes helper import smoke.

### Legacy Mock RPC Compatibility

- `test_legacy_mock_rpc_server_regression.py`
- `test_legacy_mock_rpc_chain_forward.py`
- `test_legacy_mock_rpc_unknown_echo.py`

These tests cover `tools/legacy_mock_rpc_server.py`. The primary 36-chain
closed-loop path is `tools/fake-node`, while the legacy mock RPC server remains
available for compatibility smoke tests.

### Systemd Process Detection

- `test_systemd_unit_regex.sh`
- `test_systemd_regex_prefix.sh`

These tests protect process-name matching for node service discovery. They are
still relevant because users often run nodes under systemd with custom service
names.
