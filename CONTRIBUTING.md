# Contributing

Thanks for improving AnyChain Benchmark Agent. This project is a benchmark
framework, not only a script wrapper around Vegeta, so PRs must preserve the
runtime data contracts as well as code correctness.

## Development Flow

1. Create a branch from `main`.
2. Keep changes scoped to one behavior or module.
3. Run the test set that matches the touched area.
4. Open a pull request using the template.
5. Wait for GitHub Actions and code review before merge.

Do not commit local benchmark outputs, private RPC endpoints, API keys,
provider credentials, or machine-specific paths.

## Contribution License Terms

By submitting a pull request, you confirm that:

- You have the right to submit the contribution.
- Your contribution may be licensed under AGPL 3.0 or later and under the
  project's commercial licensing terms.
- You grant the project maintainers the right to use and sublicense your
  contribution in commercial versions.

## Commit and PR Title Convention

Use Conventional Commits for PR titles and for local commits when practical:

```text
type(scope): short summary
```

Allowed types:

- `feat`: new user-facing capability
- `fix`: bug fix
- `docs`: documentation-only change
- `test`: tests-only change
- `refactor`: code restructuring without intended behavior change
- `perf`: performance improvement
- `ci`: GitHub Actions or CI changes
- `build`: dependency updates or build-system changes
- `chore`: repository maintenance, build tooling, or cleanup

Examples:

```text
feat(fake-node): add fixture coverage validation
fix(monitoring): read sync health from runtime cache
docs(readme): clarify Kubernetes quick start
ci(pr-gate): require conventional PR titles
refactor(config): centralize runtime path registry
```

PR titles are enforced because ordinary PRs should use squash merge and the PR
title becomes the final commit subject. Intermediate commits in a PR should
follow the same style where possible, but maintainers may squash or reword them
before merge.

## Required Local Checks by Change Type

Chain template, RPC method, or sync-health changes:

```bash
python3 tests/test_chain_template_proxy_extraction.py
python3 tests/test_chain_adapters.py
python3 tests/test_param_spec.py
python3 tools/chain_adapters/cli.py validate-template --chain all
bash tests/test_target_generator_mixed_weighted.sh
python3 tests/test_sync_health_audit.py
python3 tools/fake-node/check_fixture_coverage.py --json
```

Proxy or per-method attribution changes:

```bash
(cd tools/proxy && go test ./...)
python3 tests/test_per_method_attribution.py
python3 tests/test_per_method_charts.py
python3 tests/test_per_method_report.py
```

Monitoring, runtime file, or bottleneck-detection changes should be tested on
Linux or in a Linux container:

```bash
bash tests/test_runtime_path_registry.sh
bash tests/test_runtime_startup_cleanup.sh
bash tests/test_monitoring_lifecycle_audit.sh
bash tests/test_monitoring_lifecycle_smoke.sh
bash tests/test_monitoring_runtime_contract.sh
bash tests/test_node_health_cache_path.sh
bash tests/test_node_sync_health_state_machine.sh
```

Report, chart, or observability changes:

```bash
python3 tests/test_degraded_report.py
python3 tests/test_disk_visualization_synthetic.py
python3 tests/test_prometheus_exporter.py
```

Kubernetes changes:

```bash
python3 tests/test_k8s_manifests.py
python3 tests/test_k8s_api_client_regression.py
python3 tests/test_k8s_monitoring_stack.py
bash tests/run_monitoring_k8s_tests.sh
```

Entry-point or lifecycle changes should run the full smoke on Linux:

```bash
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh
```

## GitHub PR Gate

Every PR must pass `.github/workflows/pr_gate.yml`. The required jobs are:

- repository hygiene and static contracts
- chain templates, adapters, and fake-node fixtures
- reports, attribution, and observability
- monitoring lifecycle and runtime file contracts
- Go module tests
- Docker and Kubernetes static checks

The manual or scheduled `.github/workflows/full_smoke.yml` workflow runs the
full fake-node entrypoint lifecycle smoke. Maintainers should run it before
merging PRs that touch the entry script, monitor lifecycle, archive lifecycle,
proxy lifecycle, or fake-node runtime.

## Chain and RPC Extension Rules

- Add chain behavior through `config/chains/<chain>.json` and
  `tools/chain_adapters/`.
- Do not hardcode chain-specific behavior in shared shell logic.
- Mixed workloads must use `rpc_methods.mixed_weighted` and sum to 100.
- fake-node fixtures are matched by `chain + method + fixture`, not just by
  parameter names such as `address` or `tx_hash`.
- Every trusted workload method should have fixture coverage.

## Monitoring and File Contract Rules

- Runtime artifacts must go through the configured runtime directories and
  memory-share paths.
- Consumers should read CSV columns by name, not by fixed index.
- New monitor collectors must define lifecycle, PID cleanup, CSV schema, and
  smoke tests.
- Missing metrics should become an explicit degraded state or report notice,
  not a fabricated value.

## Public Repository Rules

Before opening a PR, run:

```bash
bash ci/check_parallel_entry.sh
bash ci/check_csv_registry_bypass.sh
python3 tools/check_public_repo_markers.py --root .
```

The public repo marker check rejects internal execution markers, stale
planning labels, runtime Chinese output in code paths, and other public-release
cleanup issues.
