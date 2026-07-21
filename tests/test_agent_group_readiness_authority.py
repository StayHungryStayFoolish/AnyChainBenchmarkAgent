"""R39 architecture tests for group ownership and fallback readiness."""

from __future__ import annotations

from copy import deepcopy
import unittest

from agent.harness.contracts import StateDelta
from agent.harness.invariants import StateInvariantError, validate_delta_owner
from agent.harness.routing import chain_identity_confirmed, next_group_and_reason, sync_observe_readiness
from agent.workflows.group_registry import (
    FIELD_GROUP,
    FIELD_OWNER,
    GROUPS,
    GroupSpec,
    USER_NAVIGABLE_GROUPS,
    fallback_groups_for_workflow,
    group_for_field,
    validate_group_registry,
)


def _complete_environment() -> dict[str, object]:
    return {
        "CLOUD_REGION": "test-region",
        "CLOUD_ZONE": "test-zone",
        "MACHINE_TYPE": "test-machine",
        "LEDGER_DEVICE": "/dev/vdb",
        "DATA_VOL_TYPE": "gp3",
        "DATA_VOL_SIZE": "100",
        "DATA_VOL_MAX_IOPS": "3000",
        "DATA_VOL_MAX_THROUGHPUT": "125",
        "has_accounts_device": False,
        "NETWORK_INTERFACE": "eth0",
        "NETWORK_MAX_BANDWIDTH_GBPS": "10",
    }


def _sync_state(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "target_mode": "sync-observe",
        "workflow_mode": "sync_observe",
        "chain_identity": {"canonical": "ethereum", "status": "confirmed"},
        "confirmed_config": _complete_environment(),
        "endpoint_evidence": {},
        "sync_observe": {},
        "observability": {},
        "advanced_tuning": {},
        "preflight": {},
    }
    state.update(updates)
    return state


class GroupRegistryAuthorityTests(unittest.TestCase):
    def test_registry_construction_rejects_duplicate_persisted_field_owner(self) -> None:
        groups = (
            GroupSpec(name="one", owner="alpha", fields=("SHARED",)),
            GroupSpec(name="two", owner="beta", fields=("SHARED",)),
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate persisted field ownership"):
            validate_group_registry(groups)

    def test_registry_rejects_unknown_edges_and_dependency_cycles(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"dependencies=\['missing'\]"):
            validate_group_registry(
                (GroupSpec(name="one", owner="alpha", depends_on=("missing",)),)
            )

    def test_registry_rejects_invalid_navigation_contracts(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid navigation entry"):
            validate_group_registry((
                GroupSpec(name="one", owner="alpha", navigation_entry="unsupported"),  # type: ignore[arg-type]
            ))
        with self.assertRaisesRegex(RuntimeError, "action-only.*fallback"):
            validate_group_registry((
                GroupSpec(name="one", owner="alpha", navigation_entry="action_only"),
            ))

    def test_only_configuration_entry_groups_are_public_navigation_destinations(self) -> None:
        self.assertNotIn("job_monitoring", USER_NAVIGABLE_GROUPS)
        self.assertNotIn("failure_recovery", USER_NAVIGABLE_GROUPS)
        self.assertNotIn("error_evidence_analysis", USER_NAVIGABLE_GROUPS)
        self.assertNotIn("report_artifact_analysis", USER_NAVIGABLE_GROUPS)
        self.assertIn("qps_profile", USER_NAVIGABLE_GROUPS)
        self.assertIn("sync_observe", USER_NAVIGABLE_GROUPS)
        with self.assertRaisesRegex(RuntimeError, "dependency cycle"):
            validate_group_registry(
                (
                    GroupSpec(name="one", owner="alpha", depends_on=("two",)),
                    GroupSpec(name="two", owner="beta", depends_on=("one",)),
                )
            )

    def test_production_registry_has_one_exact_owner_for_every_field(self) -> None:
        declared = [field for group in GROUPS for field in group.fields]

        self.assertEqual(len(declared), len(set(declared)))
        self.assertEqual(set(declared), set(FIELD_GROUP))
        self.assertEqual(set(declared), set(FIELD_OWNER))
        self.assertEqual(group_for_field("BLOCKCHAIN_PROCESS_NAMES"), "endpoint_process")
        self.assertEqual(group_for_field("MAINNET_RPC_URL"), "endpoint_process")
        self.assertEqual(FIELD_OWNER["BLOCKCHAIN_PROCESS_NAMES"], "chain_rpc")
        self.assertNotIn("latest_job_id", FIELD_OWNER)
        self.assertEqual(group_for_field("Latest_Job_Id"), "")

    def test_dependencies_and_invalidations_reference_registered_groups(self) -> None:
        names = {group.name for group in GROUPS}
        for group in GROUPS:
            self.assertLessEqual(set(group.depends_on), names, group.name)
            self.assertLessEqual(set(group.invalidates), names, group.name)

    def test_workflow_metadata_preserves_order_and_excludes_rpc_groups_from_sync(self) -> None:
        rpc = [group.name for group in fallback_groups_for_workflow("rpc_benchmark")]
        sync = [group.name for group in fallback_groups_for_workflow("sync_observe")]

        self.assertEqual(rpc, [group.name for group in GROUPS if group.fallback and group.name != "sync_observe"])
        self.assertNotIn("sync_observe", rpc)
        self.assertIn("sync_observe", sync)
        for rpc_only in ("workload_rpc", "target_samples_fixtures", "qps_profile"):
            self.assertNotIn(rpc_only, sync)


class SyncObservePublicContractTests(unittest.TestCase):
    def test_public_prepare_and_draft_tools_share_canonical_sync_fields(self) -> None:
        from agent.harness.sync_observe_contract import sync_observe_tool_properties
        from agent.tools.schema import TOOL_SPECS

        by_name = {spec.name: spec for spec in TOOL_SPECS}
        expected = sync_observe_tool_properties()
        for name in ("prepare_benchmark_run", "draft_request"):
            actual = {
                key: by_name[name].properties[key]
                for key in expected
            }
            self.assertEqual(actual, expected, name)

    def test_structured_tool_request_preserves_sync_observe_contract(self) -> None:
        from agent.tools.executor import _structured_request

        request = _structured_request({
            "workflow_type": "sync_observe",
            "sync_observe_source": "existing_local_node",
            "sync_observe_rpc_url": "http://127.0.0.1:8545",
            "sync_observe_rpc_url_ready": True,
            "node_process_identity": "geth",
            "mainnet_rpc_url_reviewed": True,
            "sync_observe_stop_condition": "duration",
            "sync_observe_duration_seconds": 60,
            "node_prometheus_metrics_url": "http://127.0.0.1:6060/debug/metrics/prometheus",
        })

        self.assertEqual(request["workflow_type"], "sync_observe")
        self.assertTrue(request["sync_observe_rpc_url_ready"])
        self.assertEqual(request["node_process_identity"], "geth")
        self.assertEqual(request["sync_observe_stop_condition"], "duration")
        self.assertEqual(request["sync_observe_duration_seconds"], 60)


class MutationOwnerTests(unittest.TestCase):
    def test_endpoint_and_job_fields_have_one_positive_mutation_owner(self) -> None:
        process_delta = StateDelta.set_values(
            {"confirmed_config": {"BLOCKCHAIN_PROCESS_NAMES": "geth"}}
        )
        job_delta = StateDelta.set_values({"job": {"job_id": "job-1"}})

        validate_delta_owner(process_delta, "chain_rpc")
        validate_delta_owner(job_delta, "execution")
        with self.assertRaisesRegex(StateInvariantError, "environment does not own"):
            validate_delta_owner(process_delta, "environment")
        with self.assertRaisesRegex(StateInvariantError, "analysis does not own"):
            validate_delta_owner(job_delta, "analysis")


class RegistryDrivenReadinessTests(unittest.TestCase):
    def test_raw_unknown_candidate_is_not_a_confirmed_chain(self) -> None:
        state = {
            "chain_identity": {
                "raw": "Run it against BNB Smart Chain.",
                "canonical": "Run it against BNB Smart Chain.",
                "proposed_known_chain": "bsc",
                "status": "needs_known_chain_confirmation",
            }
        }

        self.assertFalse(chain_identity_confirmed(state))

    def test_default_rpc_fallback_order_is_registry_order(self) -> None:
        state: dict[str, object] = {
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "ethereum", "status": "confirmed"},
            "confirmed_config": _complete_environment(),
            "rpc_mode": "single",
            "workload": {"confirmed": True},
            "qps_profile": {"mode": "quick", "confirmed": True},
            "observability": {"mode": "disabled"},
            "advanced_tuning": {"confirmed": True},
            "preflight": {},
        }

        self.assertEqual(next_group_and_reason(state)[0], "preflight_smoke_execution")
        state["preflight"] = {"approved": True}
        self.assertEqual(next_group_and_reason(state), ("job_monitoring", "monitor benchmark job"))

    def test_sync_observe_contract_routes_source_endpoint_and_stop_condition(self) -> None:
        state = _sync_state()
        self.assertEqual(sync_observe_readiness(state).group, "sync_observe")
        self.assertEqual(next_group_and_reason(state)[0], "sync_observe")

        state["sync_observe"] = {"source": "endpoint_only"}
        self.assertEqual(sync_observe_readiness(state).group, "endpoint_process")
        self.assertEqual(next_group_and_reason(state)[0], "endpoint_process")

        state["endpoint_evidence"] = {"sync_rpc_url_ready": True}
        state["confirmed_config"]["SYNC_OBSERVE_RPC_URL"] = "http://node:8545"
        state["confirmed_config"]["MAINNET_RPC_URL_REVIEWED"] = True
        self.assertEqual(next_group_and_reason(state), ("sync_observe", "choose sync-observe stop condition"))

        state["sync_observe"] = {
            "source": "endpoint_only",
            "stop_condition": "duration",
        }
        self.assertEqual(next_group_and_reason(state), ("sync_observe", "confirm sync-observe duration"))

    def test_sync_observe_resume_never_traverses_workload_qps_or_fixtures(self) -> None:
        state = _sync_state(
            endpoint_evidence={"sync_rpc_url_ready": True},
            sync_observe={
                "source": "endpoint_only",
                "stop_condition": "until_synced",
                "metrics": {"m_gas_per_second": 12.5},
            },
            rpc_mode="mixed",
            workload={"confirmed": False},
            fixture_evidence={"required": True, "status": "missing"},
            qps_profile={},
            job={"job_id": "job-sync", "status": "running"},
            report_context={"requested_job_id": "job-sync"},
        )
        state["confirmed_config"]["MAINNET_RPC_URL_REVIEWED"] = True
        state["confirmed_config"]["SYNC_OBSERVE_RPC_URL"] = "http://node:8545"
        state["confirmed_config"]["NODE_PROMETHEUS_METRICS_URL"] = "http://metrics:6060"
        before = deepcopy(state)

        self.assertEqual(next_group_and_reason(state)[0], "observability")
        self.assertEqual(state, before)

        state["observability"] = {"mode": "disabled"}
        state["advanced_tuning"] = {"confirmed": True}
        state["preflight"] = {"approved": True}
        self.assertEqual(next_group_and_reason(state), ("job_monitoring", "monitor sync-observe job"))
        self.assertEqual(state["sync_observe"]["metrics"]["m_gas_per_second"], 12.5)
        self.assertEqual(state["job"]["job_id"], "job-sync")
        self.assertEqual(state["report_context"]["requested_job_id"], "job-sync")


if __name__ == "__main__":
    unittest.main()
