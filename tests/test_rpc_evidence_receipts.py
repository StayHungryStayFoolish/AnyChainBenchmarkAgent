"""Contracts for secret-free receipts emitted by authoritative RPC owners."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.harness.domains.rpc_receipts import evidence_hash, validate_rpc_receipt
from agent.harness.state import new_state


def _receipts(state: dict, receipt_type: str) -> list[dict]:
    return [
        dict(item)
        for item in (state.get("turn_context") or {}).get("control_receipts") or ()
        if item.get("receipt_type") == receipt_type
    ]


def _catalog(methods: list[dict], *, chain: str) -> dict:
    return {
        "catalog": {
            "contract_version": 1,
            "revision": 1,
            "chain": chain,
            "methods": methods,
            "finished": bool(methods),
        }
    }


class RpcEvidenceReceiptTest(unittest.TestCase):
    def test_domain_receipt_survives_the_atomic_graph_commit_boundary(self) -> None:
        from tests.agent_live.graph_turn import invoke_actions

        state = new_state("rpc-receipt-commit", language="en")
        state.update(
            {
                "target_mode": "real-node",
                "workflow_mode": "rpc_benchmark",
                "chain_identity": {
                    "raw": "bsc",
                    "canonical": "bsc",
                    "status": "confirmed",
                    "adapter_family": "jsonrpc",
                },
                "custom_rpc": {
                    "status": "needs_endpoint",
                    "job_local_override": True,
                },
                "active_group": "endpoint_process",
            }
        )
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={
                "ready": True,
                "status": "ok",
                "evidence_file": "/tmp/probe.json",
            },
        ):
            result = invoke_actions(
                state,
                [
                    {
                        "type": "rpc_catalog_command",
                        "catalog_command": "set_endpoint",
                        "rpc_endpoint": "http://geth-dev:8545",
                        "source_evidence": "use the local endpoint",
                        "confidence": "high",
                    }
                ],
                "use the local endpoint",
            )

        receipts = _receipts(result, "rpc_endpoint_role")
        self.assertTrue(receipts)
        self.assertTrue(validate_rpc_receipt(receipts[-1])[0])
        self.assertEqual(receipts[-1]["owner"], "rpc_endpoint")
        self.assertEqual(receipts[-1]["role"], "validation")
        domain_commits = _receipts(result, "domain_commit")
        self.assertTrue(domain_commits)
        self.assertTrue(domain_commits[-1]["material_delta"])

    def test_rpc_receipt_semantic_tampering_fails_closed(self) -> None:
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt

        state = new_state("rpc-receipt-tamper")
        state["turn_context"] = {"text": "endpoint"}
        emit_endpoint_role_receipt(
            state,
            role="validation",
            case="custom_rpc",
            endpoint="https://example.invalid/private-token",
            ready=True,
            probe_status="ok",
            chain="bsc",
            adapter_family="jsonrpc",
            methods=["eth_blockNumber"],
        )
        receipt = deepcopy(_receipts(state, "rpc_endpoint_role")[-1])
        receipt["owner"] = "analysis"
        receipt["receipt_id"] = evidence_hash(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )

        valid, reason = validate_rpc_receipt(receipt)

        self.assertFalse(valid)
        self.assertEqual(reason, "invalid RPC receipt owner")

    def test_coordinator_preserves_valid_blocker_receipt_and_rejects_owner_spoofing(
        self,
    ) -> None:
        from agent.harness.contracts import HandlerResult
        from agent.harness.coordinator import _apply_handler_result
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt
        from agent.harness.invariants import StateInvariantError

        base = new_state("rpc-blocker-receipt")
        base["turn_context"] = {"text": "probe endpoint", "control_receipts": []}
        isolated = deepcopy(base)
        emit_endpoint_role_receipt(
            isolated,
            role="validation",
            case="custom_rpc",
            endpoint="https://example.invalid/private-token",
            ready=False,
            probe_status="blocked",
            chain="bsc",
            adapter_family="jsonrpc",
        )
        receipt = _receipts(isolated, "rpc_endpoint_role")[-1]

        committed = _apply_handler_result(
            base,
            HandlerResult(
                control_receipts=(receipt,),
                blocker="endpoint probe failed",
            ),
            owner="chain_rpc",
        )

        committed_receipts = _receipts(committed, "rpc_endpoint_role")
        self.assertEqual(
            [item["receipt_id"] for item in committed_receipts],
            [receipt["receipt_id"]],
        )

        spoofed = deepcopy(receipt)
        spoofed["owner"] = "analysis"
        spoofed["receipt_id"] = evidence_hash(
            {key: value for key, value in spoofed.items() if key != "receipt_id"}
        )
        with self.assertRaises(StateInvariantError):
            _apply_handler_result(
                base,
                HandlerResult(control_receipts=(spoofed,)),
                owner="chain_rpc",
            )

    def test_case2_promotion_rejects_probe_from_another_endpoint(self) -> None:
        from agent.harness.domains.chain_handoff import _promote_case2_endpoint

        state = new_state("case2-promotion-boundary", language="en")
        state.update(
            {
                "target_mode": "fake-node",
                "workflow_mode": "rpc_benchmark",
                "chain_identity": {
                    "raw": "flow-evm",
                    "canonical": "flow-evm",
                    "adapter_family": "jsonrpc",
                    "status": "existing_family_runtime_choice",
                    "case": "case2",
                },
                "endpoint_evidence": {
                    "candidate_endpoint": "https://current.invalid/rpc",
                    "candidate_endpoint_ready": True,
                    "new_chain_endpoint_probe": {
                        "ready": True,
                        "endpoint": "https://old.invalid/rpc",
                        "chain": "flow-evm",
                        "transport": "jsonrpc",
                        "status": "ok",
                    },
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "chain": "flow-evm",
                        "methods": [
                            {
                                "method": "eth_blockNumber",
                                "params": [],
                            }
                        ],
                        "finished": True,
                    }
                },
            }
        )

        promoted = _promote_case2_endpoint(state)

        self.assertFalse(promoted)
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertEqual(
            state["chain_identity"]["status"],
            "existing_family_needs_endpoint",
        )

    def test_endpoint_owner_records_roles_without_endpoint_or_auth_material(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer

        state = new_state("endpoint-receipts")
        state["turn_index"] = 4
        state["turn_context"] = {"text": "endpoint evidence turn"}
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        secret_validation_endpoint = "https://node.invalid/v1/private-api-key"
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={
                "ready": True,
                "status": "ok",
                "evidence_file": "/tmp/private-api-key/probe.json",
            },
        ):
            _apply_endpoint_answer(
                state,
                "custom_rpc_endpoint",
                secret_validation_endpoint,
            )

        validation = _receipts(state, "rpc_endpoint_role")[-1]
        unsigned = dict(validation)
        receipt_id = unsigned.pop("receipt_id")
        self.assertEqual(receipt_id, evidence_hash(unsigned))
        self.assertEqual(validation["role"], "validation")
        self.assertEqual(validation["case"], "custom_rpc")
        self.assertEqual(
            validation["endpoint_hash"],
            evidence_hash(secret_validation_endpoint),
        )
        serialized = json.dumps(validation, sort_keys=True)
        self.assertNotIn(secret_validation_endpoint, serialized)
        self.assertNotIn("private-api-key", serialized)

        state["custom_rpc"] = {
            **state["custom_rpc"],
            **_catalog([{"method": "demo_private", "params": []}], chain="bsc"),
        }
        state["workload"] = {"methods": ["demo_private"]}
        secret_final_endpoint = "https://final.invalid/bearer-secret"
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={
                "ready": True,
                "status": "ok",
                "evidence_file": "/tmp/bearer-secret/final.json",
            },
        ):
            _apply_endpoint_answer(state, "LOCAL_RPC_URL", secret_final_endpoint)

        final = _receipts(state, "rpc_endpoint_role")[-1]
        self.assertEqual(final["role"], "final_benchmark")
        self.assertEqual(final["methods"], ["demo_private"])
        self.assertNotIn(secret_final_endpoint, json.dumps(final, sort_keys=True))
        self.assertNotEqual(validation["endpoint_hash"], final["endpoint_hash"])

    def test_rest_identity_is_hashed_instead_of_exposing_path_material(self) -> None:
        from agent.harness.domains.rpc_receipts import (
            emit_workload_commit_receipt,
        )

        state = new_state("rest-method-receipt")
        state["turn_context"] = {"text": "commit REST workload"}
        state["rpc_mode"] = "single"
        state["workload"] = {
            "confirmed": True,
            "choice": "custom_rpc",
            "methods": ["GET /v1/accounts/private-api-key"],
            "replace_defaults": True,
            "job_local_override": True,
        }

        emit_workload_commit_receipt(state, case="custom_rpc")

        receipt = _receipts(state, "rpc_workload_commit")[-1]
        self.assertEqual(receipt["methods"], [])
        self.assertEqual(len(receipt["method_hashes"]), 1)
        self.assertNotIn("private-api-key", json.dumps(receipt, sort_keys=True))

    def test_schema_and_catalog_owners_emit_field_and_revision_provenance(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_schema_evidence

        state = new_state("schema-receipts")
        state["turn_index"] = 7
        state["turn_context"] = {"text": "schema evidence turn"}
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://validation.invalid/private-token",
            "endpoint_ready": True,
        }
        request = (
            '{"jsonrpc":"2.0","id":1,"method":"demo_lookup",'
            '"params":["0xsecret-account","latest"]}'
        )
        extracted = {
            "status": "draft",
            "method": "demo_lookup",
            "transport": "jsonrpc",
            "params": [
                {
                    "index": 0,
                    "name": "account",
                    "json_type": "string",
                    "semantic_type": "account_address",
                    "encoding": "hex",
                    "meaning": "account to query",
                    "example": "0xsecret-account",
                    "required": True,
                },
                {
                    "index": 1,
                    "name": "block",
                    "json_type": "string",
                    "semantic_type": "block_reference",
                    "encoding": "tag",
                    "meaning": "state version",
                    "example": "latest",
                    "required": True,
                },
            ],
            "response_summary": "unknown",
            "response_fields": [],
        }
        with patch(
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            return_value=extracted,
        ):
            self.assertTrue(
                _apply_schema_evidence(
                    state,
                    case="custom_rpc",
                    evidence=request,
                )
            )

        transitions = _receipts(state, "rpc_catalog_transition")
        self.assertEqual(
            [item["command"] for item in transitions],
            ["append_evidence", "correct_draft"],
        )
        self.assertEqual(
            [item["catalog_revision"] for item in transitions],
            [1, 2],
        )
        provenance = _receipts(state, "rpc_schema_provenance")[-1]
        by_path = {item["field_path"]: item for item in provenance["fields"]}
        self.assertEqual(
            by_path["params_json"]["source_kind"],
            "protocol_request_parser",
        )
        self.assertEqual(
            by_path["params[0].semantic_type"]["source_kind"],
            "model_extraction_from_user_evidence",
        )
        self.assertEqual(
            by_path["validation_endpoint"]["source_kind"],
            "rpc_endpoint_role",
        )
        serialized = json.dumps(provenance, sort_keys=True)
        self.assertNotIn("0xsecret-account", serialized)
        self.assertNotIn("private-token", serialized)
        self.assertTrue(
            all(len(item["value_hash"]) == 64 for item in provenance["fields"])
        )

    def test_workload_owner_records_the_effective_single_commit(self) -> None:
        from agent.harness.domains.rpc_workload import _apply_scope

        state = new_state("workload-receipts")
        state["turn_index"] = 9
        state["turn_context"] = {"text": "use the validated method only"}
        state["chain_identity"] = {"canonical": "bsc"}
        state["custom_rpc"] = {
            "status": "needs_scope",
            **_catalog([{"method": "demo_lookup", "params": []}], chain="bsc"),
        }

        _apply_scope(state, "custom_rpc_scope", "single_replace")

        receipt = _receipts(state, "rpc_workload_commit")[-1]
        self.assertEqual(receipt["owner"], "rpc_workload")
        self.assertEqual(receipt["rpc_mode"], "single")
        self.assertEqual(receipt["methods"], ["demo_lookup"])
        self.assertTrue(receipt["replace_defaults"])
        self.assertTrue(receipt["job_local_override"])
        self.assertEqual(receipt["catalog_revision"], 1)

    def test_probe_and_add_method_receipts_close_observed_schema_lineage(self) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
        )

        state = new_state("probe-catalog-receipts")
        state["turn_context"] = {"text": "probe and accept method"}
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        correct_draft(
            state,
            {
                "method": "demo_health",
                "params": [],
                "params_json": [],
                "response_summary": "hex status",
                "field_provenance": [
                    {
                        "field_path": "method",
                        "source_kind": "protocol_request_parser",
                        "source_revisions": [1],
                        "value_hash": evidence_hash("demo_health"),
                    }
                ],
            },
        )
        self.assertTrue(confirm_request(state, True).accepted)
        probe = {"ready": True, "status": "ok"}
        observed = {
            "shape_hash": "shape-secret",
            "sample": '{"result":"private-response"}',
            "http_status": 200,
            "evidence_file": "/tmp/private-response.json",
        }
        self.assertTrue(record_probe(state, probe, observed).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)
        self.assertTrue(
            add_validated_method(
                state,
                {
                    "method": "demo_health",
                    "params": [],
                    "schema": {},
                    "observed_response": observed,
                },
            ).accepted
        )

        provenance = _receipts(state, "rpc_schema_provenance")[-1]
        paths = {item["field_path"] for item in provenance["fields"]}
        self.assertIn("observed_response.sample", paths)
        self.assertNotIn("private-response", json.dumps(provenance, sort_keys=True))
        transition = _receipts(state, "rpc_catalog_transition")[-1]
        self.assertEqual(transition["command"], "add_method")
        self.assertEqual(transition["method_count"], 1)
        self.assertEqual(transition["method_names"], ["demo_health"])

    def test_strategy_materialization_is_observed_without_parameter_values(self) -> None:
        from agent.harness.domains.execution_runtime import (
            _execution_result,
            _prepare_benchmark_with_runtime_contract,
        )

        state = new_state("materialization-receipt")
        state["turn_index"] = 11
        state["turn_context"] = {"text": "prepare the custom workload"}
        state.update(
            {
                "chain_identity": {
                    "canonical": "case2-receipt",
                    "adapter_family": "jsonrpc",
                },
                "custom_rpc": _catalog(
                    [{"method": "demo_object", "params": {"owner": "0xsecret"}}],
                    chain="case2-receipt",
                ),
                "workload": {
                    "confirmed": True,
                    "job_local_override": True,
                    "methods": ["demo_object"],
                    "replace_defaults": True,
                },
                "rpc_mode": "single",
                "target_mode": "real-node",
                "confirmed_config": {
                    "LOCAL_RPC_URL": "https://node.invalid/private-token"
                },
            }
        )
        original = deepcopy(state)
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text("{}\n", encoding="utf-8")
            prepared = {
                "status": "blocked",
                "warnings": ["chain_template_exists"],
                "data": {
                    "plan": {"chain": "case2-receipt", "artifacts": {}},
                    "plan_file": str(plan_file),
                    "preflight": {"blockers": ["chain_template_exists"]},
                },
            }
            preflight = {
                "passed": False,
                "checks": [
                    {
                        "name": "chain_template_exists",
                        "passed": False,
                        "detail": "missing",
                    },
                    {
                        "name": "chain_template_json_valid",
                        "passed": False,
                        "detail": "missing",
                    },
                    {"name": "runtime_contract", "passed": True, "detail": "ok"},
                ],
                "blockers": [
                    "chain_template_exists",
                    "chain_template_json_valid",
                ],
            }
            with (
                patch(
                    "agent.runners.application_service.prepare_benchmark_run",
                    return_value=prepared,
                ),
                patch(
                    "agent.runners.application_service.run_preflight",
                    return_value=preflight,
                ),
            ):
                result = _prepare_benchmark_with_runtime_contract(state)

        override = result["data"]["plan"]["chain_config_override"]
        materialization = override["_meta"]["materialization_evidence"]
        self.assertEqual(
            materialization["source_kind"],
            "new_chain_runtime_template",
        )
        self.assertEqual(
            materialization["method_hashes"],
            [evidence_hash("demo_object")],
        )
        self.assertEqual(
            len(materialization["parameter_contracts"][0]["contract_hash"]),
            64,
        )
        receipt = _receipts(state, "rpc_workload_materialization")[-1]
        self.assertEqual(
            receipt["method_hashes"],
            [evidence_hash("demo_object")],
        )
        self.assertEqual(receipt["owner"], "strategy_planner")
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("0xsecret", serialized)
        self.assertNotIn("private-token", serialized)
        self.assertEqual(len(receipt["template_hash"]), 64)
        self.assertEqual(
            receipt["template_hash_scope"],
            "template_without_materialization_evidence",
        )
        handler_result = _execution_result(original, state)
        self.assertEqual(
            [
                item["receipt_type"]
                for item in handler_result.control_receipts
            ],
            ["rpc_workload_materialization"],
        )


if __name__ == "__main__":
    unittest.main()
