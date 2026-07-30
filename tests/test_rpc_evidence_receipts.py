"""Contracts for secret-free receipts emitted by authoritative RPC owners."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.harness.domains.rpc_receipts import (
    evidence_hash,
    exact_value_hash,
    validate_rpc_receipt,
)
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
    def _probe_result(
        self,
        *,
        chain: str,
        endpoint: str,
        method: str,
        params: object,
    ) -> dict:
        from agent.validators.endpoint_probe import (
            build_rpc_probe_contract,
            rpc_probe_contract_hash,
        )

        contract = build_rpc_probe_contract(
            chain=chain,
            endpoint=endpoint,
            transport="jsonrpc",
            methods=[method],
            method_params={method: params},
        )
        contract_hash = rpc_probe_contract_hash(contract)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        with handle:
            json.dump(
                {
                    "ready": True,
                    "status": "ok",
                    "probe_contract": contract,
                    "probe_contract_hash": contract_hash,
                    "checks": [{
                        "name": f"method_probe:{method}",
                        "passed": True,
                        "http_status": 200,
                        "response_shape_hash": "a" * 64,
                        "response_sample": (
                            '{"jsonrpc":"2.0","result":"0x1"}'
                        ),
                    }],
                },
                handle,
            )
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return {
            "ready": True,
            "status": "ok",
            "evidence_file": str(path),
            "probe_contract": contract,
            "probe_contract_hash": contract_hash,
        }

    def test_endpoint_source_identity_survives_redaction_collision(self) -> None:
        first = "https://user:secret-a@rpc.example"
        second = "https://user:secret-b@rpc.example"

        self.assertEqual(evidence_hash(first), evidence_hash(second))
        self.assertNotEqual(exact_value_hash(first), exact_value_hash(second))

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
        endpoint = "https://example.invalid/private-token"
        state["current_action"] = {"action_id": "endpoint-action"}
        state["turn_context"] = {
            "text": "endpoint",
            "admitted_actions": [{
                "action_id": "endpoint-action",
                "argument_value_hashes": {
                    "selected_value": exact_value_hash(endpoint),
                },
            }],
        }
        emit_endpoint_role_receipt(
            state,
            role="validation",
            case="custom_rpc",
            config_field="",
            endpoint=endpoint,
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

    def test_schema_receipt_recomputes_field_hash_from_current_draft(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import correct_draft

        evidence = '{"method":"demo_health","params":[]}'
        state = new_state("schema-field-hash-authority")
        state["current_action"] = {"action_id": "schema-action"}
        state["turn_context"] = {
            "text": evidence,
            "admitted_actions": [{
                "action_id": "schema-action",
                "argument_value_hashes": {
                    "source_evidence": exact_value_hash(evidence),
                },
            }],
        }

        correct_draft(
            state,
            {
                "method": "demo_health",
                "params": [],
                "params_json": [],
                "evidence": [{
                    "revision": 1,
                    "source": "user",
                    "kind": "protocol_request",
                    "content": evidence,
                    "method": "demo_health",
                    "source_action_value_hash": exact_value_hash(evidence),
                }],
                "field_provenance": [{
                    "field_path": "method",
                    "source_kind": "protocol_request_parser",
                    "source_revisions": [1],
                    "value_hash": "f" * 64,
                }],
            },
        )

        receipt = _receipts(state, "rpc_schema_provenance")[-1]
        self.assertEqual(
            receipt["fields"][0]["value_hash"],
            evidence_hash("demo_health"),
        )
        self.assertNotEqual(receipt["fields"][0]["value_hash"], "f" * 64)

    def test_schema_receipt_requires_nonempty_source_evidence(self) -> None:
        body = {
            "receipt_type": "rpc_schema_provenance",
            "receipt_version": 2,
            "turn_index": 1,
            "owner": "rpc_catalog",
            "method": "demo_health",
            "method_hash": evidence_hash("demo_health"),
            "catalog_revision": 1,
            "fields": [{
                "field_path": "method",
                "source_kind": "protocol_request_parser",
                "source_revisions": [1],
                "value_hash": evidence_hash("demo_health"),
            }],
            "fields_hash": "",
            "source_evidence_hashes": [],
            "source_action_value_hashes": [],
            "source_bindings": [],
            "producer_action_id": "schema-action",
        }
        body["fields_hash"] = evidence_hash(body["fields"])
        body["receipt_id"] = evidence_hash(body)

        valid, reason = validate_rpc_receipt(body)

        self.assertFalse(valid)
        self.assertEqual(reason, "schema provenance fields hash mismatch")

    def test_catalog_receipt_rejects_partially_bound_evidence(self) -> None:
        body = {
            "receipt_type": "rpc_catalog_transition",
            "receipt_version": 2,
            "turn_index": 1,
            "owner": "rpc_catalog",
            "command": "append_evidence",
            "catalog_revision": 2,
            "accepted": True,
            "phase": "evidence",
            "chain": "bsc",
            "method_names": [],
            "method_hashes": [],
            "method_count": 0,
            "draft_method": "demo_health",
            "draft_method_hash": evidence_hash("demo_health"),
            "source_evidence_hashes": ["1" * 64, "2" * 64],
            "source_action_value_hashes": ["3" * 64],
            "source_bindings": [
                {
                    "evidence_hash": "1" * 64,
                    "action_value_hash": "3" * 64,
                }
            ],
            "finished": False,
            "producer_action_id": "schema-action",
        }
        body["receipt_id"] = evidence_hash(body)

        valid, reason = validate_rpc_receipt(body)

        self.assertFalse(valid)
        self.assertEqual(
            reason,
            "RPC source evidence binding mismatch",
        )

    def test_coordinator_preserves_valid_blocker_receipt_and_rejects_owner_spoofing(
        self,
    ) -> None:
        from agent.harness.contracts import FailureDescriptor, HandlerResult
        from agent.harness.coordinator import _apply_handler_result
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt
        from agent.harness.invariants import StateInvariantError

        base = new_state("rpc-blocker-receipt")
        endpoint = "https://example.invalid/private-token"
        base["turn_context"] = {
            "text": "probe endpoint",
            "control_receipts": [],
            "admitted_actions": [{
                "action_id": "endpoint-probe-action",
                "argument_value_hashes": {
                    "selected_value": exact_value_hash(endpoint),
                },
            }],
        }
        base["current_action"] = {
            "action_id": "endpoint-probe-action",
        }
        isolated = deepcopy(base)
        emit_endpoint_role_receipt(
            isolated,
            role="validation",
            case="custom_rpc",
            config_field="",
            endpoint=endpoint,
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
                blocker=FailureDescriptor(
                    code="chain_rpc.failure.endpoint_invalid",
                    arguments={"field": "custom_rpc_endpoint"},
                    source=__name__,
                ),
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

        promoted = _promote_case2_endpoint(state, responses=[])

        self.assertFalse(promoted)
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertEqual(
            state["chain_identity"]["status"],
            "existing_family_needs_endpoint",
        )

    def test_case2_promotion_rejects_method_validated_on_another_endpoint(self) -> None:
        from agent.harness.domains.chain_handoff import _promote_case2_endpoint
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt

        endpoint = "https://current.invalid/rpc"
        state = new_state("case2-method-endpoint-boundary", language="en")
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
                    "candidate_endpoint": endpoint,
                    "candidate_endpoint_ready": True,
                    "new_chain_endpoint_probe": {
                        "ready": True,
                        "endpoint": endpoint,
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
                                "method": "eth_oldMethod",
                                "params": [],
                                "validation_endpoint": "https://old.invalid/rpc",
                                "evidence_file": "/tmp/old-probe.json",
                            }
                        ],
                        "finished": True,
                    }
                },
                "current_action": {"action_id": "candidate-endpoint-action"},
                "turn_context": {
                    "admitted_actions": [
                        {
                            "action_id": "candidate-endpoint-action",
                            "argument_value_hashes": {
                                "selected_value": exact_value_hash(endpoint),
                            },
                        }
                    ]
                },
            }
        )
        emit_endpoint_role_receipt(
            state,
            role="validation",
            case="new_chain",
            config_field="",
            endpoint=endpoint,
            ready=True,
            probe_status="ok",
            chain="flow-evm",
            adapter_family="jsonrpc",
        )
        validation_receipt = _receipts(state, "rpc_endpoint_role")[-1]
        state["endpoint_evidence"].update(
            {
                "candidate_endpoint_validation_receipt_id": validation_receipt[
                    "receipt_id"
                ],
                "candidate_endpoint_validation_receipt": validation_receipt,
            }
        )

        promoted = _promote_case2_endpoint(state, responses=[])

        self.assertFalse(promoted)
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertEqual(
            state["chain_identity"]["status"],
            "existing_family_needs_method",
        )

    def test_endpoint_owner_records_roles_without_endpoint_or_auth_material(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer

        state = new_state("endpoint-receipts")
        state["turn_index"] = 4
        state["turn_context"] = {
            "text": "endpoint evidence turn",
            "admitted_actions": [{
                "action_id": "endpoint-action",
                "argument_value_hashes": {
                    "selected_value": exact_value_hash(
                        "https://node.invalid/v1/private-api-key"
                    ),
                },
            }],
        }
        state["current_action"] = {"action_id": "endpoint-action"}
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
                {
                    "endpoint_role": "validation",
                    "rpc_case": "custom_rpc",
                    "config_field": "",
                },
                secret_validation_endpoint,
                responses=[],
            )

        validation = _receipts(state, "rpc_endpoint_role")[-1]
        unsigned = dict(validation)
        receipt_id = unsigned.pop("receipt_id")
        self.assertEqual(receipt_id, evidence_hash(unsigned))
        self.assertEqual(validation["role"], "validation")
        self.assertEqual(validation["case"], "custom_rpc")
        self.assertEqual(validation["source_kind"], "direct_action")
        self.assertEqual(validation["source_receipt_id"], "")
        self.assertEqual(
            validation["source_value_hash"],
            exact_value_hash(secret_validation_endpoint),
        )
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
        state["workload"] = {
            "confirmed": True,
            "choice": "custom_rpc",
            "methods": ["demo_private"],
            "job_local_override": True,
        }
        secret_final_endpoint = "https://final.invalid/bearer-secret"
        state["current_action"] = {"action_id": "final-endpoint-action"}
        state["turn_context"]["admitted_actions"].append({
            "action_id": "final-endpoint-action",
            "argument_value_hashes": {
                "selected_value": exact_value_hash(secret_final_endpoint),
            },
        })
        final_probe = self._probe_result(
            chain="bsc",
            endpoint=secret_final_endpoint,
            method="demo_private",
            params=[],
        )
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value=final_probe,
        ):
            _apply_endpoint_answer(
                state,
                {
                    "endpoint_role": "final_benchmark",
                    "rpc_case": "runtime",
                    "config_field": "LOCAL_RPC_URL",
                },
                secret_final_endpoint,
                responses=[],
            )

        final = _receipts(state, "rpc_endpoint_role")[-1]
        self.assertEqual(final["role"], "final_benchmark")
        self.assertEqual(final["source_kind"], "direct_action")
        self.assertEqual(final["source_receipt_id"], "")
        self.assertEqual(final["methods"], ["demo_private"])
        self.assertNotIn(secret_final_endpoint, json.dumps(final, sort_keys=True))
        self.assertNotEqual(validation["endpoint_hash"], final["endpoint_hash"])

        promoted = deepcopy(final)
        promoted["source_kind"] = "validated_endpoint"
        promoted["source_receipt_id"] = ""
        promoted["receipt_id"] = evidence_hash({
            key: value
            for key, value in promoted.items()
            if key != "receipt_id"
        })
        valid, reason = validate_rpc_receipt(promoted)
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid promoted runtime endpoint source")

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
        state["current_action"] = {"action_id": "schema-evidence-action"}
        full_user_input = f"Please inspect this request:\n{request}"
        state["turn_context"] = {
            "text": "schema evidence turn",
            "admitted_actions": [{
                "action_id": "schema-evidence-action",
                "argument_value_hashes": {
                    "source_evidence": exact_value_hash(full_user_input),
                    "rpc_schema_evidence": exact_value_hash(request),
                },
            }],
        }
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
                    responses=[],
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
        self.assertEqual(
            provenance["source_action_value_hashes"],
            [exact_value_hash(request)],
        )
        self.assertNotIn(
            exact_value_hash(full_user_input),
            provenance["source_action_value_hashes"],
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

        _apply_scope(
            state,
            "custom_rpc_scope",
            "single_replace",
            responses=[],
        )

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
        source_evidence = (
            '{"jsonrpc":"2.0","method":"demo_health","params":[]}'
        )
        state["current_action"] = {"action_id": "schema-source-action"}
        state["turn_context"] = {
            "text": "probe and accept method",
            "admitted_actions": [{
                "action_id": "schema-source-action",
                "argument_value_hashes": {
                    "source_evidence": exact_value_hash(source_evidence),
                },
            }],
        }
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
                "validation_endpoint": "https://probe.invalid/rpc",
                "evidence": [{
                    "revision": 1,
                    "source": "user",
                    "kind": "protocol_request",
                    "content": source_evidence,
                    "method": "demo_health",
                    "source_action_value_hash": exact_value_hash(
                        source_evidence
                    ),
                }],
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
        probe = self._probe_result(
            chain="bsc",
            endpoint="https://probe.invalid/rpc",
            method="demo_health",
            params=[],
        )
        evidence_file = probe["evidence_file"]
        observed = {
            "shape_hash": "shape-secret",
            "sample": '{"result":"private-response"}',
            "http_status": 200,
            "evidence_file": evidence_file,
        }
        self.assertTrue(record_probe(state, probe, observed).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)
        self.assertTrue(
            add_validated_method(state).accepted
        )

        provenance = _receipts(state, "rpc_schema_provenance")[-1]
        paths = {item["field_path"] for item in provenance["fields"]}
        self.assertIn("observed_response.sample", paths)
        self.assertNotIn("private-response", json.dumps(provenance, sort_keys=True))
        transition = _receipts(state, "rpc_catalog_transition")[-1]
        self.assertEqual(transition["command"], "add_method")
        self.assertEqual(transition["method_count"], 1)
        self.assertEqual(transition["method_names"], ["demo_health"])
        [method_probe] = _receipts(state, "rpc_method_probe")
        self.assertTrue(validate_rpc_receipt(method_probe)[0])

    def test_corrected_rpc_contract_cannot_reuse_a_prior_successful_probe(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            confirm_parameter,
            confirm_request,
            confirm_response,
            correct_draft,
            draft_view,
            record_probe,
        )

        state = new_state("stale-probe-rejected")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["current_action"] = {
            "action_id": "rpc-contract-action",
        }
        original = {
            "method": "demo_lookup",
            "params_json": ["0xold"],
            "params": [{
                "index": 0,
                "name": "account",
                "json_type": "string",
                "semantic_type": "account_address",
                "encoding": "hex",
                "meaning": "account to query",
                "required": True,
                "example": "0xold",
            }],
            "response_summary": "hex balance",
            "validation_endpoint": "https://probe.invalid/rpc",
        }
        self.assertTrue(correct_draft(state, original).accepted)
        self.assertTrue(confirm_parameter(state, 0, True).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        probe = self._probe_result(
            chain="bsc",
            endpoint="https://probe.invalid/rpc",
            method="demo_lookup",
            params=["0xold"],
        )
        self.assertTrue(record_probe(
            state,
            probe,
            {
                "shape_hash": "old-shape",
                "evidence_file": probe["evidence_file"],
            },
        ).accepted)
        self.assertTrue(
            confirm_response(state, True, observed=True).accepted
        )

        corrected = {
            **draft_view(state),
            "params_json": ["0xnew"],
            "params": [{
                **original["params"][0],
                "example": "0xnew",
            }],
        }
        self.assertTrue(correct_draft(state, corrected).accepted)
        current = draft_view(state)
        self.assertNotIn("probe", current)
        self.assertNotIn("observed_response", current)
        self.assertTrue(confirm_parameter(state, 0, True).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        self.assertTrue(confirm_response(state, True).accepted)

        rejected = add_validated_method(state)
        self.assertFalse(rejected.accepted)
        self.assertEqual(draft_view(state).get("phase"), "probe_confirmation")

    def test_validated_method_uses_current_draft_not_caller_contract(self) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
            validated_contracts_view,
        )

        state = new_state("catalog-draft-authority")
        state["current_action"] = {"action_id": "catalog-authority-action"}
        state["turn_context"] = {"control_receipts": []}
        self.assertTrue(correct_draft(state, {
            "method": "demo_original",
            "params": [],
            "params_json": [],
            "response_summary": "hex status",
            "validation_endpoint": "https://probe.invalid/rpc",
        }).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        probe = self._probe_result(
            chain="bsc",
            endpoint="https://probe.invalid/rpc",
            method="demo_original",
            params=[],
        )
        evidence_file = probe["evidence_file"]
        self.assertTrue(record_probe(
            state,
            probe,
            {
                "shape_hash": "original-shape",
                "evidence_file": evidence_file,
            },
        ).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)

        self.assertTrue(add_validated_method(state).accepted)
        [contract] = validated_contracts_view(state)
        self.assertEqual(contract["method"], "demo_original")
        self.assertEqual(contract["params"], [])
        self.assertEqual(
            contract["observed_response"],
            {
                "shape_hash": "original-shape",
                "evidence_file": evidence_file,
            },
        )

    def test_probe_rejects_durable_evidence_for_another_request(self) -> None:
        from agent.harness.domains.rpc_catalog import (
            correct_draft,
            draft_view,
            record_probe,
        )
        from agent.validators.endpoint_probe import (
            build_rpc_probe_contract,
            rpc_probe_contract_hash,
        )

        endpoint = "https://probe.invalid/rpc"
        state = new_state("probe-evidence-request-authority")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        self.assertTrue(correct_draft(state, {
            "method": "demo_current",
            "params": [],
            "params_json": [],
            "validation_endpoint": endpoint,
        }).accepted)
        unrelated = self._probe_result(
            chain="bsc",
            endpoint=endpoint,
            method="demo_other",
            params=["0x1"],
        )
        expected_contract = build_rpc_probe_contract(
            chain="bsc",
            endpoint=endpoint,
            transport="jsonrpc",
            methods=["demo_current"],
            method_params={"demo_current": []},
        )
        forged_result = {
            **unrelated,
            "probe_contract": expected_contract,
            "probe_contract_hash": rpc_probe_contract_hash(
                expected_contract
            ),
        }

        transition = record_probe(state, forged_result, {})

        self.assertFalse(transition.accepted)
        self.assertFalse((draft_view(state).get("probe") or {}).get("ready"))
        self.assertFalse(_receipts(state, "rpc_method_probe"))

    def test_validated_method_is_rejected_after_parameter_mutation(self) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
            validated_contracts_view,
            validated_method_contract_is_current,
        )

        endpoint = "https://probe.invalid/rpc"
        state = new_state("validated-method-parameter-authority")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["current_action"] = {
            "action_id": "validated-parameter-action",
        }
        state["turn_context"] = {"control_receipts": []}
        self.assertTrue(correct_draft(state, {
            "method": "demo_params",
            "params": [],
            "params_json": [],
            "response_summary": "hex status",
            "validation_endpoint": endpoint,
        }).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        probe = self._probe_result(
            chain="bsc",
            endpoint=endpoint,
            method="demo_params",
            params=[],
        )
        self.assertTrue(record_probe(
            state,
            probe,
            {
                "shape_hash": "parameter-shape",
                "evidence_file": probe["evidence_file"],
            },
        ).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)
        self.assertTrue(add_validated_method(state).accepted)
        [contract] = validated_contracts_view(state)
        self.assertTrue(validated_method_contract_is_current(
            state,
            contract,
            endpoint=endpoint,
        ))

        state["chain_identity"]["adapter_family"] = "tendermint"
        self.assertFalse(validated_method_contract_is_current(
            state,
            contract,
            endpoint=endpoint,
        ))
        state["chain_identity"]["adapter_family"] = "jsonrpc"

        state["target_mode"] = "real-node"
        state["confirmed_config"]["LOCAL_RPC_URL"] = endpoint
        self.assertFalse(validated_method_contract_is_current(
            state,
            contract,
        ))
        state["target_mode"] = "fake-node"
        state["confirmed_config"].pop("LOCAL_RPC_URL", None)

        contract["params"] = ["0x1"]
        contract["schema"]["params_json"] = ["0x1"]

        self.assertFalse(validated_method_contract_is_current(
            state,
            contract,
            endpoint=endpoint,
        ))

    def test_final_endpoint_replay_proof_authorizes_real_node_contract(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
            validated_contracts_view,
            validated_method_contract_is_current,
        )
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer

        validation_endpoint = "https://validation.invalid/rpc"
        final_endpoint = "https://final.invalid/rpc"
        method = "demo_final"
        state = new_state("final-endpoint-proof-authority")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["current_action"] = {"action_id": "validation-action"}
        state["turn_context"] = {"control_receipts": []}
        self.assertTrue(correct_draft(state, {
            "method": method,
            "params": [],
            "params_json": [],
            "response_summary": "hex status",
            "validation_endpoint": validation_endpoint,
        }).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        validation_probe = self._probe_result(
            chain="bsc",
            endpoint=validation_endpoint,
            method=method,
            params=[],
        )
        self.assertTrue(record_probe(
            state,
            validation_probe,
            {
                "shape_hash": "validation-shape",
                "evidence_file": validation_probe["evidence_file"],
            },
        ).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)
        self.assertTrue(add_validated_method(state).accepted)
        state.update({
            "target_mode": "real-node",
            "rpc_mode": "single",
            "workload": {
                "confirmed": True,
                "job_local_override": True,
                "methods": [method],
                "replace_defaults": True,
            },
        })
        state["current_action"] = {"action_id": "final-endpoint-action"}
        state["turn_context"] = {
            "control_receipts": [],
            "admitted_actions": [{
                "action_id": "final-endpoint-action",
                "argument_value_hashes": {
                    "selected_value": exact_value_hash(final_endpoint),
                },
            }],
        }
        final_probe = self._probe_result(
            chain="bsc",
            endpoint=final_endpoint,
            method=method,
            params=[],
        )
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value=final_probe,
        ):
            _apply_endpoint_answer(
                state,
                {
                    "endpoint_role": "final_benchmark",
                    "rpc_case": "runtime",
                    "config_field": "LOCAL_RPC_URL",
                },
                final_endpoint,
                responses=[],
            )

        [contract] = validated_contracts_view(state)
        self.assertTrue(validated_method_contract_is_current(
            state,
            contract,
        ))
        receipt = (
            state["endpoint_evidence"][
                "local_rpc_url_validation_receipt"
            ]
        )
        self.assertTrue(validate_rpc_receipt(receipt)[0])
        self.assertEqual(
            receipt["method_evidence_bindings"][0]["method"],
            method,
        )

    def test_observed_response_confirmation_requires_probe_observation(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import (
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
        )

        endpoint = "https://probe.invalid/rpc"
        state = new_state("observed-response-proof-authority")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["current_action"] = {"action_id": "observed-response-action"}
        state["turn_context"] = {"control_receipts": []}
        self.assertTrue(correct_draft(state, {
            "method": "demo_observed",
            "params": [],
            "params_json": [],
            "validation_endpoint": endpoint,
        }).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        probe = self._probe_result(
            chain="bsc",
            endpoint=endpoint,
            method="demo_observed",
            params=[],
        )
        self.assertTrue(record_probe(state, probe, {}).accepted)

        self.assertFalse(
            confirm_response(state, True, observed=True).accepted
        )

    def test_case2_promotion_rejects_self_asserted_current_endpoint_method(
        self,
    ) -> None:
        from agent.harness.domains.chain_handoff import _promote_case2_endpoint
        from agent.harness.domains.rpc_catalog import request_contract_hash
        from agent.harness.domains.rpc_receipts import emit_endpoint_role_receipt

        endpoint = "https://current.invalid/rpc"
        method = "eth_fakeProof"
        evidence_file = "/tmp/does-not-exist.json"
        schema = {
            "method": method,
            "params": [],
            "params_json": [],
            "validation_endpoint": endpoint,
        }
        probe_body = {
            "receipt_type": "rpc_method_probe",
            "receipt_version": 2,
            "turn_index": 1,
            "owner": "rpc_catalog",
            "method": method,
            "method_hash": evidence_hash(method),
            "endpoint_hash": evidence_hash(endpoint),
            "request_contract_hash": request_contract_hash(schema),
            "catalog_revision": 123,
            "evidence_file_hash": evidence_hash(evidence_file),
            "probe_contract_hash": "e" * 64,
            "ready": True,
            "probe_status_hash": evidence_hash("ok"),
            "producer_action_id": "forged-probe-action",
        }
        forged_probe_receipt = {
            **probe_body,
            "receipt_id": evidence_hash(probe_body),
        }
        self.assertTrue(validate_rpc_receipt(forged_probe_receipt)[0])
        state = new_state("case2-self-asserted-method", language="en")
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
                    "candidate_endpoint": endpoint,
                    "candidate_endpoint_ready": True,
                    "new_chain_endpoint_probe": {
                        "ready": True,
                        "endpoint": endpoint,
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
                                "method": method,
                                "params": [],
                                "schema": schema,
                                "validation_endpoint": endpoint,
                                "evidence_file": evidence_file,
                                "probe_receipt": forged_probe_receipt,
                            }
                        ],
                        "finished": True,
                    }
                },
                "current_action": {"action_id": "candidate-endpoint-action"},
                "turn_context": {
                    "admitted_actions": [
                        {
                            "action_id": "candidate-endpoint-action",
                            "argument_value_hashes": {
                                "selected_value": exact_value_hash(endpoint),
                            },
                        }
                    ]
                },
            }
        )
        emit_endpoint_role_receipt(
            state,
            role="validation",
            case="new_chain",
            config_field="",
            endpoint=endpoint,
            ready=True,
            probe_status="ok",
            chain="flow-evm",
            adapter_family="jsonrpc",
        )
        validation_receipt = _receipts(state, "rpc_endpoint_role")[-1]
        state["endpoint_evidence"].update(
            {
                "candidate_endpoint_validation_receipt_id":
                validation_receipt["receipt_id"],
                "candidate_endpoint_validation_receipt":
                validation_receipt,
            }
        )

        self.assertFalse(_promote_case2_endpoint(state, responses=[]))
        self.assertEqual(
            state["chain_identity"]["status"],
            "existing_family_needs_method",
        )

    def test_schema_receipt_preserves_each_evidence_action_pair(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import correct_draft

        def receipt_for(first_hash: str, second_hash: str) -> dict:
            state = new_state(f"binding-{first_hash[:4]}-{second_hash[:4]}")
            state["current_action"] = {"action_id": "schema-action"}
            state["turn_context"] = {"control_receipts": []}
            correct_draft(
                state,
                {
                    "method": "demo_pair",
                    "params": [],
                    "params_json": [],
                    "evidence": [
                        {
                            "revision": 1,
                            "source": "user",
                            "kind": "request",
                            "content": "request evidence",
                            "method": "demo_pair",
                            "source_action_value_hash": first_hash,
                        },
                        {
                            "revision": 2,
                            "source": "user",
                            "kind": "response",
                            "content": "response evidence",
                            "method": "demo_pair",
                            "source_action_value_hash": second_hash,
                        },
                    ],
                    "field_provenance": [
                        {
                            "field_path": "method",
                            "source_kind": "protocol_parser",
                            "source_revisions": [1, 2],
                            "value_hash": evidence_hash("demo_pair"),
                        }
                    ],
                },
            )
            return _receipts(state, "rpc_schema_provenance")[-1]

        first = "a" * 64
        second = "b" * 64
        original = receipt_for(first, second)
        swapped = receipt_for(second, first)

        self.assertNotEqual(
            original["source_bindings"],
            swapped["source_bindings"],
        )
        self.assertNotEqual(original["receipt_id"], swapped["receipt_id"])

    def test_schema_receipt_rejects_reused_admitted_input_binding(
        self,
    ) -> None:
        shared = "a" * 64
        body = {
            "receipt_type": "rpc_schema_provenance",
            "receipt_version": 2,
            "turn_index": 1,
            "owner": "rpc_catalog",
            "method": "demo_pair",
            "method_hash": evidence_hash("demo_pair"),
            "catalog_revision": 2,
            "fields": [{
                "field_path": "method",
                "source_kind": "protocol_parser",
                "source_revisions": [1, 2],
                "value_hash": evidence_hash("demo_pair"),
            }],
            "source_evidence_hashes": ["1" * 64, "2" * 64],
            "source_action_value_hashes": [shared],
            "source_bindings": [
                {
                    "evidence_hash": "1" * 64,
                    "action_value_hash": shared,
                },
                {
                    "evidence_hash": "2" * 64,
                    "action_value_hash": shared,
                },
            ],
            "producer_action_id": "schema-action",
        }
        body["fields_hash"] = evidence_hash(body["fields"])
        body["receipt_id"] = evidence_hash(body)

        valid, reason = validate_rpc_receipt(body)

        self.assertFalse(valid)
        self.assertEqual(
            reason,
            "RPC admitted input binding is not one-to-one",
        )

    def test_append_evidence_rejects_partial_current_action_binding(
        self,
    ) -> None:
        from agent.harness.domains.rpc_catalog import (
            append_evidence,
            correct_draft,
            draft_view,
        )

        first = "official request one"
        state = new_state("partial-evidence-binding")
        state["current_action"] = {"action_id": "evidence-action"}
        state["turn_context"] = {
            "admitted_actions": [
                {
                    "action_id": "evidence-action",
                    "argument_value_hashes": {
                        "source_evidence": exact_value_hash(first),
                    },
                }
            ]
        }
        self.assertTrue(
            correct_draft(
                state,
                {
                    "method": "demo_method",
                    "params": [],
                    "params_json": [],
                },
            ).accepted
        )
        self.assertTrue(
            append_evidence(
                state,
                content=first,
                source="user",
                kind="request",
            ).accepted
        )
        rejected = append_evidence(
            state,
            content="different unbound evidence",
            source="user",
            kind="response",
        )

        self.assertFalse(rejected.accepted)
        self.assertEqual(len(draft_view(state).get("evidence") or ()), 1)

    def test_catalog_change_after_probe_invalidates_admission(self) -> None:
        from agent.harness.domains.rpc_catalog import (
            add_validated_method,
            append_evidence,
            confirm_request,
            confirm_response,
            correct_draft,
            record_probe,
        )

        state = new_state("catalog-revision-authority")
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
        }
        state["current_action"] = {"action_id": "catalog-revision-action"}
        self.assertTrue(correct_draft(state, {
            "method": "demo_revision",
            "params": [],
            "params_json": [],
            "response_summary": "hex status",
            "validation_endpoint": "https://probe.invalid/rpc",
        }).accepted)
        self.assertTrue(confirm_request(state, True).accepted)
        probe = self._probe_result(
            chain="bsc",
            endpoint="https://probe.invalid/rpc",
            method="demo_revision",
            params=[],
        )
        self.assertTrue(record_probe(
            state,
            probe,
            {
                "shape_hash": "revision-shape",
                "evidence_file": probe["evidence_file"],
            },
        ).accepted)
        self.assertTrue(confirm_response(state, True, observed=True).accepted)
        self.assertTrue(append_evidence(
            state,
            content="later official response documentation",
            source="user",
            kind="documentation",
            method="demo_revision",
        ).accepted)

        self.assertFalse(add_validated_method(state).accepted)

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
            def prepared_with_runtime_override(**kwargs):
                return {
                    "status": "blocked",
                    "warnings": ["chain_template_exists"],
                    "data": {
                        "plan": {
                            "chain": "case2-receipt",
                            "artifacts": {},
                            "chain_config_override": deepcopy(
                                kwargs["chain_config_override"]
                            ),
                        },
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
                    "agent.harness.domains.execution_runtime.validated_method_contract_is_current",
                    return_value=True,
                ),
                patch(
                    "agent.harness.domains.rpc_catalog.validated_method_contract_is_current",
                    return_value=True,
                ),
                patch(
                    "agent.runners.application_service.prepare_benchmark_run",
                    side_effect=prepared_with_runtime_override,
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
