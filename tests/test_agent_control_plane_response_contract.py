from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

from agent.harness.contracts import FailureDescriptor, ResponseFragment
from agent.harness.control_receipts import validate_coordinator_control_receipt
from agent.harness.failures import (
    build_failure_record,
    failure_record_response_fragment,
    failure_response_fragment,
)
from agent.harness.response_catalog import MESSAGE_CATALOG, render_fragment
from agent.harness.state import STATE_SCHEMA_VERSION, migrate_state, new_state


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "agent" / "harness" / "coordinator.py"


def _signed_receipt(payload: dict[str, object]) -> dict[str, object]:
    receipt = dict(payload)
    receipt["receipt_id"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _manifest(role: str = "error") -> dict[str, str]:
    return {
        "semantic_hash": "1" * 64,
        "render_hash": "2" * 64,
        "message_id": "harness.failure.internal_contract_violation",
        "role": role,
    }


class ControlPlaneResponseContractTest(unittest.TestCase):
    def test_hierarchical_planner_stage_receipts_are_registered_and_strict(self) -> None:
        receipts = (
            {
                "receipt_type": "semantic_partition",
                "turn_index": 3,
                "status": "compile_owner",
                "unit_count": 2,
                "owner_count": 1,
                "stage_a_calls": 1,
                "errors_hash": "1" * 64,
            },
            {
                "receipt_type": "owner_compilation",
                "turn_index": 3,
                "owner": "chain_rpc",
                "cursor_before": 0,
                "cursor_after": 1,
                "status": "review_plan",
                "document_hash": "2" * 64,
                "errors_hash": "3" * 64,
            },
            {
                "receipt_type": "whole_plan_review",
                "turn_index": 3,
                "status": "reviewed",
                "result_hash": "4" * 64,
                "admission_calls": 1,
            },
        )
        for payload in receipts:
            with self.subTest(receipt_type=payload["receipt_type"]):
                receipt = _signed_receipt(payload)
                self.assertEqual(
                    validate_coordinator_control_receipt(
                        receipt,
                        turn_index=3,
                    ),
                    (True, ""),
                )

                malformed = dict(payload)
                malformed["unexpected"] = True
                valid, _ = validate_coordinator_control_receipt(
                    _signed_receipt(malformed),
                    turn_index=3,
                )
                self.assertFalse(valid)

    def test_every_registered_control_message_renders_in_both_languages(self) -> None:
        value_by_type = {
            "string": "value",
            "integer": 1,
            "number": 1.5,
            "boolean": True,
        }
        for message_id, definition in MESSAGE_CATALOG.items():
            if not message_id.startswith("harness."):
                continue
            arguments = {
                name: value_by_type[argument_type]
                for name, argument_type in definition.arguments.items()
            }
            for kind in definition.kinds:
                fragment = ResponseFragment(
                    kind=kind,
                    message_id=message_id,
                    arguments=arguments,
                    source="test",
                )
                for language in ("en", "zh"):
                    with self.subTest(
                        message_id=message_id,
                        kind=kind,
                        language=language,
                    ):
                        self.assertTrue(render_fragment(fragment, language).text)

    def test_coordinator_has_no_localized_or_legacy_response_authority(self) -> None:
        source = COORDINATOR.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(COORDINATOR))

        self.assertNotIn("_localized", source)
        self.assertNotIn("append_response_messages", source)
        self.assertNotIn("replace_response_messages", source)
        self.assertNotIn("message_fragments", source)
        self.assertNotIn("render_failure_summary", source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if called != "HandlerResult":
                continue
            for keyword in node.keywords:
                if keyword.arg == "blocker":
                    self.assertIsInstance(keyword.value, ast.Call)
                    self.assertIsInstance(keyword.value.func, ast.Name)
                    self.assertEqual(keyword.value.func.id, "_failure")

    def test_failure_records_project_through_central_catalog(self) -> None:
        direct = failure_response_fragment(
            FailureDescriptor(
                code="harness.failure.coordinator.unsupported_action",
                arguments={"action_type": "unknown"},
                payload={"action_id": "a1"},
                source="test",
            )
        )
        rendered_direct = render_fragment(direct, "en")
        self.assertIn("unknown", rendered_direct.text)

        record = build_failure_record(
            "WORKLOAD_PROCESS_FAILED",
            source="workload",
            severity="blocking",
            facts=[{"detail": "exit 2"}],
            evidence_paths=["/workspace/benchmark.log"],
            confirmed_config={"CLOUD_REGION": "us-test1"},
        )
        rendered_record = render_fragment(
            failure_record_response_fragment(record),
            "zh",
        )
        self.assertIn("WORKLOAD_PROCESS_FAILED", rendered_record.text)
        self.assertIn("exit 2", rendered_record.text)
        self.assertIn("/workspace/benchmark.log", rendered_record.text)

    def test_response_composition_receipt_uses_dual_hash_manifest(self) -> None:
        receipt = _signed_receipt({
            "receipt_type": "response_composition",
            "turn_index": 4,
            "language": "en",
            "active_group": "opening",
            "source_action_ids": ["a1"],
            "pending_contract_hash": "3" * 64,
            "fragments": [_manifest()],
        })

        self.assertEqual(
            validate_coordinator_control_receipt(receipt, turn_index=4),
            (True, ""),
        )

        legacy = dict(receipt)
        legacy["fragments"] = [
            {"fragment_hash": "1" * 64, "role": "visible_result"}
        ]
        legacy = _signed_receipt({
            key: value for key, value in legacy.items() if key != "receipt_id"
        })
        valid, _ = validate_coordinator_control_receipt(legacy, turn_index=4)
        self.assertFalse(valid)

    def test_rejected_domain_receipt_uses_typed_failure_and_manifest(self) -> None:
        receipt = _signed_receipt({
            "receipt_type": "domain_commit",
            "turn_index": 5,
            "owner": "coordinator",
            "completion": "rejected",
            "group_registry_contract_hash": "4" * 64,
            "pending_before_hash": "5" * 64,
            "pending_after_hash": "6" * 64,
            "consumed_action_ids": [],
            "invalidated_groups": [],
            "invalidated_fields": [],
            "response_fragments": [_manifest()],
            "blocker_semantic_hash": "7" * 64,
        })

        self.assertEqual(
            validate_coordinator_control_receipt(receipt, turn_index=5),
            (True, ""),
        )

    def test_v17_migration_clears_retired_turn_local_response_state(self) -> None:
        old = new_state("response-v17", language="en")
        old["schema_version"] = 17
        old["confirmed_config"] = {"CLOUD_REGION": "us-test1"}
        old["response_fragments"] = [
            {
                "kind": "message",
                "content": "retired text",
                "source": "legacy",
                "metadata": {},
            }
        ]
        old["visible_response"] = ["retired terminal text"]
        old["turn_context"] = {
            "text": "current user input",
            "response_manifest": [
                {"fragment_hash": "8" * 64, "role": "visible_result"}
            ],
            "terminal_response_hash": "9" * 64,
            "terminal_semantic_hash": "a" * 64,
        }

        migrated = migrate_state(
            old,
            thread_id="response-v17",
            language="en",
            session_purpose="user",
        )

        self.assertEqual(STATE_SCHEMA_VERSION, 18)
        self.assertEqual(migrated["schema_version"], 18)
        self.assertEqual(migrated["response_fragments"], [])
        self.assertEqual(migrated["visible_response"], [])
        self.assertEqual(migrated["turn_context"], {"text": "current user input"})
        self.assertEqual(
            migrated["confirmed_config"],
            {"CLOUD_REGION": "us-test1"},
        )


if __name__ == "__main__":
    unittest.main()
