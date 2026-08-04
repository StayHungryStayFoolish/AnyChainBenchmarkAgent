"""Direct contracts for durable Harness action ordering."""

from __future__ import annotations

import unittest

from agent.harness.queue import (
    ActionQueueConflict,
    action_can_run_while_pending,
    action_supersedes_pending_contract,
    order_action_queue,
)
from agent.harness.state import new_state


def _configured_state() -> dict:
    state = new_state("action-queue-ordering")
    state["target_mode"] = "fake-node"
    state["workflow_mode"] = "rpc_benchmark"
    state["chain_identity"] = {
        "raw": "solana",
        "canonical": "solana",
        "status": "confirmed",
        "case": "known",
    }
    return state


def _types(actions: list[dict]) -> list[str]:
    return [str(action.get("type") or "") for action in actions]


class ActionQueueOrderingTests(unittest.TestCase):
    def test_upstream_protocol_change_crosses_stale_pending_barrier(self) -> None:
        state = _configured_state()
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "queue_barrier": True,
            "barrier_policy": "explicit_detour_only",
        }
        action = {
            "type": "choose_adapter_family",
            "adapter_family": "jsonrpc",
            "_origin_text": "Change the protocol family, then validate the method.",
            "_plan_transaction_hash": "reviewed-plan",
        }

        self.assertTrue(action_supersedes_pending_contract(state, action))
        self.assertTrue(action_can_run_while_pending(state, action))

    def test_exact_target_mode_replacement_precedes_same_turn_qps(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "_plan_index": 0,
                "_submitted_turn_index": 7,
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "_plan_index": 1,
                "_submitted_turn_index": 7,
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["choose_target_mode", "set_qps_mode"],
        )

    def test_isomorphic_chain_replacement_precedes_same_turn_workload(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "_plan_index": 0,
                "_plan_scope": "turn-8",
            },
            {
                "type": "choose_chain",
                "chain_text": "ethereum",
                "_plan_index": 1,
                "_plan_scope": "turn-8",
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["choose_chain", "set_rpc_mode"],
        )

    def test_catalog_intake_precedes_same_transaction_workload_replacement(
        self,
    ) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_rpc_mode",
                "rpc_mode": "single",
                "_plan_index": 0,
                "_plan_scope": "structured-custom-rpc",
            },
            {
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": "true",
                "_plan_index": 1,
                "_plan_scope": "structured-custom-rpc",
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["rpc_catalog_command", "set_rpc_mode"],
        )

    def test_negative_independent_mutations_preserve_semantic_order(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_observability",
                "observability_mode": "local",
                "_plan_index": 0,
                "_submitted_turn_index": 9,
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "_plan_index": 1,
                "_submitted_turn_index": 9,
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["set_observability", "set_qps_mode"],
        )

    def test_neighboring_prior_turn_qps_is_not_reinterpreted_as_same_transaction(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "_plan_index": 0,
                "_submitted_turn_index": 10,
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "_plan_index": 0,
                "_submitted_turn_index": 11,
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["choose_target_mode", "set_qps_mode"],
        )
        self.assertEqual(
            [action["_submitted_turn_index"] for action in ordered],
            [11, 10],
        )

    def test_neighboring_missing_capability_keeps_cross_turn_provider_dependency(self) -> None:
        state = new_state("cross-turn-capability")
        actions = [
            {
                "type": "request_chain_selection",
                "chain_candidates": ["ethereum"],
                "_plan_index": 0,
                "_submitted_turn_index": 11,
            },
            {
                "type": "request_target_mode_selection",
                "_plan_index": 0,
                "_submitted_turn_index": 10,
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["request_target_mode_selection", "request_chain_selection"],
        )

    def test_conflicting_same_turn_typed_mutations_fail_atomically(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "_plan_index": 0,
                "_plan_scope": "turn-12",
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "_plan_index": 1,
                "_plan_scope": "turn-12",
            },
        ]

        with self.assertRaisesRegex(
            ActionQueueConflict,
            "requires explicit clarification",
        ):
            order_action_queue(state, actions)

    def test_neighboring_qps_mode_and_override_are_composable(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "_plan_index": 0,
                "_plan_scope": "turn-13",
            },
            {
                "type": "set_qps_override",
                "qps_overrides": {"initial_qps": 5},
                "_plan_index": 1,
                "_plan_scope": "turn-13",
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["set_qps_mode", "set_qps_override"],
        )

    def test_neighboring_target_replacement_precedes_custom_rpc_catalog_intake(self) -> None:
        state = _configured_state()
        actions = [
            {
                "type": "rpc_catalog_command",
                "catalog_command": "enter",
                "source_evidence": "add a custom RPC method",
                "_plan_index": 0,
                "_plan_scope": "turn-14",
            },
            {
                "type": "choose_target_mode",
                "target_mode": "real-node",
                "_plan_index": 1,
                "_plan_scope": "turn-14",
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["choose_target_mode", "rpc_catalog_command"],
        )

    def test_current_pending_settlement_precedes_same_turn_sibling_mutation(self) -> None:
        state = _configured_state()
        state["pending_question"] = {
            "id": "preflight_smoke_confirm",
            "group": "preflight_smoke_execution",
            "accepted_action_types": [
                "approve_preflight_smoke",
                "reject_preflight_smoke",
            ],
        }
        actions = [
            {
                "type": "request_qps_customization",
                "source_evidence": "change QPS",
                "_plan_index": 0,
                "_plan_scope": "turn-15",
            },
            {
                "type": "reject_preflight_smoke",
                "_plan_index": 1,
                "_plan_scope": "turn-15",
            },
        ]

        ordered = order_action_queue(state, actions)

        self.assertEqual(
            _types(ordered),
            ["reject_preflight_smoke", "request_qps_customization"],
        )


if __name__ == "__main__":
    unittest.main()
