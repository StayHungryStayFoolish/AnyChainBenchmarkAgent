"""Contracts for the external Codex simulator bridge."""

from __future__ import annotations

import io
import json
import unittest

from tests.agent_live.chaos_scheduler import ScheduledCoverageTarget
from tests.agent_live.codex_simulator_bridge import (
    CONTEXT_FRAME,
    DECISION_FRAME,
    StdioCodexSimulator,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.dynamic_dual_ai_chaos import SimulatorContext


class StdioCodexSimulatorTest(unittest.TestCase):
    def _context(self) -> SimulatorContext:
        return SimulatorContext(
            session_id="chaos-contract",
            turn_index=2,
            previous_agent_response="Agent> Choose a mode.\n1. fake-node\n2. real-node",
            previous_response_received_at_ns=123,
            scheduled_target=ScheduledCoverageTarget(
                target_id="opening-fake",
                edge_key="opening::fake-node",
                persona="impatient operator",
                goal="choose a safe framework check in natural language",
            ),
            transcript=(),
            coverage_contract={
                "edge_key": "opening::fake-node",
                "input_class": "natural_language_option",
                "action_type": "choose_target_mode",
                "expected_postcondition": {"field": "target_mode"},
            },
        )

    def _decision_frame(self, context: SimulatorContext, **changes: object) -> str:
        payload = {
            "previous_response_hash": content_hash(context.previous_agent_response),
            "user_message": "I just want a safe check without a real node.",
            "persona": context.scheduled_target.persona,
            "goal": context.scheduled_target.goal,
            "rationale": "The complete response offered fake-node as the safe option.",
            "target_coverage_ids": [context.scheduled_target.edge_key],
        }
        payload.update(changes)
        return DECISION_FRAME + json.dumps(payload) + "\n"

    def test_emits_complete_context_before_accepting_one_decision(self) -> None:
        context = self._context()
        output = io.StringIO()

        decision = StdioCodexSimulator(
            io.StringIO(self._decision_frame(context)),
            output,
        )(context)

        emitted = output.getvalue()
        self.assertTrue(emitted.startswith(CONTEXT_FRAME))
        payload = json.loads(emitted[len(CONTEXT_FRAME):])
        self.assertEqual(payload["previous_agent_response"], context.previous_agent_response)
        self.assertEqual(
            payload["previous_response_hash"],
            content_hash(context.previous_agent_response),
        )
        self.assertEqual(payload["scheduled_target"]["edge_key"], "opening::fake-node")
        self.assertEqual(payload["coverage_contract"], context.coverage_contract)
        contract = payload["decision_contract"]
        self.assertEqual(contract["frame_prefix"], DECISION_FRAME)
        self.assertEqual(contract["response_binding_key"], "previous_response_hash")
        self.assertEqual(
            contract["response_binding_value"],
            content_hash(context.previous_agent_response),
        )
        self.assertEqual(contract["immutable_persona"], context.scheduled_target.persona)
        self.assertEqual(contract["immutable_goal"], context.scheduled_target.goal)
        self.assertEqual(
            contract["decision_template"]["previous_response_hash"],
            content_hash(context.previous_agent_response),
        )
        self.assertEqual(
            contract["decision_template"]["target_coverage_ids"],
            [context.scheduled_target.edge_key],
        )
        self.assertIn("coverage_contract.input_class", contract["input_generation_rule"])
        self.assertEqual(
            set(contract["required_keys"]),
            {
                "previous_response_hash",
                "user_message",
                "persona",
                "goal",
                "rationale",
                "target_coverage_ids",
            },
        )
        self.assertEqual(decision.user_message, "I just want a safe check without a real node.")
        self.assertEqual(decision.target_coverage_ids, ("opening::fake-node",))

    def test_structured_config_contract_requires_a_concrete_supplied_value(self) -> None:
        context = self._context()
        context = SimulatorContext(
            **{
                **context.__dict__,
                "coverage_contract": {
                    "input_class": "structured_json_yaml_env_curl",
                    "action_type": "propose_config_values",
                    "expected_postcondition": {"field": "CLOUD_ZONE"},
                },
            }
        )
        output = io.StringIO()

        StdioCodexSimulator(io.StringIO(self._decision_frame(context)), output)(context)

        payload = json.loads(output.getvalue()[len(CONTEXT_FRAME):])
        rule = payload["decision_contract"]["input_generation_rule"]
        self.assertIn("supply a concrete value for CLOUD_ZONE", rule)
        self.assertIn("must not ask the Agent to invent", rule)

    def test_rejects_a_decision_for_a_stale_agent_response(self) -> None:
        context = self._context()
        stale = self._decision_frame(context, previous_response_hash="stale")

        with self.assertRaisesRegex(RuntimeError, "stale Agent response"):
            StdioCodexSimulator(io.StringIO(stale), io.StringIO())(context)

    def test_rejects_unframed_or_incomplete_decisions(self) -> None:
        context = self._context()
        with self.assertRaisesRegex(RuntimeError, "invalid frame"):
            StdioCodexSimulator(io.StringIO("{}\n"), io.StringIO())(context)

        incomplete = self._decision_frame(context, rationale="")
        with self.assertRaisesRegex(RuntimeError, "requires rationale"):
            StdioCodexSimulator(io.StringIO(incomplete), io.StringIO())(context)


if __name__ == "__main__":
    unittest.main()
