"""Contracts for the external response-driven Journey simulator bridge."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.agent_live.chaos_scheduler import build_journey_schedule, write_journey_schedule
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneySimulatorContext,
    build_journey_outcome_verifier_registry,
)
from tests.agent_live.journey_simulator_bridge import (
    CONTEXT_FRAME,
    DECISION_FRAME,
    JourneySimulatorInvalidFrame,
    JourneySimulatorStaleResponse,
    StdioCodexJourneySimulator,
    load_journey_schedule,
    load_verifier_registry,
)


REVISION = {"commit": "journey-bridge", "worktree_hash": "c" * 64}


def bridge_ready(context):
    return JourneyPostconditionResult(
        "ready_state", bool(context.completed_turns), {"turns": len(context.completed_turns)}
    )


def bridge_not_lost(context):
    return JourneyPostconditionResult("state_lost", False, {})


TEST_JOURNEY_REGISTRY = build_journey_outcome_verifier_registry((
    JourneyPostconditionVerifierDefinition(
        "ready_state", "bridge-ready", 1, "Terminal bridge verifier.", bridge_ready
    ),
    JourneyPostconditionVerifierDefinition(
        "state_lost", "bridge-lost", 1, "Forbidden bridge verifier.", bridge_not_lost
    ),
))


class JourneySimulatorBridgeTest(unittest.TestCase):
    def _schedule(self):
        return build_journey_schedule(
            revision=REVISION,
            seed=811,
            journey={
                "journey_id": "bridge-journey",
                "start_scenario": "opening",
                "persona": "uncertain operator",
                "mission": "react to each complete response",
                "allowed_risk_factors": ["multiline"],
                "max_turns": 5,
                "terminal_outcome": {
                    "outcome_id": "ready",
                    "required_postcondition_ids": ["ready_state"],
                },
                "forbidden_outcomes": [{
                    "outcome_id": "lost",
                    "required_postcondition_ids": ["state_lost"],
                }],
            },
        )

    def _context(self):
        return JourneySimulatorContext(
            session_id="journey-bridge-session",
            turn_index=4,
            previous_agent_response="Agent> Choose what to do next.\n1. Continue\n2. Change",
            previous_response_received_at_ns=1234,
            schedule=self._schedule(),
            transcript=(("Earlier input", "Earlier response"),),
            observed_edge_keys=("opening::one",),
        )

    def _decision_frame(self, context, **changes):
        payload = {
            "previous_response_hash": content_hash(context.previous_agent_response),
            "user_message": "first line\nsecond line",
            "persona": context.schedule.persona,
            "mission": context.schedule.mission,
            "rationale": "The complete response asked for the next choice.",
            "risk_factor_ids": ["multiline"],
        }
        payload.update(changes)
        return DECISION_FRAME + json.dumps(payload, ensure_ascii=False) + "\n"

    def test_context_is_response_bound_and_decision_preserves_multiline(self) -> None:
        context = self._context()
        output = io.StringIO()
        decision = StdioCodexJourneySimulator(
            io.StringIO(self._decision_frame(context)), output
        )(context)
        self.assertEqual(decision.user_message, "first line\nsecond line")
        framed = output.getvalue()
        self.assertTrue(framed.startswith(CONTEXT_FRAME))
        payload = json.loads(framed[len(CONTEXT_FRAME):])
        self.assertEqual(payload["previous_response_hash"], content_hash(
            context.previous_agent_response
        ))
        self.assertEqual(payload["transcript"][0]["user"], "Earlier input")
        self.assertNotIn("target_coverage_ids", payload)
        self.assertNotIn("future_turns", payload)

    def test_context_frame_redacts_content_but_keeps_source_response_hash(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        context = replace(
            self._context(),
            previous_agent_response=f"Endpoint https://rpc.example/{secret}",
            transcript=((f"Authorization: Bearer {secret}", "Earlier response"),),
        )
        output = io.StringIO()
        StdioCodexJourneySimulator(
            io.StringIO(self._decision_frame(context)), output
        )(context)
        framed = output.getvalue()
        self.assertNotIn(secret, framed)
        payload = json.loads(framed[len(CONTEXT_FRAME):])
        self.assertEqual(
            payload["previous_response_hash"],
            content_hash(context.previous_agent_response),
        )

    def test_stale_response_and_invalid_frame_are_typed_protocol_failures(self) -> None:
        context = self._context()
        stale = self._decision_frame(context, previous_response_hash="stale")
        with self.assertRaises(JourneySimulatorStaleResponse):
            StdioCodexJourneySimulator(io.StringIO(stale), io.StringIO())(context)
        with self.assertRaises(JourneySimulatorInvalidFrame):
            StdioCodexJourneySimulator(io.StringIO("plain text\n"), io.StringIO())(context)

    def test_schedule_loader_rejects_stale_identity_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_journey_schedule(self._schedule(), Path(tmpdir) / "journey.json")
            loaded = load_journey_schedule(path, revision=REVISION)
            self.assertEqual(loaded.schedule_id, self._schedule().schedule_id)
            payload = json.loads(path.read_text())
            payload["mission"] = "tampered"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_journey_schedule(path, revision=REVISION)

    def test_registry_loader_requires_an_authoritative_registry_object(self) -> None:
        loaded = load_verifier_registry(
            f"{bridge_ready.__module__}:TEST_JOURNEY_REGISTRY"
        )
        self.assertEqual(loaded.registry_id, TEST_JOURNEY_REGISTRY.registry_id)
        with self.assertRaisesRegex(TypeError, "wrong type"):
            load_verifier_registry(f"{bridge_ready.__module__}:REVISION")


if __name__ == "__main__":
    unittest.main()
