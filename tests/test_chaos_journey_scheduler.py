"""Focused contracts for immutable open-journey Chaos schedules."""

from __future__ import annotations

from dataclasses import replace
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from tests.agent_live.chaos_scheduler import (
    JOURNEY_SCHEDULE_SCHEMA_VERSION,
    JourneyOutcomeContract,
    build_journey_schedule,
    journey_schedule_payload,
    validate_journey_schedule,
    write_journey_schedule,
)


class JourneyScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.revision = {"commit": "test", "worktree_hash": "a" * 64}
        self.journey = {
            "journey_id": "nested-group-return",
            "start_scenario": "public_startup",
            "persona": "uncertain operator who changes direction",
            "mission": "configure five groups and recover through fallback",
            "allowed_risk_factors": [
                "mid_group_interruption",
                "language_switch",
                "correction",
            ],
            "max_turns": 24,
            "terminal_outcome": {
                "outcome_id": "configuration_ready",
                "required_postcondition_ids": [
                    "all_required_groups_complete",
                    "fallback_reached_execution_gate",
                ],
            },
            "forbidden_outcomes": [{
                "outcome_id": "lost_interrupted_group_state",
                "required_postcondition_ids": ["interrupted_group_state_missing"],
            }],
        }

    def build(self, **changes: object):
        journey = {**self.journey, **changes}
        return build_journey_schedule(
            revision=self.revision,
            seed=271,
            journey=journey,
        )

    def test_builds_typed_content_addressed_schedule_without_future_turns(self) -> None:
        schedule = self.build()
        payload = journey_schedule_payload(schedule)

        self.assertEqual(schedule.schema_version, JOURNEY_SCHEDULE_SCHEMA_VERSION)
        self.assertEqual(schedule.allowed_risk_factors, tuple(self.journey["allowed_risk_factors"]))
        self.assertIsInstance(schedule.terminal_outcome, JourneyOutcomeContract)
        self.assertEqual(
            schedule.forbidden_outcomes[0].outcome_id,
            "lost_interrupted_group_state",
        )
        self.assertNotIn("turns", payload)
        self.assertNotIn("messages", payload)
        self.assertEqual(schedule, self.build())
        validate_journey_schedule(schedule, revision=self.revision)

        with self.assertRaises(TypeError):
            schedule.revision["commit"] = "mutated"  # type: ignore[index]

    def test_every_immutable_input_participates_in_schedule_identity(self) -> None:
        baseline = self.build()
        variants = (
            self.build(journey_id="different-journey"),
            self.build(start_scenario="resume_session"),
            self.build(persona="expert operator"),
            self.build(mission="recover an execution failure"),
            self.build(allowed_risk_factors=["cancellation"]),
            self.build(max_turns=25),
            self.build(terminal_outcome={
                "outcome_id": "job_submitted",
                "required_postcondition_ids": ["job_artifact_exists"],
            }),
            self.build(forbidden_outcomes=[]),
            build_journey_schedule(
                revision={"commit": "other", "worktree_hash": "b" * 64},
                seed=271,
                journey=self.journey,
            ),
            build_journey_schedule(
                revision=self.revision,
                seed=272,
                journey=self.journey,
            ),
        )
        self.assertEqual(
            len({baseline.schedule_id, *(item.schedule_id for item in variants)}),
            len(variants) + 1,
        )

    def test_rejects_prewritten_dialogue_and_unknown_fields(self) -> None:
        for field in ("turns", "messages", "future_user_turns", "dialogue"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "unsupported fields"):
                    self.build(**{field: ["prewritten user message"]})

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.build(terminal_outcome={
                "outcome_id": "configuration_ready",
                "required_postcondition_ids": ["ready"],
                "next_user_message": "Y",
            })

    def test_rejects_incomplete_or_ambiguous_contracts(self) -> None:
        invalid = (
            ({"persona": ""}, "requires persona"),
            ({"max_turns": 0}, "positive integer"),
            ({"max_turns": True}, "positive integer"),
            ({"allowed_risk_factors": ["repeat", "repeat"]}, "unique ids"),
            ({"terminal_outcome": {"outcome_id": "ready"}}, "postcondition"),
            ({"forbidden_outcomes": "failure"}, "must be a sequence"),
            ({
                "forbidden_outcomes": [
                    {
                        "outcome_id": "same",
                        "required_postcondition_ids": ["first"],
                    },
                    {
                        "outcome_id": "same",
                        "required_postcondition_ids": ["second"],
                    },
                ],
            }, "must be unique"),
            ({
                "terminal_outcome": {
                    "outcome_id": "same",
                    "required_postcondition_ids": ["success"],
                },
                "forbidden_outcomes": [{
                    "outcome_id": "same",
                    "required_postcondition_ids": ["failure"],
                }],
            }, "cannot also be forbidden"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    self.build(**changes)

    def test_validation_rejects_stale_revision_identity_and_schema(self) -> None:
        schedule = self.build()
        with self.assertRaisesRegex(ValueError, "active revision"):
            validate_journey_schedule(
                schedule,
                revision={"commit": "other", "worktree_hash": "b" * 64},
            )
        with self.assertRaisesRegex(ValueError, "identity is stale"):
            validate_journey_schedule(
                replace(schedule, mission="modified after scheduling"),
                revision=self.revision,
            )
        with self.assertRaisesRegex(ValueError, "unsupported Journey schedule schema"):
            validate_journey_schedule(
                replace(schedule, schema_version=999),
                revision=self.revision,
            )

    def test_writes_canonical_json_payload(self) -> None:
        schedule = self.build()
        with TemporaryDirectory() as directory:
            target = write_journey_schedule(
                schedule,
                Path(directory) / "journey.json",
            )
            self.assertEqual(json.loads(target.read_text()), journey_schedule_payload(schedule))


if __name__ == "__main__":
    unittest.main()
