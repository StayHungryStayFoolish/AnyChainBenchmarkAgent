from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)
from tests.agent_live.chaos_scheduler import build_journey_schedule
from tests.agent_live.coverage_evidence import RuntimeTurnEvent
from tests.agent_live.dynamic_dual_ai_chaos import JourneyVerifierContext
from tests.agent_live.journey_simulator_bridge import load_verifier_registry
from tests.agent_live.retained_regression_obligations import (
    KNOWN_POSTCONDITION_IDS,
    build_retained_regression_obligations,
)
from tests.agent_live.retained_regression_runner import (
    build_product_obligation_evidence_artifact,
    build_retained_regression_runner_provider,
    RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY,
    RETAINED_REGRESSION_REGISTRY_IMPORT,
    retained_regression_journey_definitions,
    validate_retained_regression_runner_provider,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class RetainedRegressionRunnerProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.obligations = build_retained_regression_obligations(
            repo_root=REPO_ROOT,
            revision=REVISION,
        )
        cls.provider = build_retained_regression_runner_provider(
            obligations=cls.obligations,
            revision=REVISION,
        )

    def test_compiles_all_60_without_claiming_execution(self) -> None:
        self.assertEqual(self.provider["obligation_count"], 60)
        self.assertEqual(len(self.provider["targets"]), 60)
        self.assertFalse(self.provider["generation_is_execution"])
        self.assertEqual(self.provider["execution_status"], "not_run")
        validate_retained_regression_runner_provider(
            self.provider,
            obligations=self.obligations,
            revision=REVISION,
        )

    def test_exact_is_real_cli_and_never_dynamic(self) -> None:
        exact = [
            target
            for target in self.provider["targets"]
            if target["variant"] == "exact"
        ]
        self.assertEqual(len(exact), 15)
        for target in exact:
            execution = target["execution_contract"]
            self.assertEqual(execution["lane"], "real_cli")
            self.assertEqual(execution["transport"], "real_pty")
            self.assertEqual(execution["evidence_class"], "real_cli")
            self.assertFalse(execution["qualifies_as_dynamic"])
            self.assertTrue(execution["turns"])
            self.assertNotIn("journey_definition", target)

    def test_open_variants_have_no_prewritten_future_turns(self) -> None:
        journeys = retained_regression_journey_definitions(self.provider)
        self.assertEqual(len(journeys), 45)
        for definition in journeys:
            serialized = json.dumps(definition)
            journey = definition["journey"]
            self.assertEqual(
                definition["verifier_registry"],
                RETAINED_REGRESSION_REGISTRY_IMPORT,
            )
            schedule = build_journey_schedule(
                revision=REVISION,
                seed=20260724,
                journey=journey,
            )
            self.assertEqual(schedule.journey_id, journey["journey_id"])
            for forbidden in ("turns", "messages", "future_user_turns", "dialogue"):
                self.assertNotIn(f'"{forbidden}"', serialized)

    def test_registry_is_importable_and_fail_closed(self) -> None:
        registry = load_verifier_registry(RETAINED_REGRESSION_REGISTRY_IMPORT)
        self.assertEqual(
            registry.registry_id,
            RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.registry_id,
        )
        self.assertEqual(set(registry.definitions), set(KNOWN_POSTCONDITION_IDS))
        context = self._empty_context(self.provider["targets"][1])
        for postcondition_id, definition in registry.definitions.items():
            result = definition.verifier(
                context.__class__(
                    **{
                        **context.__dict__,
                        "evaluating_postcondition_id": postcondition_id,
                    }
                )
            )
            self.assertFalse(result.satisfied)
            self.assertTrue(result.details["fail_closed"])

    def test_every_catalog_postcondition_has_a_declarative_rule(self) -> None:
        rules = self.provider["verifier_registry"]["rules"]
        self.assertEqual(
            {rule["postcondition_id"] for rule in rules},
            set(KNOWN_POSTCONDITION_IDS),
        )
        for rule in rules:
            self.assertEqual(
                rule["required_artifact_roles"],
                ["transcript", "runtime_events", "checkpoint_diff"],
            )
            self.assertFalse(rule["natural_language_expected_is_verifier"])
            self.assertTrue(rule["keyword_matching_forbidden"])
            self.assertTrue(rule["assertions"])

    def test_provider_rejects_prewritten_open_journey_and_dynamic_exact_claim(self) -> None:
        provider = deepcopy(self.provider)
        open_target = next(
            target for target in provider["targets"] if target["variant"] != "exact"
        )
        open_target["journey_definition"]["journey"]["turns"] = ["prewritten"]
        with self.assertRaisesRegex(ValueError, "prewritten turns"):
            validate_retained_regression_runner_provider(
                provider,
                obligations=self.obligations,
                revision=REVISION,
            )

        provider = deepcopy(self.provider)
        exact_target = next(
            target for target in provider["targets"] if target["variant"] == "exact"
        )
        exact_target["execution_contract"]["qualifies_as_dynamic"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim dynamic"):
            validate_retained_regression_runner_provider(
                provider,
                obligations=self.obligations,
                revision=REVISION,
            )

    def test_provider_rejects_missing_rule_and_generation_pass_claim(self) -> None:
        provider = deepcopy(self.provider)
        provider["verifier_registry"]["rules"].pop()
        with self.assertRaisesRegex(ValueError, "registry is incomplete"):
            validate_retained_regression_runner_provider(
                provider,
                obligations=self.obligations,
                revision=REVISION,
            )

        provider = deepcopy(self.provider)
        provider["execution_status"] = "passed"
        with self.assertRaisesRegex(ValueError, "claimed execution"):
            validate_retained_regression_runner_provider(
                provider,
                obligations=self.obligations,
                revision=REVISION,
            )

    def test_adapter_evaluates_and_cannot_admit_unimplemented_semantics(self) -> None:
        obligation = next(
            row for row in self.obligations if row["variant"] == "exact"
        )
        target = next(
            row
            for row in self.provider["targets"]
            if row["obligation_id"] == obligation["obligation_id"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for role in ("transcript", "runtime_events", "checkpoint_diff"):
                path = root / f"{role}.json"
                path.write_text(json.dumps({"role": role}), encoding="utf-8")
                artifacts[role] = path
            evidence = build_product_obligation_evidence_artifact(
                obligation=obligation,
                target=target,
                revision=REVISION,
                execution={
                    "execution_id": "real-cli-execution",
                    "runner": "retained-regression-real-cli-v1",
                    "transport": "real_pty",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "started_at": "2026-07-24T00:00:00Z",
                    "finished_at": "2026-07-24T00:01:00Z",
                },
                artifact_paths=artifacts,
                verifier_context=self._empty_context(target),
            )
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            summary = admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[evidence_path],
                revision=REVISION,
            )
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["not_run"], 59)
            self.assertFalse(summary["complete"])

            with self.assertRaises(TypeError):
                build_product_obligation_evidence_artifact(
                    obligation=obligation,
                    target=target,
                    revision=REVISION,
                    execution={},
                    artifact_paths=artifacts,
                    verifier_results={},  # type: ignore[call-arg]
                )

    def test_adapter_rejects_wrong_evidence_class_and_incomplete_results(self) -> None:
        obligation = next(
            row for row in self.obligations if row["variant"] != "exact"
        )
        target = next(
            row
            for row in self.provider["targets"]
            if row["obligation_id"] == obligation["obligation_id"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for role in ("transcript", "runtime_events", "checkpoint_diff"):
                path = root / role
                path.write_text(role, encoding="utf-8")
                artifacts[role] = path
            with self.assertRaisesRegex(ValueError, "evidence class"):
                build_product_obligation_evidence_artifact(
                    obligation=obligation,
                    target=target,
                    revision=REVISION,
                    execution={
                        "execution_id": "wrong-runner",
                        "runner": "retained-regression-real-cli-v1",
                        "transport": "real_pty",
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "started_at": "start",
                        "finished_at": "finish",
                    },
                    artifact_paths=artifacts,
                    verifier_context=self._empty_context(target),
                )

    def _empty_context(self, target: dict) -> JourneyVerifierContext:
        definition = target.get("journey_definition")
        journey = (
            definition["journey"]
            if definition
            else {
                "journey_id": target["obligation_id"],
                "start_scenario": target["seed_contract"]["scenario_id"],
                "persona": "exact replay",
                "mission": "evaluate exact retained regression",
                "allowed_risk_factors": [],
                "max_turns": 12,
                "terminal_outcome": {
                    "outcome_id": "complete",
                    "required_postcondition_ids": ["exact_fixture_turns_observed"],
                },
                "forbidden_outcomes": [],
            }
        )
        schedule = build_journey_schedule(
            revision=REVISION,
            seed=20260724,
            journey=journey,
        )
        event = RuntimeTurnEvent(
            schema_version=1,
            event_type="baseline",
            thread_id="test",
            session_purpose="test",
            before_fingerprint="a",
            after_fingerprint="a",
            turn_index=0,
            active_group="",
            pending_question_id="",
            action_queue_types=(),
            revision=REVISION,
        )
        return JourneyVerifierContext(
            schedule=schedule,
            initial_event=event,
            current_event=event,
            completed_turns=(),
            transcript=(),
            observed_edge_keys=(),
            latest_turn=None,
        )


if __name__ == "__main__":
    unittest.main()
