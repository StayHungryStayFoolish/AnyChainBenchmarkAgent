from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.codex_simulator_bridge import (
    build_simulator_attestation,
    simulator_context_binding,
)
from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    content_hash,
    pty_transcript_hash,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyDecisionProvenance,
    JourneyVerifierContext,
    TerminalTurnFailure,
)
from tests.agent_live.journey_simulator_bridge import load_verifier_registry
from tests.agent_live.retained_regression_obligations import (
    KNOWN_POSTCONDITION_IDS,
    build_retained_regression_obligations,
)
from tests.agent_live.retained_regression_attestations import (
    build_variant_attestation,
)
from tests.agent_live.retained_regression_runner import (
    build_retained_regression_definition_manifest,
    build_product_obligation_evidence_artifact,
    build_retained_regression_runner_provider,
    convert_completed_retained_journey_to_product_evidence,
    execute_exact_retained_regressions,
    freeze_retained_regression_open_batch,
    RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY,
    RETAINED_REGRESSION_REGISTRY_IMPORT,
    _validate_exact_terminal_revision,
    _write_exact_retained_artifacts,
    retained_regression_journey_definitions,
    validate_retained_regression_runner_provider,
    write_retained_regression_target_set,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class RetainedRegressionExactFailureBoundaryTest(unittest.TestCase):
    def test_exact_terminal_revision_uses_protocol_origin_revision(self) -> None:
        class Outcome:
            origin_revision = REVISION

        _validate_exact_terminal_revision(
            Outcome(),
            active_revision=REVISION,
            stage="resume",
        )
        with self.assertRaisesRegex(RuntimeError, "turn terminal revision"):
            _validate_exact_terminal_revision(
                Outcome(),
                active_revision={
                    "commit": "stale",
                    "worktree_hash": REVISION["worktree_hash"],
                },
                stage="turn",
            )

    def test_exact_suite_failure_never_publishes_execution_index(self) -> None:
        obligation_id = "exact-failure-boundary"
        target = {
            "obligation_id": obligation_id,
            "variant": "exact",
        }
        provider = {
            "provider_hash": "provider-hash",
            "revision_binding": REVISION,
            "targets": [target],
        }
        obligations = [{"obligation_id": obligation_id}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exact"

            def fail_execution(**kwargs):
                runtime = kwargs["output_root"] / "failed-runtime"
                runtime.mkdir()
                (runtime / "terminal-failure.json").write_text(
                    json.dumps({
                        "qualifying_evidence": False,
                        "failure_kind": "typed_terminal_failure",
                    }),
                    encoding="utf-8",
                )
                raise TerminalTurnFailure(
                    "provider_failure",
                    "Agent terminal reported a provider failure.",
                )

            with patch(
                "tests.agent_live.retained_regression_runner."
                "execute_exact_retained_regression",
                side_effect=fail_execution,
            ):
                with self.assertRaises(TerminalTurnFailure):
                    execute_exact_retained_regressions(
                        repo_root=REPO_ROOT,
                        provider=provider,
                        obligations=obligations,
                        output_root=output,
                        obligation_id=obligation_id,
                    )

            self.assertTrue((output / "failed-runtime/terminal-failure.json").is_file())
            self.assertFalse((output / "execution-index.json").exists())
            self.assertEqual(
                list(output.rglob("product-obligation-evidence.json")),
                [],
            )


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
            verifier_input = definition["verifier_input_contract"]
            self.assertTrue(verifier_input["source_contract_hash"])
            self.assertTrue(verifier_input["variant_contract_hash"])
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
            self.assertEqual(
                definition["frozen_execution"],
                {
                    "obligation_id": schedule.journey_id,
                    "seed": schedule.seed,
                    "schedule_id": schedule.schedule_id,
                    "schedule_hash": content_hash(
                        journey_schedule_payload(schedule)
                    ),
                    "subject_group": schedule.subject_group,
                    "revision_binding": REVISION,
                },
            )
            self.assertTrue(
                definition["simulator_attestation_contract"]["required"]
            )
            for forbidden in ("turns", "messages", "future_user_turns", "dialogue"):
                self.assertNotIn(f'"{forbidden}"', serialized)

    def test_provider_is_ready_with_auditable_non_cryptographic_attestation(self) -> None:
        registry = self.provider["verifier_registry"]
        self.assertEqual(registry["execution_readiness"], "ready")
        self.assertEqual(registry["execution_blockers"], [])
        self.assertEqual(registry["unsupported_postcondition_ids"], [])
        self.assertEqual(
            registry["simulator_attestation_policy"],
            {
                "required": True,
                "identity_strength": "auditable_declaration_only",
                "cryptographic_identity_claimed": False,
                "selection_mode": "response_driven",
                "prewritten_future_turns": False,
            },
        )

    def test_missing_postcondition_evaluator_blocks_execution(self) -> None:
        evaluator_path = (
            "tests.agent_live.retained_regression_runner."
            "_IMPLEMENTED_POSTCONDITION_EVALUATORS"
        )
        incomplete = {
            key: value
            for key, value in RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.definitions.items()
            if key != "current_turn_language_preserved"
        }
        with patch(evaluator_path, incomplete):
            provider = build_retained_regression_runner_provider(
                obligations=self.obligations,
                revision=REVISION,
            )
        registry = provider["verifier_registry"]
        self.assertEqual(registry["execution_readiness"], "blocked")
        self.assertEqual(
            registry["execution_blockers"],
            [
                "missing_postcondition_evaluator:"
                "current_turn_language_preserved"
            ],
        )

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

    def test_provider_rejects_rehashed_stale_evaluator_implementation(self) -> None:
        provider = deepcopy(self.provider)
        rule = provider["verifier_registry"]["rules"][0]
        rule["evaluator_implementation_hash"] = "f" * 64
        unsigned_rule = dict(rule)
        unsigned_rule.pop("rule_hash")
        rule["rule_hash"] = content_hash(unsigned_rule)
        unsigned_provider = dict(provider)
        unsigned_provider.pop("provider_hash")
        provider["provider_hash"] = content_hash(unsigned_provider)

        with self.assertRaisesRegex(ValueError, "rule contract is incomplete"):
            validate_retained_regression_runner_provider(
                provider,
                obligations=self.obligations,
                revision=REVISION,
            )

    def test_exact_adapter_reconstructs_immutable_fixture_from_artifacts(self) -> None:
        obligation = next(
            row for row in self.obligations
            if row["variant"] == "exact"
            and len(row["stimulus_contract"]["turns"]) == 1
        )
        target = next(
            row
            for row in self.provider["targets"]
            if row["obligation_id"] == obligation["obligation_id"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._exact_one_turn_context(
                target,
                obligation["stimulus_contract"]["turns"][0],
            )
            execution_id = "real-cli-execution"
            artifacts = _write_exact_retained_artifacts(
                runtime_root=root,
                obligation=obligation,
                target=target,
                revision=REVISION,
                execution_id=execution_id,
                initial_event=context.initial_event,
                events=context.completed_events,
                turns=context.completed_turns,
            )
            evidence = build_product_obligation_evidence_artifact(
                obligation=obligation,
                target=target,
                revision=REVISION,
                execution={
                    "execution_id": execution_id,
                    "runner": "retained-regression-real-cli-v1",
                    "transport": "real_pty",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "started_at": "1",
                    "finished_at": "9999999999999999999",
                },
                artifact_paths=artifacts,
            )
            result = next(
                item
                for item in evidence["verifier_results"]
                if item["verifier_id"] == "exact_fixture_turns_observed"
            )
            self.assertEqual(result["status"], "passed")

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
                        "started_at": "1",
                        "finished_at": "9999999999999999999",
                    },
                    artifact_paths=artifacts,
                )

    def test_adapter_reconstructs_context_only_from_bound_disk_artifacts(self) -> None:
        obligation = next(
            row for row in self.obligations
            if row["variant"] != "exact"
        )
        target = next(
            row for row in self.provider["targets"]
            if row["obligation_id"] == obligation["obligation_id"]
        )
        context = self._one_turn_context(target)
        execution_id = "response-driven-run"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._write_bound_artifacts(
                root,
                obligation=obligation,
                target=target,
                context=context,
                execution_id=execution_id,
            )
            evidence = build_product_obligation_evidence_artifact(
                obligation=obligation,
                target=target,
                revision=REVISION,
                execution={
                    "execution_id": execution_id,
                    "runner": "retained-regression-response-driven-journey-v1",
                    "transport": "real_pty",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "started_at": "1",
                    "finished_at": "9999999999999999999",
                },
                artifact_paths=artifacts,
            )
            self.assertEqual(len(evidence["artifacts"]), 3)
            self.assertEqual(
                {item["role"] for item in evidence["artifacts"]},
                {"transcript", "runtime_events", "checkpoint_diff"},
            )
            self.assertTrue(evidence["verifier_results"])

            transcript_path = Path(artifacts["transcript"])
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            payload["identity"]["obligation_id"] = "different-obligation"
            unsigned = {
                key: value
                for key, value in payload.items()
                if key != "artifact_hash"
            }
            payload["artifact_hash"] = content_hash(unsigned)
            transcript_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact binding is stale"):
                build_product_obligation_evidence_artifact(
                    obligation=obligation,
                    target=target,
                    revision=REVISION,
                    execution={
                        "execution_id": execution_id,
                        "runner": (
                            "retained-regression-response-driven-journey-v1"
                        ),
                        "transport": "real_pty",
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "started_at": "1",
                        "finished_at": "9999999999999999999",
                    },
                    artifact_paths=artifacts,
                )

    def test_response_bound_lineage_and_forbidden_absence_semantics(self) -> None:
        target = next(
            row for row in self.provider["targets"] if row["variant"] != "exact"
        )
        context = self._one_turn_context(target)
        registry = RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY

        response_bound = registry.definitions[
            "response_driven_selection_observed"
        ].verifier(replace(
            context,
            evaluating_postcondition_id="response_driven_selection_observed",
        ))
        self.assertTrue(response_bound.satisfied)
        self.assertTrue(response_bound.details["complete_runtime_lineage_observed"])

        no_duplicate = registry.definitions["duplicate_job_submission"].verifier(
            replace(
                context,
                evaluating_postcondition_id="duplicate_job_submission",
            )
        )
        self.assertFalse(no_duplicate.satisfied)

        no_side_channel_rejection = registry.definitions[
            "typed_confirmation_rejected_by_side_channel"
        ].verifier(replace(
            context,
            evaluating_postcondition_id=(
                "typed_confirmation_rejected_by_side_channel"
            ),
        ))
        self.assertFalse(no_side_channel_rejection.satisfied)

    def test_forbidden_predicates_fail_when_violation_is_observed(self) -> None:
        target = next(
            row for row in self.provider["targets"] if row["variant"] != "exact"
        )
        context = self._one_turn_context(target)
        duplicate_event = replace(
            context.current_event,
            execution_receipt_summary={
                "receipt_idempotency_key": "same-key",
                "receipt_id": "receipt-b",
                "job_id": "job-b",
            },
        )
        first_event = replace(
            context.current_event,
            after_fingerprint="c" * 64,
            execution_receipt_summary={
                "receipt_idempotency_key": "same-key",
                "receipt_id": "receipt-a",
                "job_id": "job-a",
            },
        )
        second_turn = replace(
            context.latest_turn,
            turn_index=2,
            before_fingerprint="c" * 64,
            after_fingerprint=duplicate_event.after_fingerprint,
            previous_response_received_at_ns=(
                context.latest_turn.previous_response_received_at_ns + 20
            ),
            user_message_submitted_at_ns=(
                context.latest_turn.user_message_submitted_at_ns + 20
            ),
            agent_response_received_at_ns=(
                context.latest_turn.agent_response_received_at_ns + 20
            ),
        )
        second_turn = replace(
            second_turn,
            transcript_hash=pty_transcript_hash(
                session_id=second_turn.session_id,
                turn_index=second_turn.turn_index,
                previous_agent_response=second_turn.previous_agent_response,
                user_message=second_turn.user_message,
                agent_response=second_turn.agent_response,
            ),
        )
        duplicate_event = replace(
            duplicate_event,
            before_fingerprint="c" * 64,
            turn_index=2,
        )
        first_turn = replace(
            context.latest_turn,
            after_fingerprint="c" * 64,
        )
        first_turn = replace(
            first_turn,
            transcript_hash=pty_transcript_hash(
                session_id=first_turn.session_id,
                turn_index=first_turn.turn_index,
                previous_agent_response=first_turn.previous_agent_response,
                user_message=first_turn.user_message,
                agent_response=first_turn.agent_response,
            ),
        )
        first_decision = context.completed_decisions[0]
        second_decision = replace(
            first_decision,
            turn_index=2,
            selected_at_ns=first_decision.selected_at_ns + 20,
            submitted_at_ns=first_decision.submitted_at_ns + 20,
        )
        duplicate_context = replace(
            context,
            current_event=duplicate_event,
            completed_turns=(first_turn, second_turn),
            completed_events=(first_event, duplicate_event),
            completed_decisions=(first_decision, second_decision),
            latest_turn=second_turn,
            evaluating_postcondition_id="duplicate_job_submission",
        )
        duplicate = RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.definitions[
            "duplicate_job_submission"
        ].verifier(duplicate_context)
        self.assertTrue(duplicate.satisfied)
        self.assertEqual(
            duplicate.details["duplicate_submission_counts"],
            {"same-key": 2},
        )

        accepted = {
            "receipt_type": "pending_resolution",
            "turn_index": 1,
            "pending_id": "network_interface",
            "pending_group": "network",
            "pending_contract_hash": "1" * 64,
            "resolution_path": "exact_contract",
            "selected_option_id": "confirm_detected",
            "selected_value_hash": "2" * 64,
            "resolved_action_id": "action-1",
            "input_hash": "3" * 64,
            "normalizer": "typed_pending_contract",
            "verdict": "accepted",
        }
        accepted["receipt_id"] = content_hash(accepted)
        rejected = {
            "receipt_type": "domain_commit",
            "turn_index": 1,
            "owner": "environment",
            "completion": "rejected",
            "group_registry_contract_hash": "4" * 64,
            "pending_before_hash": "5" * 64,
            "pending_after_hash": "5" * 64,
            "consumed_action_ids": [],
            "invalidated_groups": [],
            "invalidated_fields": [],
            "response_fragments": [],
            "blocker_semantic_hash": "6" * 64,
        }
        rejected["receipt_id"] = content_hash(rejected)
        rejection_event = replace(
            context.current_event,
            control_receipts=(accepted, rejected),
        )
        rejection_context = replace(
            context,
            current_event=rejection_event,
            completed_events=(rejection_event,),
            evaluating_postcondition_id=(
                "typed_confirmation_rejected_by_side_channel"
            ),
        )
        rejection = RETAINED_REGRESSION_JOURNEY_VERIFIER_REGISTRY.definitions[
            "typed_confirmation_rejected_by_side_channel"
        ].verifier(rejection_context)
        self.assertTrue(rejection.satisfied)
        self.assertEqual(
            rejection.details["receipt_ids"],
            [rejected["receipt_id"]],
        )

    def test_open_definition_target_and_bounded_batch_pipeline(self) -> None:
        manifest = build_retained_regression_definition_manifest(self.provider)
        self.assertEqual(manifest["definition_count"], 45)
        self.assertFalse(manifest["prewritten_future_turns"])
        serialized = json.dumps(manifest)
        for forbidden in ('"turns"', '"messages"', '"future_user_turns"'):
            self.assertNotIn(forbidden, serialized)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets_dir = root / "targets"
            target_manifest_path = write_retained_regression_target_set(
                provider=self.provider,
                definition_manifest=manifest,
                output_dir=targets_dir,
            )
            target_manifest = json.loads(
                target_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(target_manifest["target_count"], 45)
            self.assertEqual(
                len(tuple(targets_dir.glob("[0-9][0-9].json"))),
                45,
            )

            sentinel = object()
            with patch(
                "tests.agent_live.retained_regression_runner."
                "freeze_batch_manifest",
                return_value=sentinel,
            ) as freeze:
                result = freeze_retained_regression_open_batch(
                    repo_root=REPO_ROOT,
                    provider=self.provider,
                    targets_dir=targets_dir,
                    manifest_path=root / "batch.json",
                    runtime_base=root / "runtime",
                    max_concurrency=4,
                )
            self.assertIs(result, sentinel)
            kwargs = freeze.call_args.kwargs
            self.assertEqual(kwargs["shard_count"], 45)
            self.assertEqual(kwargs["max_concurrency"], 4)
            self.assertFalse(kwargs["formal_profile"])
            self.assertEqual(
                kwargs["expected_obligation_set_hash"],
                target_manifest["expected_obligation_set_hash"],
            )

            target = targets_dir / "01.json"
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                freeze_retained_regression_open_batch(
                    repo_root=REPO_ROOT,
                    provider=self.provider,
                    targets_dir=targets_dir,
                    manifest_path=root / "tampered-batch.json",
                    runtime_base=root / "runtime-2",
                )

    def test_exact_suite_executes_all_15_authoritative_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exact"

            def fake_execute(**kwargs):
                target = kwargs["target"]
                runtime = (
                    kwargs["output_root"]
                    / f"fake-{target['obligation_id']}"
                )
                runtime.mkdir()
                evidence = runtime / "evidence.json"
                evidence.write_text(
                    json.dumps({"obligation_id": target["obligation_id"]}),
                    encoding="utf-8",
                )
                return evidence

            with patch(
                "tests.agent_live.retained_regression_runner."
                "execute_exact_retained_regression",
                side_effect=fake_execute,
            ) as execute:
                paths = execute_exact_retained_regressions(
                    repo_root=REPO_ROOT,
                    provider=self.provider,
                    obligations=self.obligations,
                    output_root=output,
                )
            self.assertEqual(len(paths), 15)
            self.assertEqual(execute.call_count, 15)
            index = json.loads(
                (output / "execution-index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["scheduled"], 15)
            self.assertEqual(index["completed"], 15)

    def test_completed_journey_converts_retained_runtime_artifacts(self) -> None:
        obligation = next(
            row for row in self.obligations
            if row["variant"] != "exact"
        )
        target = next(
            row for row in self.provider["targets"]
            if row["obligation_id"] == obligation["obligation_id"]
        )
        context = self._one_turn_context(target)
        execution_id = "response-driven-run"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._write_bound_artifacts(
                root,
                obligation=obligation,
                target=target,
                context=context,
                execution_id=execution_id,
            )
            retained = {
                role: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for role, path in artifacts.items()
            }
            schedule = build_journey_schedule(
                revision=REVISION,
                seed=20260724,
                journey=target["journey_definition"]["journey"],
            )
            (root / "journey-schedule.json").write_text(
                json.dumps(journey_schedule_payload(schedule)),
                encoding="utf-8",
            )
            source_turn = {
                "turn_identity": {
                    "previous_response_received_at_ns": (
                        context.latest_turn.previous_response_received_at_ns
                    ),
                    "agent_response_received_at_ns": (
                        context.latest_turn.agent_response_received_at_ns
                    ),
                }
            }
            source_evidence = {
                "evidence_id": "journey-evidence",
                "execution_id": execution_id,
                "provider": "deepseek",
                "model": "deepseek-chat",
                "turns": [source_turn],
                "retained_artifacts": retained,
            }
            evidence_dir = root / "evidence"
            evidence_dir.mkdir()
            source_path = evidence_dir / "journey.json"
            source_path.write_text(
                json.dumps(source_evidence),
                encoding="utf-8",
            )
            result = {
                "schema_version": 1,
                "journey_id": obligation["obligation_id"],
                "schedule_id": schedule.schedule_id,
                "revision": REVISION,
                "terminal_classification": "passed",
                "execution_status": "passed",
                "qualifying_evidence": True,
                "evidence_id": "journey-evidence",
                "evidence_path": str(source_path),
                "turns": [source_turn],
            }
            (root / "journey-result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            output = root / "product-evidence.json"
            with patch(
                "tests.agent_live.retained_regression_runner."
                "validate_journey_evidence_artifact"
            ):
                converted = (
                    convert_completed_retained_journey_to_product_evidence(
                        obligation=obligation,
                        target=target,
                        revision=REVISION,
                        runtime_root=root,
                        evidence_path=output,
                        provider="deepseek",
                        model="deepseek-chat",
                    )
                )
            self.assertEqual(converted, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["obligation_id"], obligation["obligation_id"])
            self.assertEqual(
                payload["execution"]["runner"],
                "retained-regression-response-driven-journey-v1",
            )

    def _one_turn_context(self, target: dict) -> JourneyVerifierContext:
        base = self._empty_context(target)
        before = "a" * 64
        after = "b" * 64
        user_message = "continue"
        previous_response = "Choose the next action."
        agent_response = "The next action is ready."
        now = time.monotonic_ns()
        turn = PtyCliTurnRecord(
            session_id="test",
            turn_index=1,
            previous_agent_response=previous_response,
            user_message=user_message,
            agent_response=agent_response,
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint=before,
            after_fingerprint=after,
            transcript_hash=pty_transcript_hash(
                session_id="test",
                turn_index=1,
                previous_agent_response=previous_response,
                user_message=user_message,
                agent_response=agent_response,
            ),
            previous_response_received_at_ns=now,
            user_message_submitted_at_ns=now + 2,
            agent_response_received_at_ns=now + 3,
        )
        initial_event = replace(
            base.initial_event,
            schema_version=2,
            event_type="startup_snapshot",
            thread_id="test",
            before_fingerprint=before,
            after_fingerprint=before,
        )
        event = RuntimeTurnEvent(
            schema_version=3,
            event_type="turn_committed",
            thread_id="test",
            session_purpose="test",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=1,
            active_group="opening",
            pending_question_id="",
            action_queue_types=(),
            revision=REVISION,
            turn_receipt_summary={
                "turn_id": "turn-1",
                "input_hash": hashlib.sha256(
                    user_message.encode("utf-8")
                ).hexdigest(),
            },
            pending_transition={
                "before_hash": "1" * 64,
                "after_hash": "1" * 64,
            },
            render_manifest={"fragment_hashes": []},
        )
        verifier_input = target["journey_definition"][
            "verifier_input_contract"
        ]
        source_step = verifier_input["source_contract"]["source_steps"][0]
        broker_request_id = "request-1"
        context_binding = simulator_context_binding({
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "previous_response_hash": content_hash(previous_response),
            "previous_response_received_at_ns": (
                turn.previous_response_received_at_ns
            ),
            "schedule": journey_schedule_payload(base.schedule),
            "observed_edge_keys": [],
        })
        variant_binding = {
            "source_step_id": source_step["step_id"],
            "semantic_role": source_step["semantic_role"],
        }
        unsigned_decision = {
            "previous_response_hash": content_hash(previous_response),
            "broker_request_id": broker_request_id,
            "user_message": user_message,
            "persona": "operator",
            "mission": "continue",
            "rationale": "response-bound",
            "risk_factor_ids": [],
            "variant_binding": variant_binding,
        }
        simulator_attestation = build_simulator_attestation(
            actor_kind="codex",
            task_id="task-1",
            model="gpt-test",
            request_id=broker_request_id,
            previous_response_hash=content_hash(previous_response),
            context_hash=content_hash(context_binding),
            decision_hash=content_hash(unsigned_decision),
            user_message_hash=content_hash(user_message),
            turn_index=turn.turn_index,
            declared_at_ns=now + 1,
        )
        variant_attestation = build_variant_attestation(
            actor=simulator_attestation["actor"],
            verifier_input_contract=verifier_input,
            execution_binding={
                "execution_id": "response-driven-run",
                "session_id": turn.session_id,
                "schedule_id": base.schedule.schedule_id,
                "obligation_id": base.schedule.journey_id,
                "broker_request_id": broker_request_id,
                "simulator_attestation_id": simulator_attestation[
                    "attestation_id"
                ],
            },
            source_step_id=source_step["step_id"],
            semantic_role=source_step["semantic_role"],
            turn_index=turn.turn_index,
            previous_response_hash=content_hash(previous_response),
            user_message_hash=content_hash(user_message),
            selected_at_ns=now + 1,
            declared_at_ns=now + 1,
        )
        decision = JourneyDecisionProvenance(
            turn_index=1,
            previous_response_hash=content_hash(previous_response),
            selected_at_ns=now + 1,
            submitted_at_ns=now + 2,
            user_message_hash=content_hash(user_message),
            persona="operator",
            mission="continue",
            rationale="response-bound",
            execution_id="response-driven-run",
            obligation_id=base.schedule.journey_id,
            broker_request_id=broker_request_id,
            simulator_context_binding=context_binding,
            simulator_attestation=simulator_attestation,
            variant_attestation=variant_attestation,
        )
        return replace(
            base,
            initial_event=initial_event,
            current_event=event,
            completed_turns=(turn,),
            completed_events=(event,),
            completed_decisions=(decision,),
            latest_turn=turn,
            transcript=((user_message, agent_response),),
        )

    def _exact_one_turn_context(
        self,
        target: dict,
        user_message: str,
    ) -> JourneyVerifierContext:
        base = self._empty_context(target)
        before = "a" * 64
        after = "b" * 64
        previous_response = "Choose the next action."
        agent_response = "The exact fixture was processed."
        now = time.monotonic_ns()
        initial_event = replace(
            base.initial_event,
            schema_version=2,
            event_type="startup_snapshot",
            thread_id="test",
            before_fingerprint=before,
            after_fingerprint=before,
        )
        event = RuntimeTurnEvent(
            schema_version=3,
            event_type="turn_committed",
            thread_id="test",
            session_purpose="test",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=1,
            active_group="opening",
            pending_question_id="",
            action_queue_types=(),
            revision=REVISION,
            turn_receipt_summary={
                "turn_id": "turn-1",
                "input_hash": hashlib.sha256(
                    user_message.encode("utf-8")
                ).hexdigest(),
            },
            pending_transition={
                "before_hash": "1" * 64,
                "after_hash": "1" * 64,
            },
            render_manifest={"fragment_hashes": []},
        )
        turn = PtyCliTurnRecord(
            session_id="test",
            turn_index=1,
            previous_agent_response=previous_response,
            user_message=user_message,
            agent_response=agent_response,
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint=before,
            after_fingerprint=after,
            transcript_hash=pty_transcript_hash(
                session_id="test",
                turn_index=1,
                previous_agent_response=previous_response,
                user_message=user_message,
                agent_response=agent_response,
            ),
            previous_response_received_at_ns=now,
            user_message_submitted_at_ns=now + 1,
            agent_response_received_at_ns=now + 2,
        )
        return replace(
            base,
            initial_event=initial_event,
            current_event=event,
            completed_turns=(turn,),
            completed_events=(event,),
            completed_decisions=(),
            latest_turn=turn,
            transcript=((user_message, agent_response),),
        )

    def _write_bound_artifacts(
        self,
        root: Path,
        *,
        obligation: dict,
        target: dict,
        context: JourneyVerifierContext,
        execution_id: str,
    ) -> dict[str, Path]:
        identity = {
            "obligation_id": obligation["obligation_id"],
            "execution_id": execution_id,
            "revision": REVISION,
            "schedule_id": context.schedule.schedule_id,
            "verifier_input_contract_hash": content_hash(
                target["journey_definition"]["verifier_input_contract"]
            ),
        }
        payloads = {
            "transcript": {
                "turns": [
                    {
                        "turn": asdict(turn),
                        "decision": asdict(decision),
                    }
                    for turn, decision in zip(
                        context.completed_turns,
                        context.completed_decisions,
                    )
                ],
            },
            "runtime_events": {
                "initial_event": asdict(context.initial_event),
                "events": [
                    asdict(event)
                    for event in context.completed_events
                ],
            },
            "checkpoint_diff": {
                "before_fingerprint": (
                    context.initial_event.after_fingerprint
                ),
                "after_fingerprint": (
                    context.completed_events[-1].after_fingerprint
                ),
                "material_state_diff_hashes": [
                    dict(event.material_state_diff_hashes)
                    for event in context.completed_events
                ],
            },
        }
        paths: dict[str, Path] = {}
        for role, role_payload in payloads.items():
            unsigned = {
                "schema_version": 1,
                "artifact_type": f"retained_regression_{role}",
                "identity": identity,
                **role_payload,
            }
            artifact = {
                **unsigned,
                "artifact_hash": content_hash(unsigned),
            }
            path = root / f"{role}.json"
            path.write_text(
                json.dumps(artifact, sort_keys=True),
                encoding="utf-8",
            )
            paths[role] = path
        return paths

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
