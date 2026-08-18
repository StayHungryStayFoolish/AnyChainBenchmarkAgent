from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agent.harness.runtime_identity import repository_revision
from tests.agent_live.batch_orchestrator import (
    freeze_batch_manifest,
    validate_frozen_manifest,
)
from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
    formal_journey_definitions,
)
from tests.agent_live.coverage_evidence import PtyCliTurnRecord, RuntimeTurnEvent
from tests.agent_live.dynamic_dual_ai_chaos import JourneyVerifierContext
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


REVISION = {
    "commit_sha": "a" * 40,
    "worktree_diff_sha256": "b" * 64,
    "is_dirty": "false",
}


class FormalJourneyCatalogTest(unittest.TestCase):
    def test_catalog_has_eight_open_journeys_and_versioned_verifiers(self) -> None:
        rows = formal_journey_definitions()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["journey"]["journey_id"] for row in rows}), 8)
        self.assertTrue(FORMAL_JOURNEY_VERIFIER_REGISTRY.registry_id)
        self.assertTrue(all(
            definition.verifier_version == 2
            for definition in FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions.values()
        ))
        for row in rows:
            self.assertEqual(row["lane"], "journey")
            encoded = json.dumps(row, ensure_ascii=False)
            self.assertNotIn("user_message", encoded)
            self.assertNotIn("transcript", encoded)

    def test_formal_manifest_requires_exactly_24_edge_and_8_journey_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.name", "Test"), cwd=root, check=True)
            (root / ".gitignore").write_text(".agent/\n", encoding="utf-8")
            (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
            subprocess.run(("git", "add", "."), cwd=root, check=True)
            subprocess.run(("git", "commit", "-qm", "fixture"), cwd=root, check=True)
            revision = repository_revision(root)
            ledger = build_ledger(revision=revision)
            edge = next(
                row for row in ledger["edges"]
                if row.get("executable_scenario_ids")
            )
            targets = root / ".agent" / "targets"
            targets.mkdir(parents=True)
            for index in range(1, 25):
                (targets / f"{index:02d}.json").write_text(json.dumps({
                    "lane": "edge",
                    "targets": [{
                        "target_id": f"edge-{index}",
                        "edge_key": edge["edge_key"],
                        "persona": "operator",
                        "goal": "exercise one authoritative edge",
                        "scenario_id": edge["executable_scenario_ids"][0],
                    }],
                }), encoding="utf-8")
            for offset, row in enumerate(formal_journey_definitions(), start=25):
                (targets / f"{offset:02d}.json").write_text(
                    json.dumps(row, ensure_ascii=False), encoding="utf-8"
                )
            manifest = freeze_batch_manifest(
                repo_root=root,
                targets_dir=targets,
                manifest_path=root / ".agent" / "manifest.json",
                runtime_base=root / ".agent" / "runtime",
                formal_profile=True,
                worker_runtime="linux",
            )
            self.assertEqual([row.lane for row in manifest.shards].count("edge"), 24)
            self.assertEqual([row.lane for row in manifest.shards].count("journey"), 8)
            journey = manifest.shards[24]
            self.assertIn("tests.agent_live.journey_simulator_bridge", journey.command)
            self.assertEqual(
                journey.verifier_registry_id,
                FORMAL_JOURNEY_VERIFIER_REGISTRY.registry_id,
            )
            escaped = replace(
                manifest,
                shards=(
                    replace(manifest.shards[0], command=("python", "worker.py")),
                    *manifest.shards[1:],
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "escaped its Linux worker implementation"
            ):
                validate_frozen_manifest(escaped)


class FormalJourneyVerifierTest(unittest.TestCase):
    def _event(
        self,
        *,
        turn_index: int,
        before: str,
        after: str,
        active_group: str = "opening",
        pending: str = "",
        pending_contract: dict[str, object] | None = None,
        actions: tuple[str, ...] = (),
        targets: tuple[dict[str, str], ...] = (),
        diff: dict[str, dict[str, str]] | None = None,
        after_values: dict[str, str] | None = None,
        next_result: dict[str, object] | None = None,
    ) -> RuntimeTurnEvent:
        return RuntimeTurnEvent(
            schema_version=2,
            event_type="turn_committed",
            thread_id="journey",
            session_purpose="dynamic-dual-ai-chaos",
            before_fingerprint=before,
            after_fingerprint=after,
            turn_index=turn_index,
            active_group=active_group,
            pending_question_id=pending,
            action_queue_types=(),
            pending_contract=pending_contract or ({"id": pending} if pending else {}),
            revision=REVISION,
            admitted_action_types=actions,
            admitted_action_targets=targets,
            state_diff_hashes=diff or {},
            after_value_hashes=after_values or {},
            next_result=next_result or {"kind": "result", "response_count": 1},
        )

    def _turn(
        self,
        *,
        turn_index: int,
        before: str,
        after: str,
        response: str = "Agent> Completed.",
    ) -> PtyCliTurnRecord:
        return PtyCliTurnRecord(
            session_id="journey",
            turn_index=turn_index,
            previous_agent_response="Agent> Previous",
            user_message="response-driven input",
            agent_response=response,
            provider="deepseek",
            model="deepseek-chat",
            before_fingerprint=before,
            after_fingerprint=after,
            transcript_hash="c" * 64,
            previous_response_received_at_ns=1,
            user_message_submitted_at_ns=2,
            agent_response_received_at_ns=3,
        )

    def _context(
        self,
        *,
        initial: RuntimeTurnEvent,
        current: RuntimeTurnEvent,
        turns: tuple[PtyCliTurnRecord, ...],
        transcript: tuple[tuple[str, str], ...] | None = None,
    ) -> JourneyVerifierContext:
        latest = turns[-1] if turns else None
        return JourneyVerifierContext(
            schedule=SimpleNamespace(),
            initial_event=initial,
            current_event=current,
            completed_turns=turns,
            transcript=transcript or tuple(
                (turn.user_message, turn.agent_response) for turn in turns
            ),
            observed_edge_keys=(),
            latest_turn=latest,
        )

    def _verify(self, postcondition_id: str, context: JourneyVerifierContext):
        return FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions[
            postcondition_id
        ].verifier(context)

    def test_committed_state_rejects_fingerprint_only_and_unknown_action(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        turn = self._turn(turn_index=1, before="a", after="b")
        fingerprint_only = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("choose_target_mode",),
        )
        self.assertFalse(self._verify(
            "committed_state",
            self._context(initial=initial, current=fingerprint_only, turns=(turn,)),
        ).satisfied)
        unknown = replace(
            fingerprint_only,
            admitted_action_types=("invented_action",),
            state_diff_hashes={"target_mode": {"before": "1", "after": "2"}},
        )
        self.assertFalse(self._verify(
            "committed_state",
            self._context(initial=initial, current=unknown, turns=(turn,)),
        ).satisfied)

    def test_committed_state_requires_turn_bound_material_transition(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("choose_target_mode",),
            diff={"target_mode": {"before": "1", "after": "2"}},
        )
        turn = self._turn(turn_index=1, before="a", after="b")
        self.assertTrue(self._verify(
            "committed_state",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        mismatched_turn = replace(turn, after_fingerprint="c")
        self.assertFalse(self._verify(
            "committed_state",
            self._context(initial=initial, current=current, turns=(mismatched_turn,)),
        ).satisfied)

    def test_group_changed_requires_concrete_diff_and_action_provenance(self) -> None:
        initial = self._event(
            turn_index=0, before="0", after="a", active_group="environment"
        )
        turn = self._turn(turn_index=1, before="a", after="b")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            active_group="qps_profile",
            actions=("change_group",),
            targets=({"type": "change_group", "group": "qps_profile"},),
            diff={"active_group": {"before": "1", "after": "2"}},
            next_result={
                "kind": "question",
                "question_id": "next",
                "group": "qps_profile",
            },
            pending="next",
        )
        self.assertTrue(self._verify(
            "group_changed",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        wrong_target = replace(
            current,
            admitted_action_targets=({"type": "change_group", "group": "storage"},),
        )
        self.assertFalse(self._verify(
            "group_changed",
            self._context(initial=initial, current=wrong_target, turns=(turn,)),
        ).satisfied)
        no_diff = replace(current, state_diff_hashes={"turn_index": {"before": "1", "after": "2"}})
        self.assertFalse(self._verify(
            "group_changed",
            self._context(initial=initial, current=no_diff, turns=(turn,)),
        ).satisfied)

    def test_pending_advanced_requires_consumption_and_declared_action(self) -> None:
        initial_id_hash = "1" * 64
        initial = self._event(
            turn_index=0,
            before="0",
            after="a",
            pending="first",
            pending_contract={
                "id": "first",
                "accepted_action_types": ["answer_pending"],
            },
            after_values={"pending_question.id": initial_id_hash},
            next_result={"kind": "question", "question_id": "first"},
        )
        turn = self._turn(turn_index=1, before="a", after="b")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            pending="second",
            actions=("answer_pending",),
            diff={
                "pending_question.id": {
                    "before": initial_id_hash,
                    "after": "2" * 64,
                },
                "confirmed_config.CLOUD_REGION": {"before": "", "after": "3" * 64},
            },
            after_values={"pending_question.id": "2" * 64},
            next_result={"kind": "question", "question_id": "second"},
        )
        self.assertTrue(self._verify(
            "pending_advanced",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        question_id_only = replace(
            current,
            state_diff_hashes={
                "confirmed_config.CLOUD_REGION": {"before": "", "after": "3" * 64}
            },
        )
        self.assertFalse(self._verify(
            "pending_advanced",
            self._context(initial=initial, current=question_id_only, turns=(turn,)),
        ).satisfied)

    def test_multiple_actions_require_registered_mutations_in_two_domains(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        turn = self._turn(turn_index=1, before="a", after="b")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("choose_target_mode", "choose_chain"),
            diff={
                "target_mode": {"before": "", "after": "1"},
                "chain_identity.canonical": {"before": "", "after": "2"},
            },
        )
        self.assertTrue(self._verify(
            "multiple_actions_admitted",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        one_domain = replace(
            current,
            state_diff_hashes={
                "chain_identity.raw": {"before": "", "after": "1"},
                "chain_identity.canonical": {"before": "", "after": "2"},
            },
        )
        self.assertFalse(self._verify(
            "multiple_actions_admitted",
            self._context(initial=initial, current=one_domain, turns=(turn,)),
        ).satisfied)

    def test_rpc_verifier_uses_exact_action_and_catalog_state(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        turn = self._turn(turn_index=1, before="a", after="b")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("rpc_catalog_command",),
            diff={"custom_rpc.catalog.draft.method": {"before": "", "after": "1"}},
        )
        self.assertTrue(self._verify(
            "rpc_action_admitted",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        workload_only = replace(
            current,
            admitted_action_types=("rpc_workload_command",),
        )
        self.assertFalse(self._verify(
            "rpc_action_admitted",
            self._context(initial=initial, current=workload_only, turns=(turn,)),
        ).satisfied)
        name_contains_rpc = replace(
            current,
            admitted_action_types=("invented_rpc_action",),
        )
        self.assertFalse(self._verify(
            "rpc_action_admitted",
            self._context(initial=initial, current=name_contains_rpc, turns=(turn,)),
        ).satisfied)

    def test_result_requires_visible_response_and_read_only_provenance(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        turn = self._turn(turn_index=1, before="a", after="b")
        current = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("ask_capabilities",),
            diff={"turn_index": {"before": "1", "after": "2"}},
            next_result={"kind": "result", "response_count": 1},
        )
        self.assertTrue(self._verify(
            "result_returned",
            self._context(initial=initial, current=current, turns=(turn,)),
        ).satisfied)
        no_response = replace(current, next_result={"kind": "result", "response_count": 0})
        self.assertFalse(self._verify(
            "result_returned",
            self._context(initial=initial, current=no_response, turns=(turn,)),
        ).satisfied)

    def test_loop_detection_normalizes_terminal_noise_and_detects_short_cycle(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        current = self._event(turn_index=2, before="b", after="c")
        first = self._turn(
            turn_index=1,
            before="a",
            after="b",
            response="Agent> [thinking] waiting\nAgent> Choose   one",
        )
        second = self._turn(
            turn_index=2,
            before="b",
            after="c",
            response="\x1b[32mAgent> choose one\x1b[0m",
        )
        adjacent = self._context(
            initial=initial,
            current=current,
            turns=(first, second),
        )
        self.assertTrue(self._verify("mechanical_response_loop", adjacent).satisfied)
        responses = (
            ("u1", "Agent> first"),
            ("u2", "Agent> second"),
            ("u3", "Agent> FIRST"),
            ("u4", "Agent> second"),
        )
        turns = (
            self._turn(turn_index=1, before="a", after="b", response="Agent> first"),
            self._turn(turn_index=2, before="b", after="c", response="Agent> second"),
            self._turn(turn_index=3, before="c", after="d", response="Agent> FIRST"),
            self._turn(turn_index=4, before="d", after="e", response="Agent> second"),
        )
        cycle = self._context(
            initial=initial,
            current=self._event(turn_index=4, before="d", after="e"),
            turns=turns,
            transcript=responses,
        )
        self.assertTrue(self._verify("mechanical_response_loop", cycle).satisfied)

    def test_state_regression_detects_broken_lineage_or_exact_rollback(self) -> None:
        initial = self._event(turn_index=0, before="0", after="a")
        baseline = self._context(initial=initial, current=initial, turns=())
        self.assertFalse(self._verify("state_regressed", baseline).satisfied)
        first = self._turn(turn_index=1, before="a", after="b")
        healthy_event = self._event(
            turn_index=1,
            before="a",
            after="b",
            actions=("choose_target_mode",),
            diff={"target_mode": {"before": "", "after": "1"}},
        )
        self.assertFalse(self._verify(
            "state_regressed",
            self._context(initial=initial, current=healthy_event, turns=(first,)),
        ).satisfied)
        broken = replace(first, before_fingerprint="x")
        self.assertTrue(self._verify(
            "state_regressed",
            self._context(initial=initial, current=healthy_event, turns=(broken,)),
        ).satisfied)
        rollback_turn = self._turn(turn_index=2, before="b", after="a")
        rollback_event = self._event(
            turn_index=2,
            before="b",
            after="a",
            actions=("choose_target_mode",),
            diff={"target_mode": {"before": "1", "after": ""}},
        )
        self.assertTrue(self._verify(
            "state_regressed",
            self._context(
                initial=initial,
                current=rollback_event,
                turns=(first, rollback_turn),
            ),
        ).satisfied)


if __name__ == "__main__":
    unittest.main()
