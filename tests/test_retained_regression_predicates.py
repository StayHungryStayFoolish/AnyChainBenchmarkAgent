"""Unit tests for shared retained-regression semantic predicates."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.harness.action_registry import MODE_COMPARISON_TOPIC
from agent.harness.domains.rpc_receipts import evidence_hash
from agent.harness.domains.analysis_receipts import analysis_hash
from tests.agent_live.coverage_evidence import RuntimeTurnEvent, content_hash
from tests.agent_live.retained_regression_attestations import (
    build_source_contract,
    build_variant_attestation,
    build_variant_contract,
)
from tests.agent_live.retained_regression_obligations import (
    KNOWN_POSTCONDITION_IDS,
)
from tests.agent_live.retained_regression_predicates import (
    POSTCONDITION_EVALUATORS,
    UNSUPPORTED_WITHOUT_IMMUTABLE_VARIANT_CONTRACT,
)


HASH = "a" * 64


def _hashed(body: dict) -> dict:
    return {**body, "receipt_id": content_hash(body)}


def _value_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _event(
    *receipts: dict,
    turn_index: int = 1,
    pending_transition: dict | None = None,
    turn_receipt: dict | None = None,
    admitted_actions: tuple[dict, ...] = (),
    material_diffs: dict | None = None,
    execution: dict | None = None,
) -> RuntimeTurnEvent:
    transition = pending_transition or {
        "transition": "absent",
        "before_id": "",
        "before_group": "",
        "before_hash": "1" * 64,
        "after_id": "",
        "after_group": "",
        "after_hash": "2" * 64,
        "consumer_action_ids": [],
    }
    return RuntimeTurnEvent(
        schema_version=3,
        event_type="turn_committed",
        thread_id="predicate-test",
        session_purpose="test",
        before_fingerprint="3" * 64,
        after_fingerprint="4" * 64,
        turn_index=turn_index,
        active_group="test_group",
        pending_question_id=str(transition.get("after_id") or ""),
        action_queue_types=(),
        revision={"commit": "test", "worktree_hash": "5" * 64},
        admitted_action_provenance=admitted_actions,
        turn_receipt_summary=turn_receipt or {},
        pending_transition=transition,
        control_receipts=tuple(receipts),
        material_state_diff_hashes=material_diffs or {},
        execution_receipt_summary=execution or {},
    )


def _context(*events: RuntimeTurnEvent, **values) -> SimpleNamespace:
    return SimpleNamespace(
        completed_events=events,
        completed_turns=tuple(values.get("turns") or ()),
        completed_decisions=tuple(values.get("decisions") or ()),
        verifier_input_contract=values.get("verifier_input_contract") or {},
        initial_event=values.get("initial_event"),
        schedule=values.get("schedule") or SimpleNamespace(start_scenario=""),
    )


def _verifier_input(
    variant: str,
    messages: tuple[str, ...],
    *,
    semantic_roles: tuple[str, ...] | None = None,
) -> dict:
    roles = semantic_roles or tuple("source_intent" for _ in messages)
    source = build_source_contract({
        "case_id": "contract-test",
        "source_turns_hash": content_hash(messages),
        "source_steps": [
            {
                "step_id": f"source-{index}",
                "turn_index": index,
                "semantic_role": role,
            }
            for index, role in enumerate(roles, start=1)
        ],
    })
    declarations = {
        "exact": ("exact_fixture", 0),
        "isomorphic": ("isomorphic_meaning", 1),
        "negative": ("adjacent_non_trigger", 1),
        "neighboring": ("neighboring_transition", 1),
    }
    relation, minimum = declarations[variant]
    variant_contract = build_variant_contract(
        variant=variant,
        source_contract=source,
        declaration={
            "relation": relation,
            "minimum_attestations": minimum,
        },
    )
    modes = {
        "exact": "exact_fixture_replay",
        "isomorphic": "response_driven_isomorphic",
        "negative": "response_driven_negative",
        "neighboring": "response_driven_neighboring",
    }
    return {
        "mode": modes[variant],
        "source_contract": source,
        "source_contract_hash": content_hash(source),
        "variant_contract": variant_contract,
        "variant_contract_hash": content_hash(variant_contract),
    }


def _turn_and_decision(
    message: str,
    *,
    turn_index: int = 1,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    previous = "complete response"
    turn = SimpleNamespace(
        turn_index=turn_index,
        previous_agent_response=previous,
        user_message=message,
    )
    decision = SimpleNamespace(
        turn_index=turn_index,
        previous_response_hash=content_hash(previous),
        user_message_hash=content_hash(message),
        selected_at_ns=10,
        submitted_at_ns=20,
        execution_id="execution-1",
        obligation_id="obligation-1",
        broker_request_id=f"request-{turn_index}",
        simulator_context_binding={
            "session_id": "session-1",
            "control_identity": {
                "lane": "journey",
                "schedule_id": "schedule-1",
            },
        },
        simulator_attestation={
            "attestation_id": f"simulator-{turn_index}",
            "actor": {
                "actor_kind": "codex",
                "task_id": "task-1",
                "model": "gpt-5",
            },
        },
        variant_attestation={},
    )
    return turn, decision


def _execution_binding(decision: SimpleNamespace) -> dict[str, str]:
    return {
        "execution_id": decision.execution_id,
        "session_id": decision.simulator_context_binding["session_id"],
        "schedule_id": decision.simulator_context_binding[
            "control_identity"
        ]["schedule_id"],
        "obligation_id": decision.obligation_id,
        "broker_request_id": decision.broker_request_id,
        "simulator_attestation_id": decision.simulator_attestation[
            "attestation_id"
        ],
    }


def _with_attestation(
    decision: SimpleNamespace,
    attestation: dict,
) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **vars(decision),
            "variant_attestation": attestation,
        }
    )


def _pending_receipt(**updates) -> dict:
    body = {
        "receipt_type": "pending_resolution",
        "turn_index": 1,
        "pending_id": "NETWORK_INTERFACE",
        "pending_group": "network",
        "pending_contract_hash": "1" * 64,
        "resolution_path": "exact_contract",
        "selected_option_id": "use-detected",
        "selected_value_hash": "2" * 64,
        "resolved_action_id": "action-1",
        "input_hash": "3" * 64,
        "normalizer": "exact_contract",
        "verdict": "accepted",
    }
    body.update(updates)
    return _hashed(body)


def _domain_commit(
    *,
    turn_index: int = 1,
    action_id: str = "action-1",
    paths: tuple[str, ...] = (),
) -> dict:
    return _hashed({
        "receipt_type": "domain_commit",
        "turn_index": turn_index,
        "owner": "environment",
        "completion": "in_progress",
        "group_registry_contract_hash": "4" * 64,
        "pending_before_hash": "1" * 64,
        "pending_after_hash": "6" * 64,
        "consumed_action_ids": [action_id],
        "invalidated_groups": [],
        "invalidated_fields": [],
        "response_fragments": [],
        "reconfigured_groups": [],
        "group_state_transitions": [],
        "material_delta": [
            {
                "operation": "write",
                "path": path,
                "value_hash": "7" * 64,
            }
            for path in paths
        ],
        "navigation_operation": "",
        "navigation_origin_group": "",
        "navigation_target_group": "",
        "pending_after_id": "",
    })


def _confirmed_field_context(
    *,
    message: str,
    role: str,
    field: str,
    resolution_path: str = "typed_manual_value",
    option_id: str = "",
    normalizer: str = "semantic_scalar",
    committed_field: str | None = None,
    admitted: bool = True,
    commit_owner: str = "environment",
    transition_before_hash: str = "1" * 64,
) -> SimpleNamespace:
    turn, _ = _turn_and_decision(message)
    receipt = _pending_receipt(
        pending_id=field,
        pending_group="ledger_disk",
        resolution_path=resolution_path,
        selected_option_id=option_id,
        normalizer=normalizer,
        input_hash=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        selected_value_hash="7" * 64,
    )
    commit = _domain_commit(
        paths=(f"confirmed_config.{committed_field or field}",),
    )
    if commit_owner != "environment":
        commit = _hashed({
            **{
                key: value for key, value in commit.items()
                if key != "receipt_id"
            },
            "owner": commit_owner,
        })
    return _context(
        _event(
            receipt,
            commit,
            pending_transition={
                "transition": "consumed",
                "before_id": field,
                "before_group": "ledger_disk",
                "before_hash": transition_before_hash,
                "after_id": "",
                "after_group": "",
                "after_hash": "6" * 64,
                "consumer_action_ids": ["action-1"],
            },
            turn_receipt={
                "admitted_action_ids": ["action-1"] if admitted else [],
                "owner_bindings": {"action-1": "environment"},
            },
            material_diffs={
                f"confirmed_config.{committed_field or field}": {
                    "before": "",
                    "after": "7" * 64,
                },
            },
        ),
        turns=(turn,),
        verifier_input_contract=_verifier_input(
            "exact",
            (message,),
            semantic_roles=(role,),
        ),
    )


def _workload_receipt(**updates) -> dict:
    methods = ["eth_accounts"]
    body = {
        "receipt_type": "rpc_workload_commit",
        "receipt_version": 1,
        "turn_index": 1,
        "owner": "rpc_workload",
        "case": "custom_rpc",
        "rpc_mode": "single",
        "choice": "custom",
        "methods": methods,
        "method_hashes": [evidence_hash(method) for method in methods],
        "mixed_weight_entries": [],
        "replace_defaults": True,
        "job_local_override": True,
        "catalog_revision": 1,
        "fixture_required": True,
        "fixture_status": "recorded",
    }
    body.update(updates)
    return {**body, "receipt_id": evidence_hash(body)}


class RetainedRegressionPredicatesTest(unittest.TestCase):
    def test_disk_size_resolution_accepts_detected_or_manual_capacity(self) -> None:
        predicate = POSTCONDITION_EVALUATORS["disk_size_resolved"]
        for resolution_path, option_id in (
            ("typed_manual_value", ""),
            ("exact_contract", "1"),
        ):
            with self.subTest(resolution_path=resolution_path):
                context = _confirmed_field_context(
                    message="926" if not option_id else "1",
                    role="provide_disk_size",
                    field="DATA_VOL_SIZE",
                    resolution_path=resolution_path,
                    option_id=option_id,
                    normalizer="semantic_scalar",
                )
                satisfied, details = predicate(context)
                self.assertTrue(satisfied, details)

    def test_disk_size_resolution_rejects_unrelated_or_invalid_receipts(
        self,
    ) -> None:
        predicate = POSTCONDITION_EVALUATORS["disk_size_resolved"]
        unrelated = _confirmed_field_context(
            message="1",
            role="provide_disk_size",
            field="NETWORK_INTERFACE",
            resolution_path="exact_contract",
            option_id="1",
        )
        satisfied, details = predicate(unrelated)
        self.assertFalse(satisfied, details)

        unbound_confirmation = _confirmed_field_context(
            message="1",
            role="provide_disk_size",
            field="DATA_VOL_SIZE",
            resolution_path="exact_contract",
            option_id="",
        )
        satisfied, details = predicate(unbound_confirmation)
        self.assertFalse(satisfied, details)

        wrong_commit = _confirmed_field_context(
            message="926",
            role="provide_disk_size",
            field="DATA_VOL_SIZE",
            committed_field="ACCOUNTS_VOL_SIZE",
        )
        satisfied, details = predicate(wrong_commit)
        self.assertFalse(satisfied, details)

        wrong_role = _confirmed_field_context(
            message="926",
            role="provide_disk_iops",
            field="DATA_VOL_SIZE",
        )
        satisfied, details = predicate(wrong_role)
        self.assertFalse(satisfied, details)

        unadmitted = _confirmed_field_context(
            message="926",
            role="provide_disk_size",
            field="DATA_VOL_SIZE",
            admitted=False,
        )
        satisfied, details = predicate(unadmitted)
        self.assertFalse(satisfied, details)

        wrong_owner = _confirmed_field_context(
            message="926",
            role="provide_disk_size",
            field="DATA_VOL_SIZE",
            commit_owner="chain_rpc",
        )
        satisfied, details = predicate(wrong_owner)
        self.assertFalse(satisfied, details)

        wrong_pending_contract = _confirmed_field_context(
            message="926",
            role="provide_disk_size",
            field="DATA_VOL_SIZE",
            transition_before_hash="8" * 64,
        )
        satisfied, details = predicate(wrong_pending_contract)
        self.assertFalse(satisfied, details)

    def test_disk_scalar_and_limits_require_user_confirmed_field_lineage(
        self,
    ) -> None:
        normalized, details = POSTCONDITION_EVALUATORS[
            "copied_scalar_normalized"
        ](_confirmed_field_context(
            message="hyperdisk-balanced,",
            role="provide_disk_type",
            field="DATA_VOL_TYPE",
        ))
        self.assertTrue(normalized, details)

        turns = []
        events = []
        messages = ("20000", "1000")
        roles = ("provide_disk_iops", "provide_disk_throughput")
        fields = ("DATA_VOL_MAX_IOPS", "DATA_VOL_MAX_THROUGHPUT")
        for index, (message, field) in enumerate(
            zip(messages, fields, strict=True),
            start=1,
        ):
            turn, _ = _turn_and_decision(message, turn_index=index)
            turns.append(turn)
            action_id = f"action-{index}"
            events.append(_event(
                _pending_receipt(
                    turn_index=index,
                    pending_id=field,
                    pending_group="ledger_disk",
                    resolved_action_id=action_id,
                    resolution_path="typed_manual_value",
                    selected_option_id="",
                    normalizer="semantic_scalar",
                    input_hash=hashlib.sha256(message.encode("utf-8")).hexdigest(),
                    selected_value_hash="7" * 64,
                ),
                _domain_commit(
                    turn_index=index,
                    action_id=action_id,
                    paths=(f"confirmed_config.{field}",),
                ),
                turn_index=index,
                pending_transition={
                    "transition": "consumed",
                    "before_id": field,
                    "before_group": "ledger_disk",
                    "before_hash": "1" * 64,
                    "after_id": "",
                    "after_group": "",
                    "after_hash": "6" * 64,
                    "consumer_action_ids": [action_id],
                },
                turn_receipt={
                    "admitted_action_ids": [action_id],
                    "owner_bindings": {action_id: "environment"},
                },
                material_diffs={
                    f"confirmed_config.{field}": {
                        "before": "",
                        "after": "7" * 64,
                    },
                },
            ))
        context = _context(
            *events,
            turns=tuple(turns),
            verifier_input_contract=_verifier_input(
                "exact",
                messages,
                semantic_roles=roles,
            ),
        )
        collected, details = POSTCONDITION_EVALUATORS[
            "disk_limits_collected_once"
        ](context)
        self.assertTrue(collected, details)
        repeated, details = POSTCONDITION_EVALUATORS[
            "disk_subgroup_repeated"
        ](context)
        self.assertFalse(repeated, details)

        inferred_only = _context(_event(_domain_commit(
            paths=("inferred_config.DATA_VOL_MAX_IOPS",),
        )))
        collected, details = POSTCONDITION_EVALUATORS[
            "disk_limits_collected_once"
        ](inferred_only)
        self.assertFalse(collected, details)

    def test_environment_retention_distinguishes_durable_and_invocation_state(
        self,
    ) -> None:
        mode_commit = _domain_commit(paths=("target_mode",))
        clean = _context(_event(mode_commit))
        retained, details = POSTCONDITION_EVALUATORS[
            "compatible_environment_retained"
        ](clean)
        self.assertTrue(retained, details)

        discovery_mutated = _context(_event(
            mode_commit,
            material_diffs={
                "discovery.disk_candidates": {
                    "before": "8" * 64,
                    "after": "9" * 64,
                },
            },
        ))
        retained, details = POSTCONDITION_EVALUATORS[
            "compatible_environment_retained"
        ](discovery_mutated)
        self.assertTrue(retained, details)

        durable_mutated = _context(_event(
            mode_commit,
            material_diffs={
                "inferred_config.DATA_VOL_SIZE": {
                    "before": "8" * 64,
                    "after": "9" * 64,
                },
            },
        ))
        retained, details = POSTCONDITION_EVALUATORS[
            "compatible_environment_retained"
        ](durable_mutated)
        self.assertFalse(retained, details)

        body = {
            key: value
            for key, value in mode_commit.items()
            if key != "receipt_id"
        }
        body["navigation_operation"] = "go_back"
        body["navigation_origin_group"] = "qps_profile"
        body["navigation_target_group"] = "workload_rpc"
        body["material_delta"] = [{
            "operation": "delete",
            "path": "confirmed_config.DATA_VOL_SIZE",
            "value_hash": "",
        }]
        lost, details = POSTCONDITION_EVALUATORS[
            "backtrack_lost_configuration_state"
        ](_context(_event(_hashed(body))))
        self.assertTrue(lost, details)

    def test_mapping_covers_known_ids_and_implements_variant_claims(
        self,
    ) -> None:
        self.assertGreaterEqual(len(POSTCONDITION_EVALUATORS), 50)
        self.assertLessEqual(
            set(POSTCONDITION_EVALUATORS),
            set(KNOWN_POSTCONDITION_IDS),
        )
        self.assertEqual(
            UNSUPPORTED_WITHOUT_IMMUTABLE_VARIANT_CONTRACT,
            frozenset(),
        )
        self.assertTrue(
            {
                "exact_fixture_turns_observed",
                "isomorphic_meaning_attested",
                "adjacent_non_trigger_attested",
                "neighboring_transition_attested",
                "environment_text_consumed_as_region",
            }.issubset(POSTCONDITION_EVALUATORS)
        )

    def test_exact_fixture_hash_rejects_reordered_or_duplicate_turns(self) -> None:
        messages = ("first", "second")
        contract = _verifier_input("exact", messages)
        turns = tuple(_turn_and_decision(message, turn_index=index)[0]
                      for index, message in enumerate(messages, start=1))
        satisfied, details = POSTCONDITION_EVALUATORS[
            "exact_fixture_turns_observed"
        ](_context(turns=turns, verifier_input_contract=contract))
        self.assertTrue(satisfied, details)

        reordered = tuple(reversed(turns))
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "exact_fixture_turns_observed"
        ](_context(turns=reordered, verifier_input_contract=contract))
        self.assertFalse(satisfied)
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "exact_fixture_turns_observed"
        ](_context(
            turns=(*turns, turns[-1]),
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied)

    def test_open_variant_attestations_are_contract_and_response_bound(
        self,
    ) -> None:
        predicate_by_variant = {
            "isomorphic": "isomorphic_meaning_attested",
            "negative": "adjacent_non_trigger_attested",
            "neighboring": "neighboring_transition_attested",
        }
        for variant, predicate_id in predicate_by_variant.items():
            with self.subTest(variant=variant):
                contract = _verifier_input(variant, ("source",))
                turn, decision = _turn_and_decision("selected")
                attestation = build_variant_attestation(
                    actor={
                        "actor_kind": "codex",
                        "task_id": "task-1",
                        "model": "gpt-5",
                    },
                    verifier_input_contract=contract,
                    execution_binding=_execution_binding(decision),
                    source_step_id="source-1",
                    semantic_role="source_intent",
                    turn_index=1,
                    previous_response_hash=decision.previous_response_hash,
                    user_message_hash=decision.user_message_hash,
                    selected_at_ns=10,
                    declared_at_ns=15,
                )
                context = _context(
                    turns=(turn,),
                    decisions=(_with_attestation(decision, attestation),),
                    verifier_input_contract=contract,
                )
                satisfied, details = POSTCONDITION_EVALUATORS[
                    predicate_id
                ](context)
                self.assertTrue(satisfied, details)

                stale = dict(attestation)
                stale["user_message_hash"] = "0" * 64
                stale["attestation_id"] = content_hash({
                    key: value for key, value in stale.items()
                    if key != "attestation_id"
                })
                satisfied, details = POSTCONDITION_EVALUATORS[
                    predicate_id
                ](_context(
                    turns=(turn,),
                    decisions=(_with_attestation(decision, stale),),
                    verifier_input_contract=contract,
                ))
                self.assertFalse(satisfied)
                self.assertTrue(details["invalid_attestations"])

                cross_execution = dict(attestation)
                cross_execution["execution_binding"] = {
                    **cross_execution["execution_binding"],
                    "execution_id": "other-execution",
                }
                cross_execution["attestation_id"] = content_hash({
                    key: value
                    for key, value in cross_execution.items()
                    if key != "attestation_id"
                })
                satisfied, details = POSTCONDITION_EVALUATORS[
                    predicate_id
                ](_context(
                    turns=(turn,),
                    decisions=(
                        _with_attestation(
                            decision,
                            cross_execution,
                        ),
                    ),
                    verifier_input_contract=contract,
                ))
                self.assertFalse(satisfied)
                self.assertTrue(details["invalid_attestations"])

    def test_neighboring_attestation_rejects_missing_duplicate_and_role_drift(
        self,
    ) -> None:
        contract = _verifier_input(
            "neighboring",
            ("source-a", "source-b"),
            semantic_roles=("source_intent", "chain_change_request"),
        )
        first_turn, first_decision = _turn_and_decision("selected-a")
        second_turn, second_decision = _turn_and_decision(
            "selected-b",
            turn_index=2,
        )

        def attestation(
            *,
            step_id: str,
            role: str,
            turn: SimpleNamespace,
            decision: SimpleNamespace,
        ) -> dict:
            return build_variant_attestation(
                actor={
                    "actor_kind": "codex",
                    "task_id": "task-1",
                    "model": "gpt-5",
                },
                verifier_input_contract=contract,
                execution_binding=_execution_binding(decision),
                source_step_id=step_id,
                semantic_role=role,
                turn_index=turn.turn_index,
                previous_response_hash=decision.previous_response_hash,
                user_message_hash=decision.user_message_hash,
                selected_at_ns=10,
                declared_at_ns=15,
            )

        first = attestation(
            step_id="source-1",
            role="source_intent",
            turn=first_turn,
            decision=first_decision,
        )
        second = attestation(
            step_id="source-2",
            role="chain_change_request",
            turn=second_turn,
            decision=second_decision,
        )
        predicate = POSTCONDITION_EVALUATORS[
            "neighboring_transition_attested"
        ]
        satisfied, details = predicate(_context(
            turns=(first_turn, second_turn),
            decisions=(
                _with_attestation(first_decision, first),
                _with_attestation(second_decision, second),
            ),
            verifier_input_contract=contract,
        ))
        self.assertTrue(satisfied, details)

        satisfied, details = predicate(_context(
            turns=(first_turn, second_turn),
            decisions=(
                _with_attestation(first_decision, first),
                second_decision,
            ),
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied)
        self.assertEqual(details["missing_source_step_ids"], ["source-2"])

        duplicate = attestation(
            step_id="source-1",
            role="source_intent",
            turn=second_turn,
            decision=second_decision,
        )
        satisfied, details = predicate(_context(
            turns=(first_turn, second_turn),
            decisions=(
                _with_attestation(first_decision, first),
                _with_attestation(second_decision, duplicate),
            ),
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied)
        self.assertEqual(details["duplicate_source_step_ids"], ["source-1"])

        wrong_role = dict(second)
        wrong_role["semantic_role"] = "source_intent"
        wrong_role["attestation_id"] = content_hash({
            key: value for key, value in wrong_role.items()
            if key != "attestation_id"
        })
        satisfied, details = predicate(_context(
            turns=(first_turn, second_turn),
            decisions=(
                _with_attestation(first_decision, first),
                _with_attestation(second_decision, wrong_role),
            ),
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied)
        self.assertTrue(details["invalid_attestations"])

        wrong_relation = dict(second)
        wrong_relation["relation"] = "adjacent_non_trigger"
        wrong_relation["attestation_id"] = content_hash({
            key: value for key, value in wrong_relation.items()
            if key != "attestation_id"
        })
        satisfied, details = predicate(_context(
            turns=(first_turn, second_turn),
            decisions=(
                _with_attestation(first_decision, first),
                _with_attestation(second_decision, wrong_relation),
            ),
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied)
        self.assertTrue(details["invalid_attestations"])

        wrong_actor = dict(second)
        wrong_actor["actor"] = {
            **second["actor"],
            "actor_kind": "script",
        }
        wrong_actor["attestation_id"] = content_hash({
            key: value for key, value in wrong_actor.items()
            if key != "attestation_id"
        })
        stale_hash = dict(second)
        stale_hash["variant_contract_hash"] = "0" * 64
        stale_hash["attestation_id"] = content_hash({
            key: value for key, value in stale_hash.items()
            if key != "attestation_id"
        })
        for invalid_attestation in (wrong_actor, stale_hash):
            satisfied, details = predicate(_context(
                turns=(first_turn, second_turn),
                decisions=(
                    _with_attestation(first_decision, first),
                    _with_attestation(second_decision, invalid_attestation),
                ),
                verifier_input_contract=contract,
            ))
            self.assertFalse(satisfied)
            self.assertTrue(details["invalid_attestations"])

    def test_environment_region_violation_requires_independent_semantic_binding(
        self,
    ) -> None:
        message = "selected chain-change intent"
        contract = _verifier_input(
            "isomorphic",
            ("source",),
            semantic_roles=("chain_change_request",),
        )
        turn, decision = _turn_and_decision(message)
        attestation = build_variant_attestation(
            actor={
                "actor_kind": "codex",
                "task_id": "task-1",
                "model": "gpt-5",
            },
            verifier_input_contract=contract,
            execution_binding=_execution_binding(decision),
            source_step_id="source-1",
            semantic_role="chain_change_request",
            turn_index=1,
            previous_response_hash=decision.previous_response_hash,
            user_message_hash=decision.user_message_hash,
            selected_at_ns=10,
            declared_at_ns=15,
        )
        action_id = "action-1"
        receipt = _pending_receipt(
            pending_id="CLOUD_REGION",
            pending_group="provider_deployment",
            resolution_path="typed_manual_value",
            selected_option_id="",
            selected_value_hash="2" * 64,
            resolved_action_id=action_id,
            input_hash=hashlib.sha256(message.encode("utf-8")).hexdigest(),
            normalizer="declared_value_type",
        )
        transition = {
            "transition": "consumed",
            "before_id": "CLOUD_REGION",
            "before_group": "provider_deployment",
            "before_hash": "1" * 64,
            "after_id": "",
            "after_group": "",
            "after_hash": "2" * 64,
            "consumer_action_ids": [action_id],
        }
        event = _event(
            receipt,
            pending_transition=transition,
            turn_receipt={"admitted_action_ids": [action_id]},
        )
        context = _context(
            event,
            turns=(turn,),
            decisions=(_with_attestation(decision, attestation),),
            verifier_input_contract=contract,
        )
        observed, details = POSTCONDITION_EVALUATORS[
            "environment_text_consumed_as_region"
        ](context)
        self.assertTrue(observed, details)

        unrelated_contract = _verifier_input("isomorphic", ("source",))
        observed, _ = POSTCONDITION_EVALUATORS[
            "environment_text_consumed_as_region"
        ](_context(
            event,
            turns=(turn,),
            decisions=(decision,),
            verifier_input_contract=unrelated_contract,
        ))
        self.assertFalse(observed)

        confirmation = "Y"
        confirmation_contract = _verifier_input(
            "exact",
            (confirmation,),
            semantic_roles=("confirmation",),
        )
        confirmation_turn, _ = _turn_and_decision(confirmation)
        confirmation_receipt = _pending_receipt(
            pending_id="CLOUD_REGION",
            pending_group="provider_deployment",
            resolution_path="typed_manual_value",
            selected_option_id="",
            selected_value_hash="3" * 64,
            resolved_action_id=action_id,
            input_hash=hashlib.sha256(
                confirmation.encode("utf-8")
            ).hexdigest(),
            normalizer="declared_value_type",
        )
        confirmation_event = _event(
            confirmation_receipt,
            pending_transition=transition,
            turn_receipt={"admitted_action_ids": [action_id]},
        )
        observed, details = POSTCONDITION_EVALUATORS[
            "environment_text_consumed_as_region"
        ](_context(
            confirmation_event,
            turns=(confirmation_turn,),
            verifier_input_contract=confirmation_contract,
        ))
        self.assertFalse(observed, details)

        request = "change chain and mode"
        request_turn, _ = _turn_and_decision(request, turn_index=1)
        confirmation_turn, _ = _turn_and_decision(
            confirmation,
            turn_index=2,
        )
        paired_contract = _verifier_input(
            "exact",
            (request, confirmation),
            semantic_roles=(
                "chain_mode_change_request",
                "confirmation",
            ),
        )
        paired_receipt = _pending_receipt(
            turn_index=2,
            pending_id="CLOUD_REGION",
            pending_group="provider_deployment",
            resolution_path="typed_manual_value",
            selected_option_id="",
            selected_value_hash="3" * 64,
            resolved_action_id=action_id,
            input_hash=hashlib.sha256(
                confirmation.encode("utf-8")
            ).hexdigest(),
            normalizer="declared_value_type",
        )
        paired_event = _event(
            paired_receipt,
            turn_index=2,
            pending_transition=transition,
            turn_receipt={"admitted_action_ids": [action_id]},
        )
        observed, details = POSTCONDITION_EVALUATORS[
            "environment_text_consumed_as_region"
        ](_context(
            paired_event,
            turns=(request_turn, confirmation_turn),
            verifier_input_contract=paired_contract,
        ))
        self.assertTrue(observed, details)

    def test_chain_mode_change_requires_compound_transaction_evidence(
        self,
    ) -> None:
        messages = (
            "change chain",
            "Y",
            "change chain and mode",
            "Y",
        )
        contract = _verifier_input(
            "exact",
            messages,
            semantic_roles=(
                "chain_change_request",
                "confirmation",
                "chain_mode_change_request",
                "confirmation",
            ),
        )
        turns = tuple(
            _turn_and_decision(message, turn_index=index + 2)[0]
            for index, message in enumerate(messages, start=1)
        )
        prior_chain_change = _event(
            turn_index=4,
            material_diffs={
                "chain_identity.canonical": {
                    "before": "1" * 64,
                    "after": "2" * 64,
                },
            },
        )
        compound = _event(
            turn_index=5,
            admitted_actions=(
                {
                    "type": "change_chain",
                    "action_id": "chain-action",
                    "owner": "chain_rpc",
                },
                {
                    "type": "request_target_mode_selection",
                    "action_id": "mode-action",
                    "owner": "chain_rpc",
                },
            ),
        )
        confirmation = _event(
            _domain_commit(
                turn_index=6,
                action_id="chain-action",
                paths=("chain_identity.change_candidate.canonical",),
            ),
            turn_index=6,
            pending_transition={
                "transition": "replaced",
                "before_id": "chain_change_confirm",
                "before_group": "chain_identity",
                "before_hash": "3" * 64,
                "after_id": "target_mode_select",
                "after_group": "target_mode",
                "after_hash": "4" * 64,
                "consumer_action_ids": ["chain-action", "mode-action"],
            },
            material_diffs={
                "chain_identity.canonical": {
                    "before": "2" * 64,
                    "after": "5" * 64,
                },
            },
        )
        predicate = POSTCONDITION_EVALUATORS[
            "chain_mode_change_confirmed"
        ]

        satisfied, details = predicate(_context(
            prior_chain_change,
            compound,
            confirmation,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertTrue(satisfied, details)

        satisfied, details = predicate(_context(
            prior_chain_change,
            replace(compound, admitted_action_provenance=()),
            confirmation,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied, details)

        mode_only_confirmation = _event(
            _domain_commit(
                turn_index=6,
                action_id="mode-action",
                paths=("target_mode",),
            ),
            turn_index=6,
        )
        satisfied, details = predicate(_context(
            prior_chain_change,
            compound,
            mode_only_confirmation,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied, details)

        chain_only_confirmation = _event(
            _domain_commit(
                turn_index=6,
                action_id="chain-action",
                paths=("chain_identity.canonical",),
            ),
            turn_index=6,
        )
        satisfied, details = predicate(_context(
            prior_chain_change,
            compound,
            chain_only_confirmation,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied, details)

        premature_direct_commits = _event(
            _domain_commit(
                turn_index=5,
                action_id="chain-action",
                paths=("chain_identity.canonical",),
            ),
            _domain_commit(
                turn_index=5,
                action_id="mode-action",
                paths=("target_mode",),
            ),
            turn_index=5,
            admitted_actions=compound.admitted_action_provenance,
        )
        satisfied, details = predicate(_context(
            prior_chain_change,
            premature_direct_commits,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied, details)

        wrong_mode_lineage = replace(
            confirmation,
            pending_transition={
                **confirmation.pending_transition,
                "consumer_action_ids": [
                    "chain-action",
                    "unrelated-mode-action",
                ],
            },
        )
        satisfied, details = predicate(_context(
            prior_chain_change,
            compound,
            wrong_mode_lineage,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(satisfied, details)

        compound_with_mode_transition = _event(
            turn_index=5,
            admitted_actions=compound.admitted_action_provenance,
            pending_transition={
                "transition": "replaced",
                "before_id": "CLOUD_REGION",
                "before_group": "provider_deployment",
                "before_hash": "3" * 64,
                "after_id": "target_mode_change_confirm",
                "after_group": "target_mode",
                "after_hash": "4" * 64,
                "consumer_action_ids": [
                    "chain-action",
                    "mode-action",
                ],
            },
        )
        staged_chain_and_mode = _event(
            _domain_commit(
                turn_index=6,
                action_id="chain-action",
                paths=("chain_identity.change_candidate.canonical",),
            ),
            turn_index=6,
            pending_transition={
                "transition": "replaced",
                "before_id": "target_mode_change_confirm",
                "before_group": "target_mode",
                "before_hash": "3" * 64,
                "after_id": "chain_change_confirm",
                "after_group": "chain_identity",
                "after_hash": "4" * 64,
                "consumer_action_ids": [
                    "mode-action",
                    "chain-action",
                ],
            },
        )
        satisfied, details = predicate(_context(
            prior_chain_change,
            compound_with_mode_transition,
            staged_chain_and_mode,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertTrue(satisfied, details)

    def test_pending_evaluator_accepts_valid_receipt_and_rejects_rehashed_bad_semantics(
        self,
    ) -> None:
        valid = _pending_receipt()
        transition = {
            "transition": "consumed",
            "before_id": "NETWORK_INTERFACE",
            "before_group": "network",
            "before_hash": "1" * 64,
            "after_id": "",
            "after_group": "",
            "after_hash": "2" * 64,
            "consumer_action_ids": ["action-1"],
        }
        event = _event(
            valid,
            pending_transition=transition,
            turn_receipt={
                "admitted_action_ids": ["action-1"],
                "execution_order": ["action-1"],
            },
        )
        satisfied, details = POSTCONDITION_EVALUATORS[
            "current_menu_binding_preserved"
        ](_context(event))
        self.assertTrue(satisfied)
        self.assertEqual(details["invalid_receipts"], [])

        invalid = _pending_receipt(resolution_path="guessed_from_text")
        invalid_event = replace(event, control_receipts=(invalid,))
        satisfied, details = POSTCONDITION_EVALUATORS[
            "current_menu_binding_preserved"
        ](_context(invalid_event))
        self.assertFalse(satisfied)
        self.assertIn("semantics", details["invalid_receipts"][0]["reason"])

        wrong_action = _pending_receipt(resolved_action_id="action-2")
        wrong_action_event = replace(event, control_receipts=(wrong_action,))
        satisfied, details = POSTCONDITION_EVALUATORS[
            "current_menu_binding_preserved"
        ](_context(wrong_action_event))
        self.assertFalse(satisfied)
        self.assertEqual(details["matched_receipt_count"], 0)

    def test_mode_change_journey_is_verified_at_its_source_turns(self) -> None:
        turns = tuple(
            _turn_and_decision(message, turn_index=index)[0]
            for index, message in enumerate(
                ("2", "compare modes", "use fake-node", "2"),
                start=1,
            )
        )
        planner = _hashed({
            "receipt_type": "semantic_planner",
            "turn_index": 3,
            "input_hash": "1" * 64,
            "pending_contract_hash": "2" * 64,
            "resolver_invoked": True,
            "result_reason_hash": "3" * 64,
            "planned_action_types": ["choose_target_mode"],
            "semantic_units": [
                {"unit_id": "unit-1", "disposition": "action"},
            ],
            "planner_metrics": {"unit_count": 1},
        })
        routed = _event(
            planner,
            turn_index=3,
            pending_transition={
                "transition": "replaced",
                "before_id": "chain",
                "before_group": "chain_identity",
                "before_hash": "2" * 64,
                "after_id": "target_mode_change_confirm",
                "after_group": "target_mode",
                "after_hash": "4" * 64,
                "consumer_action_ids": ["action-3"],
            },
            turn_receipt={
                "owner_bindings": {"action-3": "chain_rpc"},
                "admitted_action_ids": ["action-3"],
                "execution_order": ["action-3"],
            },
            admitted_actions=({
                "type": "choose_target_mode",
                "action_id": "action-3",
                "owner": "chain_rpc",
                "effect": "configuration_mutation",
                "group": "target_mode",
                "argument_value_hashes": {
                    "target_mode": hashlib.sha256(
                        json.dumps(
                            "fake-node",
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            },),
        )
        resolution = _pending_receipt(
            turn_index=4,
            pending_id="target_mode_change_confirm",
            pending_group="target_mode",
            pending_contract_hash="4" * 64,
            selected_option_id="no",
            resolved_action_id="action-4",
        )
        resumed = _event(
            resolution,
            turn_index=4,
            pending_transition={
                "transition": "replaced",
                "before_id": "target_mode_change_confirm",
                "before_group": "target_mode",
                "before_hash": "4" * 64,
                "after_id": "chain",
                "after_group": "chain_identity",
                "after_hash": "5" * 64,
                "consumer_action_ids": ["action-4"],
            },
            turn_receipt={
                "execution_order": ["action-4"],
                "admitted_action_ids": ["action-4"],
            },
        )
        context = _context(routed, resumed, turns=turns)

        satisfied, details = POSTCONDITION_EVALUATORS[
            "mode_change_request_routed_from_chain_pending"
        ](context)
        self.assertTrue(satisfied, details)
        resumed_ok, details = POSTCONDITION_EVALUATORS[
            "declined_mode_change_resumes_chain_pending"
        ](context)
        self.assertTrue(resumed_ok, details)
        misrouted, details = POSTCONDITION_EVALUATORS[
            "mode_request_consumed_as_chain_identity"
        ](context)
        self.assertFalse(misrouted, details)
        selected, details = POSTCONDITION_EVALUATORS[
            "real_node_selection_executed_at_source"
        ](context)
        self.assertFalse(selected, details)
        consulted, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](context)
        self.assertFalse(consulted, details)

        wrong_planner = _hashed({
            **{
                key: value
                for key, value in planner.items()
                if key != "receipt_id"
            },
            "planned_action_types": ["answer_pending"],
        })
        wrong = _event(
            wrong_planner,
            turn_index=3,
            pending_transition={
                "transition": "replaced",
                "before_id": "chain",
                "before_group": "chain_identity",
                "before_hash": "2" * 64,
                "after_id": "unknown_chain_identity_confirm",
                "after_group": "chain_identity",
                "after_hash": "6" * 64,
                "consumer_action_ids": ["action-3"],
            },
            turn_receipt={
                "owner_bindings": {"action-3": "chain_rpc"},
            },
            material_diffs={
                "chain_identity.raw": {
                    "before": "",
                    "after": "7" * 64,
                },
            },
        )
        wrong_context = _context(wrong, resumed, turns=turns)
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "mode_change_request_routed_from_chain_pending"
        ](wrong_context)
        self.assertFalse(satisfied)
        misrouted, details = POSTCONDITION_EVALUATORS[
            "mode_request_consumed_as_chain_identity"
        ](wrong_context)
        self.assertTrue(misrouted, details)

    def test_rr003_source_selection_and_consultation_require_exact_product_lineage(
        self,
    ) -> None:
        messages = ("2", "compare modes", "use fake-node", "2")
        roles = (
            "select_real_node",
            "mode_comparison_consultation",
            "request_fake_node_change",
            "decline_mode_change",
        )
        turns = tuple(
            _turn_and_decision(message, turn_index=index)[0]
            for index, message in enumerate(messages, start=1)
        )
        contract = _verifier_input(
            "exact",
            messages,
            semantic_roles=roles,
        )
        selection_action = {
            "type": "choose_target_mode",
            "action_id": "action-1",
            "owner": "chain_rpc",
            "effect": "configuration_mutation",
            "group": "target_mode",
            "argument_value_hashes": {
                "target_mode": _value_hash("real-node"),
            },
        }
        pending_action = {
            "type": "answer_pending",
            "action_id": "action-0",
            "owner": "coordinator",
            "effect": "configuration_mutation",
            "group": "",
            "argument_value_hashes": {},
        }
        selection_receipt = _pending_receipt(
            turn_index=1,
            pending_id="opening_next_action",
            pending_group="opening",
            pending_contract_hash="1" * 64,
            selected_option_id="2",
            selected_value_hash=_value_hash("real-node"),
            resolved_action_id="action-0",
        )
        selected = _event(
            selection_receipt,
            turn_index=1,
            pending_transition={
                "transition": "replaced",
                "before_id": "opening_next_action",
                "before_group": "opening",
                "before_hash": "1" * 64,
                "after_id": "chain",
                "after_group": "chain_identity",
                "after_hash": "2" * 64,
                "consumer_action_ids": ["action-0", "action-1"],
            },
            turn_receipt={
                "admitted_action_ids": ["action-0", "action-1"],
                "execution_order": ["action-0", "action-1"],
            },
            admitted_actions=(pending_action, selection_action),
            material_diffs={
                "target_mode": {
                    "before": "",
                    "after": _value_hash("real-node"),
                },
            },
        )
        consultation_body = {
            "receipt_type": "orientation_response",
            "schema_version": 1,
            "owner": "orientation",
            "turn_index": 2,
            "topic": MODE_COMPARISON_TOPIC,
            "action_type": "answer_opening_question",
            "action_id": "action-2",
            "pending_contract_hash": "2" * 64,
            "response_hash": "3" * 64,
            "projection_fields": ["pending_id", "target_mode"],
            "state_projection_hash": "4" * 64,
            "read_only": True,
        }
        consultation_action = {
            "type": "answer_opening_question",
            "action_id": "action-2",
            "owner": "orientation",
            "effect": "read_only",
            "group": "",
            "argument_value_hashes": {
                "topic": _value_hash(MODE_COMPARISON_TOPIC),
            },
        }
        consulted = _event(
            _hashed(consultation_body),
            turn_index=2,
            pending_transition={
                "transition": "preserved",
                "before_id": "chain",
                "before_group": "chain_identity",
                "before_hash": "2" * 64,
                "after_id": "chain",
                "after_group": "chain_identity",
                "after_hash": "2" * 64,
                "consumer_action_ids": ["action-2"],
            },
            turn_receipt={
                "admitted_action_ids": ["action-2"],
                "execution_order": ["action-2"],
            },
            admitted_actions=(consultation_action,),
        )
        context = _context(
            selected,
            consulted,
            turns=turns,
            verifier_input_contract=contract,
        )

        selection_ok, details = POSTCONDITION_EVALUATORS[
            "real_node_selection_executed_at_source"
        ](context)
        self.assertTrue(selection_ok, details)
        consultation_ok, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](context)
        self.assertTrue(consultation_ok, details)

        unexecuted_selection = replace(
            selected,
            turn_receipt_summary={
                "admitted_action_ids": ["action-0", "action-1"],
                "execution_order": [],
            },
        )
        selection_ok, details = POSTCONDITION_EVALUATORS[
            "real_node_selection_executed_at_source"
        ](_context(
            unexecuted_selection,
            consulted,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(selection_ok, details)

        wrong_mode_action = {
            **selection_action,
            "argument_value_hashes": {
                "target_mode": _value_hash("fake-node"),
            },
        }
        selection_ok, details = POSTCONDITION_EVALUATORS[
            "real_node_selection_executed_at_source"
        ](_context(
            replace(
                selected,
                admitted_action_provenance=(pending_action, wrong_mode_action),
            ),
            consulted,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(selection_ok, details)

        wrong_committed_mode = replace(
            selected,
            material_state_diff_hashes={
                "target_mode": {
                    "before": "",
                    "after": _value_hash("fake-node"),
                },
            },
        )
        selection_ok, details = POSTCONDITION_EVALUATORS[
            "real_node_selection_executed_at_source"
        ](_context(
            wrong_committed_mode,
            consulted,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(selection_ok, details)

        wrong_receipt_action = _pending_receipt(
            turn_index=1,
            pending_id="opening_next_action",
            pending_group="opening",
            pending_contract_hash="1" * 64,
            selected_option_id="2",
            selected_value_hash=_value_hash("real-node"),
            resolved_action_id="action-1",
        )
        invalid_selection_events = {
            "preserved_transition": replace(
                selected,
                pending_transition={
                    **selected.pending_transition,
                    "transition": "preserved",
                },
            ),
            "wrong_pending_effect": replace(
                selected,
                admitted_action_provenance=({
                    **pending_action,
                    "effect": "read_only",
                }, selection_action),
            ),
            "reversed_execution_order": replace(
                selected,
                turn_receipt_summary={
                    "admitted_action_ids": ["action-0", "action-1"],
                    "execution_order": ["action-1", "action-0"],
                },
            ),
            "missing_transition_consumer": replace(
                selected,
                pending_transition={
                    **selected.pending_transition,
                    "consumer_action_ids": ["action-0"],
                },
            ),
            "receipt_bound_to_followup": replace(
                selected,
                control_receipts=(wrong_receipt_action,),
            ),
        }
        for name, invalid_event in invalid_selection_events.items():
            with self.subTest(name=name):
                selection_ok, details = POSTCONDITION_EVALUATORS[
                    "real_node_selection_executed_at_source"
                ](_context(
                    invalid_event,
                    consulted,
                    turns=turns,
                    verifier_input_contract=contract,
                ))
                self.assertFalse(selection_ok, details)

        mutated_consultation = replace(
            consulted,
            material_state_diff_hashes={
                "target_mode": {
                    "before": _value_hash("real-node"),
                    "after": _value_hash("fake-node"),
                },
            },
        )
        consultation_ok, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](_context(
            selected,
            mutated_consultation,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(consultation_ok, details)

        changed_pending = replace(
            consulted,
            pending_transition={
                **consulted.pending_transition,
                "transition": "replaced",
                "after_hash": "5" * 64,
            },
        )
        consultation_ok, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](_context(
            selected,
            changed_pending,
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(consultation_ok, details)

        wrong_topic_action = {
            **consultation_action,
            "argument_value_hashes": {
                "topic": _value_hash("capabilities"),
            },
        }
        consultation_ok, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](_context(
            selected,
            replace(
                consulted,
                admitted_action_provenance=(wrong_topic_action,),
            ),
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(consultation_ok, details)

        wrong_topic_receipt = _hashed({
            **consultation_body,
            "topic": "capabilities",
        })
        consultation_ok, details = POSTCONDITION_EVALUATORS[
            "mode_consultation_preserves_chain_pending"
        ](_context(
            selected,
            replace(
                consulted,
                control_receipts=(wrong_topic_receipt,),
            ),
            turns=turns,
            verifier_input_contract=contract,
        ))
        self.assertFalse(consultation_ok, details)

    def test_mode_change_open_variant_uses_attested_source_turns_after_clarification(
        self,
    ) -> None:
        contract = _verifier_input(
            "isomorphic",
            ("select real", "compare", "request fake", "decline"),
            semantic_roles=(
                "select_real_node",
                "mode_comparison_consultation",
                "request_fake_node_change",
                "decline_mode_change",
            ),
        )
        pairs = [
            _turn_and_decision(message, turn_index=index)
            for index, message in enumerate(
                (
                    "select real",
                    "compare",
                    "please clarify",
                    "request fake",
                    "decline",
                ),
                start=1,
            )
        ]
        for source_step_id, semantic_role, actual_index in (
            ("source-3", "request_fake_node_change", 4),
            ("source-4", "decline_mode_change", 5),
        ):
            turn, decision = pairs[actual_index - 1]
            attestation = build_variant_attestation(
                actor=decision.simulator_attestation["actor"],
                verifier_input_contract=contract,
                execution_binding=_execution_binding(decision),
                source_step_id=source_step_id,
                semantic_role=semantic_role,
                turn_index=actual_index,
                previous_response_hash=decision.previous_response_hash,
                user_message_hash=decision.user_message_hash,
                selected_at_ns=decision.selected_at_ns,
                declared_at_ns=decision.submitted_at_ns,
            )
            pairs[actual_index - 1] = (
                turn,
                _with_attestation(decision, attestation),
            )
        planner = _hashed({
            "receipt_type": "semantic_planner",
            "turn_index": 4,
            "input_hash": "1" * 64,
            "pending_contract_hash": "2" * 64,
            "resolver_invoked": True,
            "result_reason_hash": "3" * 64,
            "planned_action_types": ["choose_target_mode"],
            "semantic_units": [{"unit_id": "unit-1", "disposition": "action"}],
            "planner_metrics": {"unit_count": 1},
        })
        routed = _event(
            planner,
            turn_index=4,
            pending_transition={
                "transition": "replaced",
                "before_id": "chain",
                "before_group": "chain_identity",
                "before_hash": "2" * 64,
                "after_id": "target_mode_change_confirm",
                "after_group": "target_mode",
                "after_hash": "4" * 64,
                "consumer_action_ids": ["action-4"],
            },
            turn_receipt={
                "owner_bindings": {"action-4": "chain_rpc"},
                "admitted_action_ids": ["action-4"],
                "execution_order": ["action-4"],
            },
            admitted_actions=({
                "type": "choose_target_mode",
                "action_id": "action-4",
                "owner": "chain_rpc",
                "effect": "configuration_mutation",
                "group": "target_mode",
                "argument_value_hashes": {
                    "target_mode": hashlib.sha256(
                        json.dumps(
                            "fake-node",
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            },),
        )
        resolution = _pending_receipt(
            turn_index=5,
            pending_id="target_mode_change_confirm",
            pending_group="target_mode",
            pending_contract_hash="4" * 64,
            selected_option_id="no",
            resolved_action_id="action-5",
        )
        resumed = _event(
            resolution,
            turn_index=5,
            pending_transition={
                "transition": "replaced",
                "before_id": "target_mode_change_confirm",
                "before_group": "target_mode",
                "before_hash": "4" * 64,
                "after_id": "chain",
                "after_group": "chain_identity",
                "after_hash": "5" * 64,
                "consumer_action_ids": ["action-5"],
            },
            turn_receipt={
                "execution_order": ["action-5"],
                "admitted_action_ids": ["action-5"],
            },
        )
        context = _context(
            routed,
            resumed,
            turns=tuple(pair[0] for pair in pairs),
            decisions=tuple(pair[1] for pair in pairs),
            verifier_input_contract=contract,
        )

        routed_ok, routed_details = POSTCONDITION_EVALUATORS[
            "mode_change_request_routed_from_chain_pending"
        ](context)
        resumed_ok, resumed_details = POSTCONDITION_EVALUATORS[
            "declined_mode_change_resumes_chain_pending"
        ](context)

        self.assertTrue(routed_ok, routed_details)
        self.assertTrue(resumed_ok, resumed_details)

    def test_workload_predicate_requires_semantically_valid_replacement_receipt(
        self,
    ) -> None:
        satisfied, details = POSTCONDITION_EVALUATORS[
            "effective_workload_commit_replaces_defaults"
        ](_context(_event(_workload_receipt())))
        self.assertTrue(satisfied)
        self.assertEqual(details["matched_receipt_count"], 1)

        invalid = _workload_receipt(
            rpc_mode="mixed",
            mixed_weight_entries=[
                {"method_hash": evidence_hash("eth_accounts"), "weight": 40}
            ],
        )
        satisfied, details = POSTCONDITION_EVALUATORS[
            "effective_workload_commit_replaces_defaults"
        ](_context(_event(invalid)))
        self.assertFalse(satisfied)
        self.assertIn("weight", details["invalid_receipts"][0]["reason"])

    def test_sync_observe_rpc_load_violation_requires_same_event_mode_and_operation(
        self,
    ) -> None:
        sync_event = _event(
            execution={
                "intent_action_type": "final_benchmark",
                "manager_submission_receipt_id": "7" * 64,
                "manager_read_receipt_id": "8" * 64,
                "manager_observed_status": "completed",
                "job_status": "completed",
            },
        )
        sync_event = replace(
            sync_event,
            after_value_hashes={"workflow_mode": content_hash("sync_observe")},
        )
        observed, details = POSTCONDITION_EVALUATORS[
            "sync_observe_ran_rpc_load"
        ](_context(sync_event))
        self.assertTrue(observed)
        self.assertEqual(
            details["violations"][0]["operation"],
            "final_benchmark",
        )

        proper_sync = replace(
            sync_event,
            execution_receipt_summary={"intent_action_type": "sync_observe"},
        )
        observed, _ = POSTCONDITION_EVALUATORS[
            "sync_observe_ran_rpc_load"
        ](_context(proper_sync))
        self.assertFalse(observed)

        rpc_mode = replace(
            sync_event,
            after_value_hashes={"workflow_mode": content_hash("rpc_benchmark")},
        )
        observed, _ = POSTCONDITION_EVALUATORS[
            "sync_observe_ran_rpc_load"
        ](_context(rpc_mode))
        self.assertFalse(observed)

        planned_only = replace(
            sync_event,
            execution_receipt_summary={
                "intent_action_type": "final_benchmark",
            },
        )
        observed, _ = POSTCONDITION_EVALUATORS[
            "sync_observe_ran_rpc_load"
        ](_context(planned_only))
        self.assertFalse(observed)

    def test_semantic_unit_predicates_cover_order_preservation_and_drop(
        self,
    ) -> None:
        planner = _hashed({
            "receipt_type": "semantic_planner",
            "turn_index": 1,
            "input_hash": "1" * 64,
            "pending_contract_hash": "2" * 64,
            "resolver_invoked": True,
            "result_reason_hash": "3" * 64,
            "planned_action_types": ["set_value"],
            "semantic_units": [
                {"unit_id": "unit-1", "disposition": "action"},
                {"unit_id": "unit-2", "disposition": "unresolved"},
            ],
            "planner_metrics": {"unit_count": 2},
        })
        summary = {
            "semantic_units": [
                {"unit_id": "unit-1", "start": 0, "end": 4},
                {"unit_id": "unit-2", "start": 5, "end": 9},
            ],
            "semantic_order": ["unit-1", "unit-2"],
            "unit_action_bindings": {"unit-1": ["action-1"]},
            "unresolved_unit_ids": ["unit-2"],
        }
        context = _context(_event(planner, turn_receipt=summary))
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "semantic_units_partitioned_in_order"
        ](context)
        self.assertTrue(satisfied)
        preserved, _ = POSTCONDITION_EVALUATORS[
            "unresolved_units_preserved"
        ](context)
        self.assertTrue(preserved)
        dropped, details = POSTCONDITION_EVALUATORS[
            "semantic_unit_dropped"
        ](context)
        self.assertFalse(dropped, details)

        dropped_context = _context(
            _event(
                planner,
                turn_receipt={
                    **summary,
                    "unresolved_unit_ids": [],
                },
            )
        )
        dropped, details = POSTCONDITION_EVALUATORS[
            "semantic_unit_dropped"
        ](dropped_context)
        self.assertTrue(dropped)
        self.assertEqual(details["dropped_units_by_turn"], {1: ["unit-2"]})

    def test_execution_predicates_distinguish_single_submission_and_duplicate(
        self,
    ) -> None:
        first = _event(
            execution={
                "intent_idempotency_key": "execution-key",
                "manager_submission_receipt_id": "1" * 64,
                "manager_submission_disposition": "created",
                "manager_matching_job_count": 1,
                "job_id": "job-1",
            }
        )
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "approved_execution_submitted_once"
        ](_context(first))
        self.assertTrue(satisfied)
        duplicated = replace(
            first,
            turn_index=2,
            execution_receipt_summary={
                **first.execution_receipt_summary,
                "manager_submission_receipt_id": "2" * 64,
                "job_id": "job-2",
            },
        )
        duplicate, details = POSTCONDITION_EVALUATORS[
            "duplicate_job_submission"
        ](_context(first, duplicated))
        self.assertTrue(duplicate)
        self.assertEqual(
            details["duplicate_submission_counts"],
            {"execution-key": 2},
        )

    def test_read_only_consultation_positive_and_material_mutation_negative_side(
        self,
    ) -> None:
        body = {
            "receipt_type": "orientation_response",
            "schema_version": 1,
            "owner": "orientation",
            "turn_index": 1,
            "topic": "capabilities",
            "action_type": "answer_opening_question",
            "action_id": "action-1",
            "pending_contract_hash": "1" * 64,
            "response_hash": "2" * 64,
            "projection_fields": ["chain", "pending_id"],
            "state_projection_hash": "3" * 64,
            "read_only": True,
        }
        receipt = _hashed(body)
        transition = {
            "transition": "preserved",
            "before_id": "CLOUD_REGION",
            "before_group": "provider_deployment",
            "before_hash": "4" * 64,
            "after_id": "CLOUD_REGION",
            "after_group": "provider_deployment",
            "after_hash": "4" * 64,
            "consumer_action_ids": ["action-1"],
        }
        clean = _context(_event(receipt, pending_transition=transition))
        satisfied, _ = POSTCONDITION_EVALUATORS[
            "consultation_answered_read_only"
        ](clean)
        self.assertTrue(satisfied)
        preserved, _ = POSTCONDITION_EVALUATORS[
            "consultation_preserved_pending_work"
        ](clean)
        self.assertTrue(preserved)
        mutated, _ = POSTCONDITION_EVALUATORS[
            "workflow_state_mutated_by_consultation"
        ](clean)
        self.assertFalse(mutated)

        changed = _context(
            _event(
                receipt,
                pending_transition=transition,
                material_diffs={
                    "confirmed_config.CLOUD_REGION": {
                        "before": "5" * 64,
                        "after": "6" * 64,
                    }
                },
            )
        )
        mutated, details = POSTCONDITION_EVALUATORS[
            "workflow_state_mutated_by_consultation"
        ](changed)
        self.assertTrue(mutated, details)

    def test_every_registered_predicate_fails_closed_without_evidence(
        self,
    ) -> None:
        empty = _context()
        unexpectedly_satisfied = {
            postcondition_id: details
            for postcondition_id, evaluator in POSTCONDITION_EVALUATORS.items()
            for satisfied, details in (evaluator(empty),)
            if satisfied
        }
        self.assertEqual(unexpectedly_satisfied, {})

    def test_resume_contract_and_evidence_classification_use_structured_evidence(
        self,
    ) -> None:
        resume_contract = {
            "id": "resume_harness_session",
            "options": [
                {"id": "1", "expected_patch": {"resume_context": {}}},
                {"id": "2", "expected_patch": {"resume_context": {}}},
            ],
        }
        resume_event = replace(
            _event(),
            pending_question_id="resume_harness_session",
            pending_contract=resume_contract,
        )
        context = _context(
            resume_event,
            schedule=SimpleNamespace(start_scenario="resume-scenario"),
        )
        with patch(
            "tests.agent_live.runtime_checkpoint.reviewed_scenario",
            return_value=SimpleNamespace(question=resume_contract),
        ):
            exposed, _ = POSTCONDITION_EVALUATORS[
                "resume_action_contract_exposed"
            ](context)
        self.assertTrue(exposed)

        changed_contract = {
            **resume_contract,
            "options": [
                {
                    **resume_contract["options"][0],
                    "semantic_action": "reset_session",
                },
                resume_contract["options"][1],
            ],
        }
        changed_event = replace(
            resume_event,
            pending_contract=changed_contract,
        )
        with patch(
            "tests.agent_live.runtime_checkpoint.reviewed_scenario",
            return_value=SimpleNamespace(question=resume_contract),
        ):
            exposed, details = POSTCONDITION_EVALUATORS[
                "resume_action_contract_exposed"
            ](
                _context(
                    changed_event,
                    schedule=SimpleNamespace(start_scenario="resume-scenario"),
                )
            )
        self.assertFalse(exposed)
        self.assertEqual(details["matched_pending_contract_hashes"], [])

        orientation_body = {
            "receipt_type": "orientation_response",
            "schema_version": 1,
            "owner": "orientation",
            "turn_index": 1,
            "topic": "capabilities",
            "action_type": "answer_opening_question",
            "action_id": "action-1",
            "pending_contract_hash": "1" * 64,
            "response_hash": "2" * 64,
            "projection_fields": ["chain"],
            "state_projection_hash": "3" * 64,
            "read_only": True,
        }
        evidence_body = {
            "receipt_type": "analysis_evidence_block",
            "receipt_version": 2,
            "turn_index": 1,
            "owner": "analysis",
            "operation": "start",
            "block_id": "4" * 64,
            "question_id_hash": "5" * 64,
            "question_kind_hash": "6" * 64,
            "line_count": 1,
            "block_content_hash": "7" * 64,
            "input_disposition": "accepted",
            "input_non_empty_line_count": 1,
            "input_hash": "8" * 64,
            "status": "active",
            "response_message_ids": [],
            "response_render_hash": "",
            "response_rendered": False,
            "response_semantic_hash": "",
        }
        evidence_receipt = {
            **evidence_body,
            "receipt_id": analysis_hash(evidence_body),
        }
        counted, details = POSTCONDITION_EVALUATORS[
            "ordinary_question_counted_as_evidence"
        ](
            _context(
                _event(
                    _hashed(orientation_body),
                    evidence_receipt,
                )
            )
        )
        self.assertTrue(counted, details)


if __name__ == "__main__":
    unittest.main()
