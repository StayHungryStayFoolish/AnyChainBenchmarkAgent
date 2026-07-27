"""Test adapter that executes the same compiled LangGraph used by the product."""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from copy import deepcopy
import json
from typing import Any, Mapping
from unittest.mock import patch

from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state


TEST_SEMANTIC_PLANNER = None


def resolve_product_action_queue_for_test(
    state: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    """Drive all production planning stages without adding a product wrapper."""

    from agent.harness.hierarchical_planner import (
        begin_semantic_partition,
        compile_next_owner,
        review_semantic_plan,
    )

    document = begin_semantic_partition(dict(state), text)
    while document.get("status") == "compile_owner":
        document = compile_next_owner(dict(state), document)
    return review_semantic_plan(dict(state), document)


def action_types(state: Mapping[str, Any], key: str = "action_queue") -> list[str]:
    """Project typed action envelopes without exposing their storage shape."""

    return [
        str(item.get("action_type") or item.get("type") or "")
        for item in state.get(key) or []
        if isinstance(item, Mapping)
    ]


def invoke_actions(
    state: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
    text: str | None = None,
) -> dict[str, Any]:
    """Admit proposed actions through the product graph."""

    current = deepcopy(dict(state))
    if text is not None:
        current["last_user_input"] = text
    with patch(
        "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
        side_effect=lambda planner_state, planner_text: reviewed_action_plan(
            planner_state,
            planner_text,
            actions,
        ),
    ):
        return invoke_product_graph_turn(current)


def invoke_action(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    text: str | None = None,
) -> dict[str, Any]:
    """Admit one proposed action through the product graph."""

    return invoke_actions(state, [action], text)


def answer_pending(
    state: Mapping[str, Any],
    answer: str,
    _question: Mapping[str, Any] | None = None,
    *,
    selected_value: Any = None,
    manual_value: Any = None,
) -> dict[str, Any]:
    """Answer the active question through deterministic graph admission."""

    current = deepcopy(dict(state))
    resolved = manual_value if manual_value is not None else selected_value
    current["last_user_input"] = str(resolved if resolved is not None else answer)
    pending = dict(current.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, Mapping)]
    exact_values = {
        str(item.get("value"))
        for item in options
        if item.get("value") is not None
    }
    if resolved is None and str(answer) in exact_values:
        return invoke_product_graph_turn(current)
    value = resolved if resolved is not None else answer
    return invoke_actions(
        current,
        [{
            "type": "answer_pending",
            "answer": value,
            "selected_value": value,
            "source_evidence": current["last_user_input"],
            "confidence": "high",
        }],
        current["last_user_input"],
    )


@contextmanager
def reviewed_execution_planner(
    *,
    expected_input: str,
    expected_admitted: bool,
    reviewed_value: Any = None,
):
    """Install one input-bound reviewed planner for deterministic evidence.

    This fixture does not infer or repair actions. It can only approve the
    exact input declared by a reviewed execution case, or record that the case
    is expected to produce no admitted action.
    """

    def resolve(state: Mapping[str, Any], text: str) -> dict[str, Any]:
        if text != expected_input.strip():
            raise AssertionError(
                "reviewed execution planner received a different input"
            )
        actions: list[dict[str, Any]] = []
        if expected_admitted:
            value = expected_input if reviewed_value is None else reviewed_value
            actions.append({
                "type": "answer_pending",
                "answer": value,
                "selected_value": value,
                "source_evidence": text,
                "confidence": "high",
            })
        return reviewed_semantic_plan(state, text, actions)

    with ExitStack() as stack:
        stack.enter_context(patch(
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
            side_effect=resolve,
        ))
        stack.enter_context(reviewed_stage_planner(resolve))
        stack.enter_context(patch(
            "agent.harness.advisory.provider_from_config",
            side_effect=RuntimeError(
                "model-backed advisory is outside deterministic evidence"
            ),
        ))
        yield


@contextmanager
def reviewed_stage_planner(resolve):
    """Adapt one reviewed test resolver to the checkpointed graph stages."""

    def partition(
        planner_state: Mapping[str, Any],
        planner_text: str,
    ) -> dict[str, Any]:
        return {
            "contract_version": 1,
            "status": "review_plan",
            "clauses": [
                dict(item)
                for item in (planner_state.get("turn_receipt") or {}).get(
                    "clauses"
                ) or []
                if isinstance(item, Mapping)
            ],
            "reviewed_queue": resolve(planner_state, planner_text),
        }

    with ExitStack() as stack:
        stack.enter_context(patch(
            "agent.harness.coordinator.compile_bounded_semantic_value",
            return_value=None,
        ))
        stack.enter_context(patch(
            "agent.harness.hierarchical_planner.begin_semantic_partition",
            side_effect=partition,
        ))
        stack.enter_context(patch(
            "agent.harness.hierarchical_planner.review_semantic_plan",
            side_effect=lambda _state, document: dict(
                document.get("reviewed_queue") or {}
            ),
        ))
        yield


def reviewed_semantic_plan(
    state: Mapping[str, Any],
    text: str,
    proposed_actions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit deterministic evidence through the product semantic boundary."""

    from agent.harness.semantic_admission import (
        ALLOWED_ACTION_TYPES,
        _admitted_action_queue,
        _freeze_bounded_semantic_plan,
        prepare_hierarchical_candidate,
    )
    from agent.harness.plan_coverage import segment_user_turn
    from agent.harness.semantic_compiler import validate_whole_plan_admission

    actions = [deepcopy(dict(item)) for item in proposed_actions]
    clauses = segment_user_turn(text)
    semantic_units = [
        {
            "unit_id": f"reviewed-clause-{index + 1}",
            "clause_id": clause.clause_id,
            "start": 0,
            "end": len(clause.text),
            "source_text": clause.text,
            "disposition": "action" if actions else "unresolved",
            "action_indexes": list(range(len(actions))),
            "reason": "explicit reviewed execution evidence",
        }
        for index, clause in enumerate(clauses)
    ]
    candidate, validation = prepare_hierarchical_candidate(
        json.dumps(
            {
                "actions": actions,
                "semantic_units": semantic_units,
                "reason": "explicit reviewed execution evidence",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        dict(state),
        clauses,
        pending_choice_unit_ids=frozenset(
            str(unit["unit_id"]) for unit in semantic_units
        ),
    )
    if not validation.valid:
        return {
            "actions": [],
            "semantic_units": semantic_units,
            "pending_choice_contracts": [],
            "admission_rejections": list(validation.errors),
            "reason": "reviewed evidence failed product candidate validation",
        }
    plan = _freeze_bounded_semantic_plan(
        candidate,
        dict(state),
        clauses,
    )
    request = plan.request_payload()
    units = {
        str(row["unit_id"]): row
        for row in request["semantic_units"]
    }
    action_verdicts = []
    for action in request["actions"]:
        pending_candidates = list(action.get("pending_value_candidates") or [])
        selected_identity = (
            str(pending_candidates[0].get("identity") or "")
            if len(pending_candidates) == 1
            else ""
        )
        pending_argument = next(
            (
                str(candidate_row["candidate_id"])
                for candidate_row in pending_candidates
                if str(candidate_row.get("identity") or "") == selected_identity
            ),
            "",
        )
        action_verdicts.append({
            "action_id": action["action_id"],
            "verdict": "admit",
            "unit_ids": list(action["unit_ids"]),
            "evidence": [
                {
                    "unit_id": unit_id,
                    "quote": str(units[unit_id]["source_text"]),
                    "relation": "direct",
                    "support_relation": "",
                }
                for unit_id in action["unit_ids"]
            ],
            "grounded_arguments": [
                {
                    "argument_name": argument,
                    "evidence_quote": str(units[action["unit_ids"][0]]["source_text"]),
                }
                for argument in action.get("required_value_grounding_arguments") or []
            ],
            "pending_answer_argument": pending_argument,
            "turn_candidate_verdicts": [
                {
                    "candidate_id": str(candidate_row["candidate_id"]),
                    "verdict": (
                        "selected"
                        if str(candidate_row.get("identity") or "") == selected_identity
                        else "not_selected"
                    ),
                    "evidence_quote": str(
                        units[str(candidate_row["source_unit_ids"][0])]["source_text"]
                    ),
                    "reason": "the reviewed fixture preserves the candidate source role",
                }
                for candidate_row in action.get("turn_pending_value_candidates") or []
            ],
            "reason": "the explicit reviewed action preserves its source units",
        })
    admission_payload = {
        "plan_hash": request["plan_hash"],
        "action_verdicts": action_verdicts,
        "unit_verdicts": [
            {
                "unit_id": unit["unit_id"],
                "verdict": "complete",
                "owner_action_ids": list(unit["owner_action_ids"]),
                "evidence_quote": str(unit["source_text"]),
                "omitted_action_type": "",
                "reason": "the explicit reviewed unit has its declared owner",
            }
            for unit in request["semantic_units"]
        ],
        "reason": "the explicit deterministic evidence plan is admitted",
    }
    admission = validate_whole_plan_admission(
        json.dumps(admission_payload, ensure_ascii=False, sort_keys=True),
        plan,
        allowed_action_types=ALLOWED_ACTION_TYPES,
    )
    if not admission.valid:
        return {
            "actions": [],
            "semantic_units": semantic_units,
            "pending_choice_contracts": [],
            "admission_rejections": list(admission.errors),
            "reason": "reviewed evidence failed whole-plan admission",
        }
    return _admitted_action_queue(plan, admission, dict(state))


def reviewed_action_plan(
    state: Mapping[str, Any],
    text: str,
    proposed_actions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete planner-boundary fixture with immutable receipts."""

    from agent.harness.action_registry import (
        ACTION_BY_TYPE,
        build_admission_transaction_hash,
        build_proposal_field_receipt,
    )

    actions = [deepcopy(dict(item)) for item in proposed_actions]
    turn_clauses = [
        dict(item)
        for item in (state.get("turn_receipt") or {}).get("clauses") or []
        if isinstance(item, Mapping)
        and str(item.get("clause_id") or "")
    ]

    def semantic_unit(index: int) -> dict[str, Any]:
        clause = (
            turn_clauses[min(index, len(turn_clauses) - 1)]
            if turn_clauses
            else {
                "clause_id": f"test-clause-{index}",
                "text": str(text),
            }
        )
        clause_id = str(clause.get("clause_id") or "")
        source_text = str(clause.get("text") or text)
        return {
            "unit_id": f"reviewed-{clause_id}-{index}",
            "clause_id": clause_id,
            "start": 0,
            "end": len(source_text),
            "source_text": source_text,
            "disposition": "action",
            "action_indexes": [index],
            "reason": "reviewed test planner output",
        }

    if actions and all(
        str(item.get("_plan_transaction_hash") or "")
        for item in actions
    ):
        return {
            "actions": actions,
            "semantic_units": [semantic_unit(index) for index in range(len(actions))],
            "pending_choice_contracts": [],
            "reason": "pre-reviewed deterministic planner fixture",
        }
    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, Mapping)]
    selected_options: dict[int, Mapping[str, Any]] = {}
    for index, action in enumerate(actions):
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if (
            (spec is not None and "source_evidence" in spec.required_arguments)
            or str(action.get("type") or "") == "start_custom_rpc"
        ):
            action.setdefault("source_evidence", str(text))
        option: Mapping[str, Any] = {}
        if str(action.get("type") or "") == "answer_pending":
            selected = action.get("selected_value")
            option = next(
                (item for item in options if item.get("value") == selected),
                {},
            )
        elif spec is not None and spec.pending_option_admission:
            option = next(
                (
                    item
                    for item in options
                    if isinstance(item.get("action"), Mapping)
                    and str(item["action"].get("type") or "")
                    == str(action.get("type") or "")
                    and all(
                        key == "type" or action.get(key) == value
                        for key, value in item["action"].items()
                    )
                ),
                {},
            )
        if option:
            selected = option.get("value")
            actions[index] = {
                "type": "answer_pending",
                "answer": selected,
                "selected_value": selected,
                "source_evidence": str(action.get("source_evidence") or text),
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": str(action.get("confidence") or "medium"),
            }
            selected_options[index] = option

    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    turn_index = int(state.get("turn_index") or 0)
    action_ids = [f"test-admission-{index}" for index in range(len(actions))]
    semantic_units = [semantic_unit(index) for index in range(len(actions))]
    transaction_hash = build_admission_transaction_hash(
        thread_id=thread_id,
        session_id=session_id,
        submitted_turn_index=turn_index,
        actions=actions,
        semantic_units=semantic_units,
        admission_action_ids=action_ids,
    )
    for index, action in enumerate(actions):
        action["_admission_action_id"] = action_ids[index]
        action["_transaction_action_ids"] = list(action_ids)
        action["_plan_transaction_hash"] = transaction_hash
        if str(action.get("type") or "") != "propose_config_values":
            continue
        receipts = {}
        for field, value in dict(action.get("config_values") or {}).items():
            receipts[field] = build_proposal_field_receipt(
                thread_id=thread_id,
                session_id=session_id,
                submitted_turn_index=turn_index,
                transaction_hash=transaction_hash,
                admission_action_id=action_ids[index],
                config_field=field,
                canonical_value=value,
                source_unit_id=semantic_units[index]["unit_id"],
                source_unit_text=str(text),
                source_quote=str(text),
            )
        action["_proposal_transaction_hashes"] = [transaction_hash]
        action["_proposal_field_receipts"] = receipts
    return {
        "actions": actions,
        "semantic_units": semantic_units,
        "pending_choice_contracts": [
            {
                "action_index": index,
                "admission_action_id": action_ids[index],
                "question": {
                    "id": str(pending.get("id") or ""),
                    "group": str(pending.get("group") or ""),
                    "contract_version": pending.get("contract_version"),
                },
                "option": {
                    "id": str(option.get("id") or ""),
                    "selected_value": option.get("value"),
                },
                "semantic_units": [semantic_units[index]],
            }
            for index, option in selected_options.items()
        ],
        "reason": "reviewed deterministic planner fixture",
    }


def admitted_action_queue(
    state: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build current durable envelopes for tests that seed deferred work."""

    from agent.harness.action_registry import ACTION_BY_TYPE
    from agent.harness.contracts import ActionEnvelope, action_envelope_to_dict
    from agent.harness.domains.registry import GROUP_OWNER

    pending = dict(state.get("pending_question") or {})
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        action = dict(raw)
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE[action_type]
        owner = (
            GROUP_OWNER.get(str(pending.get("group") or ""), spec.owner)
            if action_type == "answer_pending"
            else spec.owner
        )
        arguments = {
            name: action[name]
            for name in spec.arguments
            if name in action
        }
        envelope = ActionEnvelope(
            action_id=str(action.get("action_id") or f"test-action-{index}"),
            action_type=action_type,
            owner=owner,
            target_group=spec.target_group,
            arguments=arguments,
            confidence=str(action.get("confidence") or "high"),
            reason=str(action.get("reason") or ""),
            semantic_order=index,
            execution_order=index,
            submitted_turn_index=int(state.get("turn_index") or 0),
            origin_group=str(state.get("active_group") or ""),
            origin_text=str(
                action.get("_origin_text")
                or action.get("source_evidence")
                or ""
            ),
            plan_scope="test-fixture",
            effect_kind=(
                "external"
                if spec.effect == "execution"
                else "read_only"
                if spec.effect == "read_only"
                else "pure"
            ),
            idempotency_key=f"test-fixture:{index}:{action_type}",
        )
        output.append(action_envelope_to_dict(envelope))
    return output


def invoke_product_graph_turn(
    state: Mapping[str, Any],
    *,
    allow_semantic_resolver: bool = False,
) -> dict[str, Any]:
    """Run one deterministic uncheckpointed turn through the product graph.

    Contract and unit evidence must never depend on a live model response. A
    caller exercising semantic input has to inject the reviewed resolver it is
    testing; real provider behavior belongs to the PTY/CLI acceptance lanes.
    """

    from agent.harness.domains.rpc_catalog import migrate_legacy_catalog
    from agent.harness.state import (
        STATE_SCHEMA_VERSION,
        migrate_state,
    )
    from agent.harness import advisory
    from agent.harness.domains import analysis, chain_identity, recovery, rpc_endpoint

    current = deepcopy(dict(state))
    _materialize_handwritten_question_fixture(current)
    migrate_legacy_catalog(current)
    if int(current.get("schema_version") or 0) < STATE_SCHEMA_VERSION:
        current = migrate_state(
            current,
            thread_id=str(current.get("thread_id") or "test-fixture"),
            language=str(current.get("language") or "en"),
            session_purpose=str((current.get("session") or {}).get("purpose") or "test"),
        )
    guarded_entries = (
        (
            "agent.harness.domains.chain_identity.resolve_unknown_chain_identity",
            chain_identity.resolve_unknown_chain_identity,
            advisory.resolve_unknown_chain_identity,
        ),
        (
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            rpc_endpoint.extract_rpc_schema_from_evidence,
            advisory.extract_rpc_schema_from_evidence,
        ),
        (
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            analysis.analyze_evidence_with_model,
            advisory.analyze_evidence_with_model,
        ),
        (
            "agent.harness.domains.recovery.analyze_evidence_with_model",
            recovery.analyze_evidence_with_model,
            advisory.analyze_evidence_with_model,
        ),
    )
    with ExitStack() as stack:
        active_resolver = TEST_SEMANTIC_PLANNER
        if callable(active_resolver):
            def reviewed_partition(
                planner_state: Mapping[str, Any],
                planner_text: str,
            ) -> dict[str, Any]:
                return {
                    "contract_version": 1,
                    "status": "review_plan",
                    "clauses": [
                        dict(item)
                        for item in (
                            planner_state.get("turn_receipt") or {}
                        ).get("clauses") or []
                        if isinstance(item, Mapping)
                    ],
                    "reviewed_queue": active_resolver(
                        planner_state,
                        planner_text,
                    ),
                }

            stack.enter_context(patch(
                "agent.harness.coordinator.compile_bounded_semantic_value",
                return_value=None,
            ))
            stack.enter_context(patch(
                "agent.harness.hierarchical_planner.begin_semantic_partition",
                side_effect=reviewed_partition,
            ))
            stack.enter_context(patch(
                "agent.harness.hierarchical_planner.review_semantic_plan",
                side_effect=lambda _state, document: dict(
                    document.get("reviewed_queue") or {}
                ),
            ))
        elif not allow_semantic_resolver:
            stack.enter_context(patch(
                "agent.harness.coordinator.compile_bounded_semantic_value",
                side_effect=AssertionError(
                    "deterministic graph turn attempted to call a live model entry"
                ),
            ))
            stack.enter_context(patch(
                "agent.harness.hierarchical_planner.begin_semantic_partition",
                side_effect=AssertionError(
                    "deterministic graph turn attempted to call a live model entry"
                ),
            ))
        for target, active, original in guarded_entries:
            if active is original:
                stack.enter_context(patch(
                    target,
                    side_effect=AssertionError(
                        "deterministic graph turn attempted to call a live model entry"
                    ),
                ))
        invocation_context = {
            key: deepcopy(current.get(key) or {})
            for key in ("discovery", "framework_summary", "web_research")
        }
        result = dict(
            build_graph(None).invoke(
                current,
                context=invocation_context,
            )
        )
    validate_state(result)
    return result


def _materialize_handwritten_question_fixture(state: dict[str, Any]) -> None:
    """Upgrade old hand-written test questions to the current typed contract.

    Product checkpoints use the versioned state migrator. This adapter exists
    only because the historical monolithic test module constructs hundreds of
    current-schema dictionaries directly instead of using question builders.
    """

    from agent.workflows.group_registry import GROUP_OWNER

    questions = []
    pending = state.get("pending_question")
    if isinstance(pending, dict) and pending:
        questions.append(pending)
    resume_pending = (state.get("resume_context") or {}).get("pending_question")
    if isinstance(resume_pending, dict) and resume_pending:
        questions.append(resume_pending)
    for question in questions:
        group = str(question.get("group") or "")
        question.setdefault("contract_version", 2)
        if group in GROUP_OWNER:
            question.setdefault("owner", GROUP_OWNER[group])
        if question.get("owner") != "chain_rpc":
            continue
        question_id = str(question.get("id") or "")
        if question_id.startswith("custom_rpc_"):
            question.setdefault("domain_context", {"rpc_case": "custom_rpc"})
        elif question_id.startswith("new_chain_"):
            question.setdefault("domain_context", {"rpc_case": "new_chain"})
