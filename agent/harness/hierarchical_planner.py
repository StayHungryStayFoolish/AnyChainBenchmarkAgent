"""Hierarchical semantic planning for current product turns.

Stage A partitions the complete turn and routes exact source units to domain
owners. Stage B compiles only the actions owned by each selected owner. The
result continues through the immutable whole-plan admission boundary until
the graph owns the complete turn receipt in Phase 3.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..llm.providers import provider_from_config
from ..llm.types import (
    LLMProviderError,
    LLMTurnTimeoutError,
)
from .action_registry import (
    ACTION_ARGUMENT_SCHEMAS,
    ACTION_BY_TYPE,
    ACTION_SPECS,
    SEMANTIC_OPERATION_PURPOSES,
    SEMANTIC_OPERATIONS,
    pending_barrier_semantics,
    registered_semantic_value_domains,
    semantic_grounding_arguments,
    semantic_value_domain_conflicts,
    validate_action_contract,
)
from .context import (
    action_schema,
    group_schema,
    owner_workflow_snapshot,
    workflow_snapshot,
)
from .domains.environment import extract_structured_input_candidates
from .semantic_admission import (
    _admitted_action_queue,
    _review_bounded_semantic_candidate,
    _semantic_fulfillment_prompt,
    _unresolved_action_queue,
    prepare_hierarchical_candidate,
)
from .plan_coverage import TurnClause, segment_user_turn, validate_semantic_partition
from .questions import (
    pending_option_value_exists,
    pending_value_identity,
    typed_pending_value_candidates,
)
from .semantic_compiler import (
    STRICT_JSON_REASONING_MODE,
    closed_enum_quote_names_only_competing_values,
    request_semantic_compilation,
    whole_plan_admission_prompt,
)
from .state import AgentGraphState
from agent.workflows.group_registry import GROUP_SPEC_BY_NAME


_OWNERS = frozenset(spec.owner for spec in ACTION_BY_TYPE.values())
_UNIVERSAL_OPERATIONS = SEMANTIC_OPERATIONS
_UNIVERSAL_OPERATION_OWNER = {
    "pending_answer": "coordinator",
    "consultation": "orientation",
    "navigation": "coordinator",
    "administrative": "coordinator",
    "evidence_analysis": "analysis",
    "report_analysis": "analysis",
}
_PARTITION_KEYS = frozenset({
    "unit_id",
    "clause_id",
    "source_text",
    "operation",
    "owner_routes",
    "reason",
})
_ROUTE_KEYS = frozenset({"owner", "group"})
_BINDING_KEYS = frozenset({"unit_id", "action_indexes", "disposition", "reason"})


@dataclass(frozen=True)
class HierarchicalPlannerMetrics:
    stage_a_calls: int
    stage_b_calls: int
    admission_calls: int
    prompt_bytes: int
    largest_request_bytes: int
    elapsed_ms: int
    owner_count: int
    unit_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "stage_a_calls": self.stage_a_calls,
            "stage_b_calls": self.stage_b_calls,
            "admission_calls": self.admission_calls,
            "model_calls": (
                self.stage_a_calls + self.stage_b_calls + self.admission_calls
            ),
            "prompt_bytes": self.prompt_bytes,
            "largest_request_bytes": self.largest_request_bytes,
            "elapsed_ms": self.elapsed_ms,
            "owner_count": self.owner_count,
            "unit_count": self.unit_count,
        }


def begin_semantic_partition(
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    """Run Stage A and persist an immutable owner-compilation schedule."""

    started = time.monotonic()
    clauses = _turn_clauses(state, text)
    document: dict[str, Any] = {
        "contract_version": 1,
        "status": "partition",
        "started_monotonic": started,
        "clauses": [clause.as_dict() for clause in clauses],
        "source_partition": [],
        "routed_partition": [],
        "owner_requests": [],
        "owner_cursor": 0,
        "owner_documents": {},
        "request_sizes": [],
        "stage_a_calls": 0,
        "stage_b_calls": 0,
        "admission_calls": 0,
        "errors": [],
    }
    if not clauses:
        document["status"] = "failed"
        document["errors"] = ["empty semantic turn"]
        return document
    try:
        provider = provider_from_config()
        stage_a_payload = _stage_a_payload(state, text, clauses)
        partition: list[dict[str, Any]] = []
        partition_errors: tuple[str, ...] = ()
        stage_a_prompt = _stage_a_prompt()
        previous_output = ""
        for attempt in range(2):
            request_payload = dict(stage_a_payload)
            request_prompt = stage_a_prompt
            if attempt:
                request_payload["contract_repair"] = {
                    "prior_invalid_output": previous_output,
                    "validation_errors": list(partition_errors),
                    "instruction": (
                        "Return a complete replacement document. Correct every "
                        "validation error without dropping or paraphrasing source text."
                    ),
                }
                request_prompt = (
                    f"{stage_a_prompt} This is a contract-repair attempt. The prior "
                    f"document was rejected for: {'; '.join(partition_errors)}. "
                    "Return a new full document that satisfies those errors exactly."
                )
            document["request_sizes"].append(
                _wire_size(request_prompt, request_payload)
            )
            document["stage_a_calls"] += 1
            previous_output = request_semantic_compilation(
                provider,
                system_prompt=request_prompt,
                request_payload=request_payload,
                max_tokens=2600,
                reasoning_mode=STRICT_JSON_REASONING_MODE,
            )
            partition, partition_errors = _validate_partition_document(
                previous_output,
                clauses,
            )
            partition_errors = tuple(dict.fromkeys((
                *partition_errors,
                *_cross_domain_pending_errors(partition, state),
            )))
            if not partition_errors:
                break
        if partition_errors:
            document["status"] = "failed"
            document["errors"] = list(partition_errors)
            document["unit_count"] = len(partition)
            return document
        (
            admission_errors,
            stage_a_admission_sizes,
            redundant_unit_ids,
        ) = _review_stage_a_partition(provider, stage_a_payload, partition)
        document["request_sizes"].extend(stage_a_admission_sizes)
        document["admission_calls"] += len(stage_a_admission_sizes)
        if admission_errors:
            document["status"] = "failed"
            document["errors"] = list(admission_errors)
            document["unit_count"] = len(partition)
            return document
        source_partition, compilation_partition = (
            _partition_after_stage_a_admission(partition, redundant_unit_ids)
        )
        source_partition, compilation_partition = (
            _canonicalize_atomic_evidence_partition(
                state, clauses, source_partition, compilation_partition
            )
        )
        source_partition, compilation_partition = (
            _canonicalize_unique_manual_pending_partition(
                state, clauses, source_partition, compilation_partition
            )
        )
        routed_partition = _expand_partition_routes(compilation_partition)
        owner_requests = []
        for owner, unit_ids in _owner_requests(routed_partition).items():
            groups = sorted({
                str(route["group"])
                for unit in routed_partition
                if unit["unit_id"] in unit_ids
                for route in unit["owner_routes"]
                if route["owner"] == owner and route["group"]
            })
            owner_requests.append({
                "owner": owner,
                "unit_ids": list(unit_ids),
                "groups": groups,
            })
        document.update({
            "status": "compile_owner" if owner_requests else "review_plan",
            "source_partition": source_partition,
            "routed_partition": routed_partition,
            "owner_requests": owner_requests,
            "owner_count": len(owner_requests),
            "unit_count": len(routed_partition),
        })
        return document
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        document["status"] = "failed"
        document["errors"] = [
            f"hierarchical planner failed: {type(exc).__name__}"
        ]
        return document


def compile_next_owner(
    state: AgentGraphState,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile exactly one owner document and advance the persisted cursor."""

    output = dict(document)
    if output.get("status") != "compile_owner":
        return output
    requests = [
        dict(item)
        for item in output.get("owner_requests") or []
        if isinstance(item, Mapping)
    ]
    cursor = int(output.get("owner_cursor") or 0)
    if cursor >= len(requests):
        output["status"] = "review_plan"
        return output
    request = requests[cursor]
    owner = str(request.get("owner") or "")
    try:
        owner_document, errors, request_sizes = _compile_owner_document(
            state,
            owner,
            frozenset(str(item) for item in request.get("groups") or []),
            [
                dict(item)
                for item in output.get("routed_partition") or []
                if isinstance(item, Mapping)
            ],
            tuple(str(item) for item in request.get("unit_ids") or []),
        )
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        errors = (f"Stage B {owner} failed: {type(exc).__name__}",)
        owner_document = {}
        request_sizes = ()
    output["request_sizes"] = [
        *[int(value) for value in output.get("request_sizes") or []],
        *[int(value) for value in request_sizes],
    ]
    output["stage_b_calls"] = int(output.get("stage_b_calls") or 0) + len(
        request_sizes
    )
    if errors:
        output["status"] = "failed"
        output["errors"] = list(errors)
        return output
    owner_documents = {
        str(key): dict(value)
        for key, value in dict(output.get("owner_documents") or {}).items()
        if isinstance(value, Mapping)
    }
    owner_documents[owner] = owner_document
    output["owner_documents"] = owner_documents
    output["owner_cursor"] = cursor + 1
    output["status"] = (
        "compile_owner" if cursor + 1 < len(requests) else "review_plan"
    )
    return output


def review_semantic_plan(
    state: AgentGraphState,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Run whole-plan admission after every owner document is checkpointed."""

    output = dict(document)
    clauses = tuple(
        TurnClause(
            str(item.get("clause_id") or ""),
            str(item.get("text") or ""),
            str(item.get("input_shape") or "prose"),
        )
        for item in output.get("clauses") or []
        if isinstance(item, Mapping)
    )
    started = float(output.get("started_monotonic") or time.monotonic())
    request_sizes = [int(value) for value in output.get("request_sizes") or []]
    stage_a_calls = int(output.get("stage_a_calls") or 0)
    stage_b_calls = int(output.get("stage_b_calls") or 0)
    admission_calls = int(output.get("admission_calls") or 0)
    owner_count = int(output.get("owner_count") or 0)
    unit_count = int(output.get("unit_count") or 0)
    if output.get("status") == "failed":
        return _with_metrics(
            _unresolved_action_queue(clauses, tuple(output.get("errors") or [])),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    if output.get("status") != "review_plan":
        return _with_metrics(
            _unresolved_action_queue(
                clauses,
                ("semantic plan reached review before owner compilation completed",),
            ),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    source_partition = [
        dict(item)
        for item in output.get("source_partition") or []
        if isinstance(item, Mapping)
    ]
    routed_partition = [
        dict(item)
        for item in output.get("routed_partition") or []
        if isinstance(item, Mapping)
    ]
    owner_documents = {
        str(key): dict(value)
        for key, value in dict(output.get("owner_documents") or {}).items()
        if isinstance(value, Mapping)
    }
    candidate = _merge_owner_documents(
        source_partition,
        routed_partition,
        owner_documents,
    )
    candidate_text, validation = prepare_hierarchical_candidate(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True),
        state,
        clauses,
        pending_choice_unit_ids=frozenset(
            str(unit["unit_id"])
            for unit in source_partition
            if str(unit.get("operation") or "") == "pending_answer"
        ),
    )
    if not validation.valid:
        return _with_metrics(
            _unresolved_action_queue(clauses, validation.errors),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    provider = provider_from_config()
    plan, admission, admission_errors = _review_bounded_semantic_candidate(
        provider,
        candidate_text,
        validation,
        state,
        clauses,
        allowed_action_types=_allowed_action_types_for_partition(
            routed_partition
        ),
        whole_plan_contract_repair=True,
        reasoning_mode=STRICT_JSON_REASONING_MODE,
    )
    admission_calls += (
        int(getattr(admission, "request_count", 1))
        if admission is not None
        else 0
    )
    if admission is not None and getattr(admission, "request_sizes", ()):
        request_sizes.extend(admission.request_sizes)
    elif plan is not None:
        request_sizes.append(
            len(
                whole_plan_admission_prompt(
                    _semantic_fulfillment_prompt()
                ).encode("utf-8")
            )
            + len(plan.request_json.encode("utf-8"))
        )
    if (
        plan is not None
        and admission is not None
        and not admission.valid
    ):
        repaired_candidate = _conservative_closed_enum_intake_repair(
            candidate,
            plan,
            admission,
        )
        if repaired_candidate is not None:
            repaired_text, repaired_validation = prepare_hierarchical_candidate(
                json.dumps(
                    repaired_candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                state,
                clauses,
                pending_choice_unit_ids=frozenset(
                    str(unit["unit_id"])
                    for unit in source_partition
                    if str(unit.get("operation") or "") == "pending_answer"
                ),
            )
            if repaired_validation.valid:
                repaired_plan, repaired_admission, repaired_errors = (
                    _review_bounded_semantic_candidate(
                        provider,
                        repaired_text,
                        repaired_validation,
                        state,
                        clauses,
                        allowed_action_types=_allowed_action_types_for_partition(
                            routed_partition
                        ),
                        whole_plan_contract_repair=True,
                        reasoning_mode=STRICT_JSON_REASONING_MODE,
                    )
                )
                if repaired_admission is not None:
                    admission_calls += int(
                        getattr(repaired_admission, "request_count", 1)
                    )
                    request_sizes.extend(
                        tuple(
                            getattr(
                                repaired_admission,
                                "request_sizes",
                                (),
                            )
                            or ()
                        )
                    )
                plan = repaired_plan
                admission = repaired_admission
                admission_errors = repaired_errors
    result = (
        _admitted_action_queue(plan, admission, state)
        if admission is not None and admission.valid and plan is not None
        else _unresolved_action_queue(clauses, admission_errors)
    )
    return _with_metrics(
        result,
        started,
        request_sizes=request_sizes,
        stage_a_calls=stage_a_calls,
        stage_b_calls=stage_b_calls,
        admission_calls=admission_calls,
        owner_count=owner_count,
        unit_count=unit_count,
    )


def _conservative_closed_enum_intake_repair(
    candidate: Mapping[str, Any],
    plan: Any,
    admission: Any,
) -> dict[str, Any] | None:
    """Replace rejected enum guesses with one registry-owned typed intake.

    Stage A already established that the source is a domain request. The
    independent admission reviewer may still reject Stage B's concrete enum
    selection. When the registry exposes exactly one intake for that same
    group, asking the user is the only lossless projection.
    """

    payload = json.loads(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    )
    actions = payload.get("actions")
    units = payload.get("semantic_units")
    if not isinstance(actions, list) or not isinstance(units, list):
        return None
    action_ids = tuple(getattr(plan, "action_ids", ()) or ())
    verdicts = {
        str(row.get("action_id") or ""): row
        for row in tuple(getattr(admission, "action_verdicts", ()) or ())
        if isinstance(row, Mapping)
    }
    changed = False
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Mapping) or index >= len(action_ids):
            continue
        verdict = verdicts.get(str(action_ids[index]) or "")
        if (
            not isinstance(verdict, Mapping)
            or str(verdict.get("verdict") or "") == "admit"
        ):
            continue
        action = dict(raw_action)
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.target_group:
            continue
        closed_enum_arguments = [
            argument
            for argument in semantic_grounding_arguments(action)
            if isinstance(
                (ACTION_ARGUMENT_SCHEMAS.get(argument) or {}).get("enum"),
                list,
            )
            and action.get(argument) not in {"", None}
        ]
        if not closed_enum_arguments:
            continue
        intake_specs = [
            candidate_spec
            for candidate_spec in ACTION_SPECS
            if candidate_spec.incomplete_mutation_intake
            and candidate_spec.target_group == spec.target_group
            and set(candidate_spec.required_arguments).issubset(
                {"source_evidence"}
            )
        ]
        if len(intake_specs) != 1:
            continue
        owned_units = [
            unit
            for unit in units
            if isinstance(unit, Mapping)
            and str(unit.get("disposition") or "") == "action"
            and index in tuple(unit.get("action_indexes") or ())
            and str(unit.get("source_text") or "").strip()
        ]
        if len(owned_units) != 1:
            continue
        replacement = {
            "type": intake_specs[0].action_type,
            "source_evidence": str(
                owned_units[0].get("source_text") or ""
            ).strip(),
        }
        try:
            actions[index] = validate_action_contract(replacement)
        except ValueError:
            continue
        changed = True
    return payload if changed else None


def _turn_clauses(
    state: AgentGraphState,
    text: str,
) -> tuple[TurnClause, ...]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    pending = dict(state.get("pending_question") or {})
    validation = dict(pending.get("validation") or {})
    if (
        raw
        and pending.get("manual_input_allowed") is True
        and str(validation.get("value_type") or "") == "evidence_contribution"
    ):
        return (TurnClause("clause-1", raw, "structured"),)
    return segment_user_turn(raw)


def _stage_a_prompt() -> str:
    return (
        "You are the Stage A semantic partitioner for AnyChain Benchmark Agent. "
        "Return one strict JSON object with semantic_units and reason only. "
        "Identify ordered minimal exact source anchors for every semantic operation. The "
        "Harness expands the intervals between prose anchors into the lossless partition, so "
        "do not repeat surrounding prose merely to cover conjunctions or punctuation. "
        "Do not choose product actions, mutate state, answer the user, or copy values from state. "
        "Each semantic unit must contain exactly: unit_id, clause_id, source_text, operation, "
        "owner_routes, reason. operation must be one of the supplied universal_operations. "
        "Treat universal_operation_purposes as authoritative. pending_answer requires a present "
        "commitment to the active question; a hypothetical, counterfactual, consequence, or "
        "explanation question is consultation and never authorizes the pending action. A request "
        "to analyze logs, errors, traces, or other evidence is evidence_analysis even when the "
        "user has not pasted the evidence yet. "
        "owner_routes is an ordered list of {owner,group}; use an empty list only for context "
        "or unresolved. A unit may route to several owners when one atomic structured block "
        "contains independently owned values. Preserve questions, corrections, contradictions, "
        "pending answers, navigation, multiline evidence, and every sibling demand separately. "
        "input_shape and structured_candidates describe terminal syntax only; they do not classify "
        "intent, authorize an action, or assign an owner. Treat each parser candidate as a read-only "
        "structural fact. Route the user's actual semantic request using the supplied owner/group "
        "registries and routing purposes, including several immutable owner routes when one structured "
        "block supplies independently owned values. Never route every structured value to one default "
        "owner. If the semantic operation or any required owner/group route is uncertain, mark the "
        "unit unresolved; do not guess, omit, or repair a route. "
        "When a turn answers the active pending question and also supplies sibling configuration, "
        "emit the exact answer excerpt as one pending_answer unit routed only to coordinator, using "
        "the active pending_question.group as that route's group, and "
        "emit every sibling as separate domain_request units. Never label a normal domain_request "
        "as a pending answer, and never combine a pending answer with a sibling mutation. "
        "Respect pending_question.value_domain and registered_semantic_value_domains. A known "
        "identity or closed value owned by another registered dimension is routed to that "
        "dimension's owner; it is not consumed by an unrelated manual-text question merely "
        "because that question is pending. If the source explicitly claims that a registered "
        "value is instead a new identity, "
        "mark that unit unresolved so the user can disambiguate. "
        "registered_value_mentions contains only registry values that occur in this turn. "
        "It proves spelling and ownership, not intent. Use the complete source meaning to "
        "decide whether each mention participates in a mutation, consultation, hypothetical, "
        "comparison, or context. When one compact phrase uses a registered value plus adjacent "
        "operation framing to express one domain request, keep the complete phrase in one "
        "domain_request unit. Do not split its framing into a second pending_answer or "
        "unresolved demand unless that framing independently answers the active question or "
        "requests another operation. "
        "A semantic option selection and adjacent prose that only explains the reason, uncertainty, "
        "basis, referential application, or declared completion effect for that same selection form "
        "one pending_answer operation even when punctuation or line breaks create several clauses. "
        "When the source rejects one or more pending options and affirmatively requests another "
        "declared option's meaning or action, form one pending_answer for the affirmed option. "
        "The rejected alternative is contrast evidence, not a pending answer of its own, and the "
        "affirmed option action is not also a sibling domain request. "
        "pending_barrier_contract governs pending-answer ownership and queue scheduling only. "
        "It never forbids semantic routing to another registered group. In particular, when "
        "registered_cross_group_values is route_to_registered_owner, route an independent "
        "registered value to that owner even under exclusive_owner; do not mark it unresolved "
        "merely because another group has a pending question. "
        "The supporting clause may be context, but it is not a separate demand. The explanation is not a "
        "domain request merely because it discusses the option's subject. Split it only when the "
        "user independently asks for research, explanation, navigation, or a mutation. "
        "Apply the same transaction boundary to one parser-proven manual value for the active "
        "pending question: adjacent prose that only states the value's purpose, scope, exclusion, "
        "or non-application is support for that pending answer, not a sibling domain request. "
        "Keep a separate operation only when the prose independently requests another value, "
        "mutation, consultation, navigation, or analysis. "
        "A semantic unit represents one indivisible source excerpt. Split distinct source "
        "excerpts even when they share an owner. When one compact prose or structured excerpt "
        "directly supplies several independently owned values, keep one unit with several "
        "owner routes instead of duplicating or overlapping its source excerpt. "
        "Universal operation ownership is fixed by universal_operation_owners. The group on a "
        "consultation route identifies its subject but never transfers read-only consultation "
        "ownership away from orientation. "
        "Use exact owner and group identifiers from the supplied registries. "
        "Do not infer a mutation from examples, hypothetical values, logs, or current state."
    )


def _stage_a_payload(
    state: AgentGraphState,
    text: str,
    clauses: Sequence[TurnClause],
) -> dict[str, Any]:
    pending = dict(state.get("pending_question") or {})
    structured_candidates = []
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = extract_structured_input_candidates(clause.text)
        if candidates:
            structured_candidates.append({
                "clause_id": clause.clause_id,
                **candidates,
            })
    return {
        "product": "AnyChain Benchmark Agent",
        "user_text": text,
        "clauses": [clause.as_dict() for clause in clauses],
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "opening",
        "pending_question": pending,
        "pending_barrier_contract": pending_barrier_semantics(pending),
        "registered_semantic_value_domains": list(
            registered_semantic_value_domains()
        ),
        "registered_value_mentions": list(
            _registered_value_mentions(text)
        ),
        "pending_typed_candidates": [
            value
            for clause in clauses
            for value in typed_pending_value_candidates(clause.text, pending)
        ],
        "group_readiness": workflow_snapshot(state).get("group_states") or {},
        "interruption_stack": state.get("interruption_stack") or [],
        "workflow_goals": state.get("workflow_goals") or [],
        "structured_candidates": structured_candidates,
        "universal_operations": sorted(_UNIVERSAL_OPERATIONS),
        "universal_operation_purposes": dict(SEMANTIC_OPERATION_PURPOSES),
        "universal_operation_owners": dict(_UNIVERSAL_OPERATION_OWNER),
        "owners": sorted(_OWNERS),
        "groups": [
            {
                "name": row["name"],
                "owner": row["owner"],
                "category": row["category"],
                "fields": row["fields"],
                "depends_on": row["depends_on"],
                "entry_actions": row["entry_actions"],
                "routing_purposes": [
                    {
                        "owner": action["owner"],
                        "purpose": action["purpose"],
                    }
                    for action in action_schema(groups=frozenset({row["name"]}))
                ],
            }
            for row in group_schema()
        ],
    }


def _registered_value_mentions(text: str) -> tuple[dict[str, Any], ...]:
    """Project only registry values that occur in the complete source turn."""

    records = semantic_value_domain_conflicts(
        text,
        owning_group="",
    )
    allowed_keys = (
        "action_type",
        "argument",
        "target_group",
        "semantic_owner",
        "value",
        "canonical_value",
        "domain_kind",
    )
    return tuple({
        key: record.get(key)
        for key in allowed_keys
    } for record in records)


def _cross_domain_pending_errors(
    partition: Sequence[Mapping[str, Any]],
    state: AgentGraphState,
) -> tuple[str, ...]:
    """Enforce registered value ownership while another group is pending."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return ()
    pending_group = str(pending.get("group") or "")
    errors: list[str] = []
    for unit in partition:
        operation = str(unit.get("operation") or "")
        source = str(unit.get("source_text") or "")
        conflicts = semantic_value_domain_conflicts(
            source,
            owning_group=pending_group,
            pending_question=pending,
        )
        if not conflicts:
            continue
        if operation == "pending_answer":
            for record in conflicts:
                errors.append(
                    "Stage A assigned a registered semantic value to the wrong "
                    "pending owner: "
                    f"{unit.get('unit_id')}/{record.get('semantic_owner')}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
            continue
        if operation == "unresolved":
            for record in conflicts:
                errors.append(
                    "Stage A left a registered cross-group semantic value "
                    "unresolved while another group was pending; route an "
                    "independent request to its registered owner instead of "
                    "treating the pending barrier as a routing barrier: "
                    f"{unit.get('unit_id')}/{record.get('semantic_owner')}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
            continue
        if operation != "domain_request":
            continue
        routed_groups = {
            str(route.get("group") or "")
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        }
        for record in conflicts:
            target_group = str(record.get("target_group") or "")
            if target_group and target_group not in routed_groups:
                errors.append(
                    "Stage A routed a registered cross-group semantic value "
                    "to the wrong domain owner: "
                    f"{unit.get('unit_id')}/{target_group}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
    return tuple(dict.fromkeys(errors))


def _validate_partition_document(
    text: str,
    clauses: Sequence[TurnClause],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], ("Stage A did not return strict JSON",)
    if not isinstance(payload, dict) or set(payload) - {"semantic_units", "reason"}:
        return [], ("Stage A returned undeclared top-level fields",)
    raw_units = payload.get("semantic_units")
    if not isinstance(raw_units, list):
        return [], ("Stage A semantic_units is not a list",)
    raw_units = _retain_unclaimed_prose_clauses(raw_units, clauses)
    partition = validate_semantic_partition(raw_units, clauses)
    errors = list(partition.errors)
    output: list[dict[str, Any]] = []
    for raw in partition.units:
        unit = dict(raw)
        unit_id = str(unit.get("unit_id") or "")
        unit["unit_id"] = unit_id
        unit["clause_id"] = str(unit.get("clause_id") or "")
        if set(unit) - (_PARTITION_KEYS | {"start", "end"}):
            errors.append(f"Stage A unit has undeclared fields: {unit_id}")
        operation = str(unit.get("operation") or "")
        if operation not in _UNIVERSAL_OPERATIONS:
            errors.append(f"Stage A unit has invalid operation: {unit_id}")
        routes = unit.get("owner_routes")
        if not isinstance(routes, list):
            errors.append(f"Stage A owner_routes is not a list: {unit_id}")
            routes = []
        normalized_routes: list[dict[str, str]] = []
        route_identities: set[tuple[str, str]] = set()
        for raw_route in routes:
            if not isinstance(raw_route, Mapping) or set(raw_route) != _ROUTE_KEYS:
                errors.append(f"Stage A route contract is invalid: {unit_id}")
                continue
            owner = str(raw_route.get("owner") or "")
            group = str(raw_route.get("group") or "")
            if owner not in _OWNERS:
                errors.append(f"Stage A route owner is invalid: {unit_id}/{owner}")
            group_spec = GROUP_SPEC_BY_NAME.get(group) if group else None
            if operation == "domain_request" and not group:
                errors.append(
                    f"Stage A domain request route has no group: {unit_id}/{owner}"
                )
            if group and group_spec is None:
                errors.append(f"Stage A route group is invalid: {unit_id}/{group}")
            if (
                group_spec is not None
                and owner not in {"coordinator", "orientation", "analysis"}
                and group_spec.owner != owner
            ):
                errors.append(
                    f"Stage A route owner/group mismatch: {unit_id}/{owner}/{group}"
                )
            identity = (owner, group)
            if identity in route_identities:
                errors.append(
                    f"Stage A route is duplicated: {unit_id}/{owner}/{group}"
                )
            route_identities.add(identity)
            normalized_routes.append({"owner": owner, "group": group})
        required_owner = _UNIVERSAL_OPERATION_OWNER.get(operation)
        if required_owner is not None and (
            len(normalized_routes) != 1
            or normalized_routes[0]["owner"] != required_owner
        ):
            errors.append(
                f"Stage A universal operation owner mismatch: "
                f"{unit_id}/{operation}/{required_owner}"
            )
        if operation in {"context", "unresolved"} and normalized_routes:
            errors.append(f"Stage A non-action unit declares routes: {unit_id}")
        if operation not in {"context", "unresolved"} and not normalized_routes:
            errors.append(f"Stage A actionable unit has no route: {unit_id}")
        unit["owner_routes"] = normalized_routes
        output.append(unit)
    return output, tuple(dict.fromkeys(errors))


def _retain_unclaimed_prose_clauses(
    raw_units: Sequence[Any],
    clauses: Sequence[TurnClause],
) -> list[Any]:
    """Keep unclaimed prose visible to the independent omission reviewer."""

    clause_order = {
        clause.clause_id: index
        for index, clause in enumerate(clauses)
    }
    rows_by_clause: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_order: list[int] = []
    used_ids: set[str] = set()
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            return list(raw_units)
        clause_id = str(raw.get("clause_id") or "")
        if clause_id not in clause_order:
            return list(raw_units)
        observed_order.append(clause_order[clause_id])
        rows_by_clause[clause_id].append(raw)
        used_ids.add(str(raw.get("unit_id") or ""))
    if observed_order != sorted(observed_order):
        return list(raw_units)

    output: list[Any] = []
    for clause in clauses:
        rows = rows_by_clause.get(clause.clause_id, [])
        if rows:
            output.extend(rows)
            continue
        if clause.input_shape != "prose":
            continue
        sequence = 1
        while True:
            unit_id = f"__harness_context_{clause.clause_id}_{sequence}"
            if unit_id not in used_ids:
                used_ids.add(unit_id)
                break
            sequence += 1
        output.append({
            "unit_id": unit_id,
            "clause_id": clause.clause_id,
            "source_text": clause.text,
            "operation": "context",
            "owner_routes": [],
            "reason": "unclaimed prose retained for independent coverage review",
        })
    return output


def _partition_after_stage_a_admission(
    partition: Sequence[Mapping[str, Any]],
    redundant_unit_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate immutable source coverage from executable owner compilation."""

    source_partition: list[dict[str, Any]] = []
    compilation_partition: list[dict[str, Any]] = []
    for raw in partition:
        unit = dict(raw)
        if str(unit.get("unit_id") or "") in redundant_unit_ids:
            unit["operation"] = "context"
            unit["owner_routes"] = []
            unit["reason"] = (
                f"{str(unit.get('reason') or '').strip()} "
                "Stage A admission classified this as a non-executable duplicate."
            ).strip()
            source_partition.append(unit)
            continue
        source_partition.append(unit)
        compilation_partition.append(dict(unit))
    return source_partition, compilation_partition


def _canonicalize_atomic_evidence_partition(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve one typed evidence contribution as one lossless owner unit."""

    pending = dict(state.get("pending_question") or {})
    if (
        str((pending.get("validation") or {}).get("value_type") or "")
        != "evidence_contribution"
        or len(clauses) != 1
        or clauses[0].input_shape != "structured"
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    executable = [
        unit
        for unit in compilation_partition
        if str(unit.get("operation") or "") != "context"
    ]
    if (
        not executable
        or any(
            str(unit.get("operation") or "") != "pending_answer"
            for unit in executable
        )
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    atomic = {
        "unit_id": str(executable[0].get("unit_id") or "evidence-unit"),
        "clause_id": clauses[0].clause_id,
        "source_text": clauses[0].text,
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(pending.get("group") or state.get("active_group") or ""),
        }],
        "reason": (
            "Typed evidence_contribution remains one lossless pending-owner "
            "transaction."
        ),
    }
    return [atomic], [dict(atomic)]


def _canonicalize_unique_manual_pending_partition(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one typed manual value executable and retain its wrapper as context."""

    pending = dict(state.get("pending_question") or {})
    if (
        pending.get("manual_input_allowed") is not True
        or pending.get("options")
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    pending_units = [
        unit
        for unit in compilation_partition
        if str(unit.get("operation") or "") == "pending_answer"
    ]
    if len(pending_units) <= 1:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    candidates: list[tuple[str, str]] = []
    for clause in clauses:
        for value in typed_pending_value_candidates(clause.text, pending):
            identity = pending_value_identity(value, pending)
            if identity:
                candidates.append((clause.clause_id, identity))
    identities = {identity for _clause_id, identity in candidates}
    candidate_clause_ids = {
        clause_id for clause_id, _identity in candidates
    }
    if len(identities) != 1 or len(candidate_clause_ids) != 1:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    candidate_clause_id = next(iter(candidate_clause_ids))
    if not any(
        str(unit.get("clause_id") or "") == candidate_clause_id
        for unit in pending_units
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    clause_by_id = {clause.clause_id: clause for clause in clauses}
    candidate_clause = clause_by_id[candidate_clause_id]
    candidate_unit = next(
        unit
        for unit in pending_units
        if str(unit.get("clause_id") or "") == candidate_clause_id
    )
    atomic = {
        "unit_id": str(
            candidate_unit.get("unit_id")
            or f"manual-{candidate_clause_id}"
        ),
        "clause_id": candidate_clause_id,
        "source_text": candidate_clause.text,
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(
                pending.get("group")
                or state.get("active_group")
                or ""
            ),
        }],
        "reason": (
            "The complete turn contains one typed manual candidate; wrapper "
            "clauses remain visible as non-executable source context."
        ),
    }

    def canonicalize(
        partition: Sequence[Mapping[str, Any]],
        *,
        retain_context: bool,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        emitted = False
        for raw in partition:
            unit = dict(raw)
            if str(unit.get("operation") or "") != "pending_answer":
                result.append(unit)
                continue
            if (
                str(unit.get("clause_id") or "") == candidate_clause_id
                and not emitted
            ):
                result.append(dict(atomic))
                emitted = True
                continue
            if retain_context:
                unit["operation"] = "context"
                unit["owner_routes"] = []
                unit["reason"] = (
                    "Stage A classified this complete wrapper as part of the "
                    "same unique manual transaction."
                )
                result.append(unit)
        if not emitted:
            result.append(dict(atomic))
        return result

    return (
        canonicalize(source_partition, retain_context=True),
        canonicalize(compilation_partition, retain_context=False),
    )


def _stage_a_admission_prompt() -> str:
    return (
        "You are the independent Stage A coverage authority for AnyChain Benchmark Agent. "
        "You are not a planner and must not create product actions. Compare the complete user "
        "clauses with the immutable semantic units and the supplied owner/group purposes. "
        "Treat universal_operation_purposes as authoritative. Reject pending_answer for a "
        "hypothetical, counterfactual, consequence, explanation, or capability question because "
        "it contains no present authorization. Require evidence_analysis for a request to ingest "
        "or analyze logs, errors, traces, or evidence even when the evidence will be pasted later. "
        "Return one strict JSON object with exactly unit_verdicts, clause_verdicts, and reason. "
        "unit_verdicts contains exactly one row per supplied unit in order: "
        "{unit_id,verdict:'complete'|'redundant'|'unresolved',supports_unit_id,reason}. "
        "supports_unit_id must be empty for complete/unresolved. Use redundant only when the "
        "unit duplicates, contrasts with, or merely restates one other complete unit in this "
        "turn; set supports_unit_id to that exact distinct complete unit id regardless of whether "
        "the supporting prose appears before or after it, and require the redundant unit's clause "
        "to contain no omitted demand. "
        "registered_value_mentions proves only spelling and owner identity. When one complete "
        "unit already represents a compact registered-domain request, adjacent operation framing "
        "that adds no independent value, question, navigation, analysis, or mutation supports "
        "that unit; it is not a second pending answer or unresolved demand. "
        "clause_verdicts contains exactly one "
        "row per supplied clause in order: {clause_id,verdict:'complete'|'omitted'|'unresolved',"
        "omitted_owner_routes:[{owner,group}],reason}. A clause is complete only when every "
        "independent present request, question, correction, navigation, value, and evidence "
        "contribution appears in an appropriate semantic unit. Connective or framing prose is "
        "not an omitted demand. A context unit has no independently admissible product action; "
        "clause_verdicts, not its unit verdict, are authoritative for detecting whether its "
        "source actually contains an omitted demand. A reason, uncertainty statement, or basis attached to one "
        "pending option selection supports that same selection unless it independently asks "
        "for research, explanation, navigation, or mutation; do not approve a fictitious "
        "second demand created only from such support. Compare alleged sibling demands with the "
        "immutable pending_question option actions and completion effects. Mark a unit redundant "
        "when it only restates the selected option's declared effect; do not count the same effect "
        "once as a pending answer and again as a domain request. Reject a partition that treats "
        "rejection of one declared option as selecting that rejected option while separately "
        "representing an affirmed declared option as a sibling demand. When pending_question "
        "accepts manual input and pending_typed_candidates proves one candidate, prose that only "
        "limits that candidate's purpose, application scope, or exclusions supports the candidate "
        "unit. Mark such wrapper units redundant unless they independently request another value, "
        "mutation, consultation, navigation, or analysis. If a demand is absent, return omitted and identify its declared "
        "owner/group route; use unresolved when no safe route can be identified. Do not accept "
        "planner reason text as evidence and never invent source text, ids, owners, or groups."
    )


def _review_stage_a_partition(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[int, ...], frozenset[str]]:
    prompt = _stage_a_admission_prompt()
    payload = {
        "user_text": stage_a_payload["user_text"],
        "clauses": stage_a_payload["clauses"],
        "semantic_units": [dict(unit) for unit in partition],
        "pending_question": dict(stage_a_payload.get("pending_question") or {}),
        "pending_barrier_contract": dict(
            stage_a_payload.get("pending_barrier_contract") or {}
        ),
        "pending_typed_candidates": list(
            stage_a_payload.get("pending_typed_candidates") or []
        ),
        "registered_semantic_value_domains": list(
            stage_a_payload.get("registered_semantic_value_domains") or []
        ),
        "registered_value_mentions": list(
            stage_a_payload.get("registered_value_mentions") or []
        ),
        "groups": stage_a_payload["groups"],
        "universal_operations": stage_a_payload["universal_operations"],
        "universal_operation_purposes": dict(
            stage_a_payload.get("universal_operation_purposes")
            or SEMANTIC_OPERATION_PURPOSES
        ),
    }
    request_sizes: list[int] = []
    response = ""
    contract_errors: tuple[str, ...] = ()
    semantic_errors: tuple[str, ...] = ()
    redundant_unit_ids: frozenset[str] = frozenset()
    for attempt in range(2):
        request_payload = dict(payload)
        request_prompt = prompt
        if attempt:
            request_payload["contract_repair"] = {
                "prior_invalid_output": response,
                "validation_errors": list(contract_errors),
                "instruction": (
                    "Return a complete replacement admission document. Preserve "
                    "the source meaning while correcting every structural and "
                    "cross-verdict consistency error. Change unit or clause verdicts "
                    "when the reported inconsistency requires it."
                ),
            }
            request_prompt = (
                f"{prompt} This is a contract-repair attempt. The prior document "
                f"was structurally rejected for: {'; '.join(contract_errors)}. "
                "Return one complete replacement document with every required row "
                "and key. Do not weaken or invent the source meaning merely to pass; "
                "make unit and clause verdicts mutually consistent."
            )
        request_sizes.append(_wire_size(request_prompt, request_payload))
        response = request_semantic_compilation(
            provider,
            system_prompt=request_prompt,
            request_payload=request_payload,
            max_tokens=2200,
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        contract_errors, semantic_errors, redundant_unit_ids = (
            _validate_stage_a_admission_document(
                response,
                stage_a_payload,
                partition,
            )
        )
        if not contract_errors:
            return semantic_errors, tuple(request_sizes), redundant_unit_ids
    return (
        tuple(dict.fromkeys((*contract_errors, *semantic_errors))),
        tuple(request_sizes),
        frozenset(),
    )


def _validate_stage_a_admission_document(
    response: str,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Separate malformed reviewer output from a valid semantic rejection."""

    try:
        document = json.loads(response)
    except json.JSONDecodeError:
        return ("Stage A admission did not return strict JSON",), (), frozenset()
    contract_errors: list[str] = []
    semantic_errors: list[str] = []
    redundant_unit_ids: set[str] = set()
    if not isinstance(document, Mapping) or set(document) != {
        "unit_verdicts",
        "clause_verdicts",
        "reason",
    }:
        contract_errors.append(
            "Stage A admission returned an invalid top-level contract"
        )
        document = {}
    unit_verdicts = document.get("unit_verdicts")
    clause_verdicts = document.get("clause_verdicts")
    expected_unit_ids = [str(unit["unit_id"]) for unit in partition]
    unit_operations = {
        str(unit["unit_id"]): str(unit.get("operation") or "")
        for unit in partition
    }
    expected_clause_ids = [
        str(clause["clause_id"]) for clause in stage_a_payload["clauses"]
    ]
    if (
        not isinstance(unit_verdicts, list)
        or len(unit_verdicts) != len(expected_unit_ids)
        or not all(isinstance(row, Mapping) for row in unit_verdicts)
        or [str(row.get("unit_id") or "") for row in unit_verdicts]
        != expected_unit_ids
    ):
        contract_errors.append(
            "Stage A admission unit verdict order or cardinality mismatch"
        )
        unit_verdicts = []
    if (
        not isinstance(clause_verdicts, list)
        or len(clause_verdicts) != len(expected_clause_ids)
        or not all(isinstance(row, Mapping) for row in clause_verdicts)
        or [str(row.get("clause_id") or "") for row in clause_verdicts]
        != expected_clause_ids
    ):
        contract_errors.append(
            "Stage A admission clause verdict order or cardinality mismatch"
        )
        clause_verdicts = []
    valid_routes = {
        (str(group["owner"]), str(group["name"]))
        for group in stage_a_payload["groups"]
    }
    for row in unit_verdicts:
        if set(row) != {
            "unit_id",
            "verdict",
            "supports_unit_id",
            "reason",
        }:
            contract_errors.append(
                "Stage A admission unit verdict contract is invalid"
            )
            continue
        if not str(row.get("reason") or "").strip():
            contract_errors.append(
                "Stage A admission unit verdict has no reason"
            )
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "redundant", "unresolved"}:
            contract_errors.append(
                "Stage A admission unit verdict is invalid"
            )
        elif verdict == "redundant":
            redundant_unit_ids.add(str(row.get("unit_id") or ""))
        elif str(row.get("supports_unit_id") or ""):
            contract_errors.append(
                "Stage A admission non-redundant unit declares support"
            )
        elif (
            verdict != "complete"
            and unit_operations.get(str(row.get("unit_id") or "")) != "context"
        ):
            semantic_errors.append(
                f"Stage A admission found unresolved unit: {row.get('unit_id')}"
            )
    for row in clause_verdicts:
        if set(row) != {
            "clause_id",
            "verdict",
            "omitted_owner_routes",
            "reason",
        }:
            contract_errors.append(
                "Stage A admission clause verdict contract is invalid"
            )
            continue
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "omitted", "unresolved"}:
            contract_errors.append("Stage A admission clause verdict is invalid")
        if not str(row.get("reason") or "").strip():
            contract_errors.append(
                "Stage A admission clause verdict has no reason"
            )
        routes = row.get("omitted_owner_routes")
        if not isinstance(routes, list):
            contract_errors.append(
                "Stage A admission omitted_owner_routes is not a list"
            )
            routes = []
        for route in routes:
            if not isinstance(route, Mapping) or set(route) != _ROUTE_KEYS:
                contract_errors.append(
                    "Stage A admission omitted route contract is invalid"
                )
                continue
            identity = (
                str(route.get("owner") or ""),
                str(route.get("group") or ""),
            )
            if identity not in valid_routes:
                contract_errors.append(
                    f"Stage A admission returned an invalid omitted route: "
                    f"{identity[0]}/{identity[1]}"
                )
        if verdict in {"omitted", "unresolved"}:
            semantic_errors.append(
                f"Stage A admission found {verdict or 'invalid'} demand in "
                f"{row.get('clause_id')}"
            )
        elif routes:
            contract_errors.append(
                f"Stage A complete clause declares omitted routes: "
                f"{row.get('clause_id')}"
            )
    if not str(document.get("reason") or "").strip():
        contract_errors.append("Stage A admission has no reason")
    complete_clauses = {
        str(row.get("clause_id") or "")
        for row in clause_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    unresolved_non_context_units = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if (
            str(row.get("verdict") or "") == "unresolved"
            and unit_operations.get(str(row.get("unit_id") or "")) != "context"
        )
    }
    unit_clause = {
        str(unit["unit_id"]): str(unit.get("clause_id") or "")
        for unit in partition
    }
    reviewer_complete_units = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    for unit_id, operation in unit_operations.items():
        if operation == "unresolved" and unit_id in reviewer_complete_units:
            contract_errors.append(
                "Stage A admission is internally inconsistent: unresolved "
                f"source operation was declared complete: {unit_id}"
            )
    for unit_id in unresolved_non_context_units:
        if unit_clause.get(unit_id, "") in complete_clauses:
            contract_errors.append(
                "Stage A admission is internally inconsistent: non-context "
                f"unresolved unit belongs to a complete clause: {unit_id}"
            )
    complete_unit_ids = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    support_by_redundant_unit = {
        str(row.get("unit_id") or ""): str(row.get("supports_unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "redundant"
    }
    for unit_id in redundant_unit_ids:
        clause_id = unit_clause.get(unit_id, "")
        support_unit_id = support_by_redundant_unit.get(unit_id, "")
        if (
            clause_id not in complete_clauses
            or support_unit_id not in complete_unit_ids
            or support_unit_id == unit_id
        ):
            contract_errors.append(
                f"Stage A admission redundant unit has invalid support: {unit_id}"
            )
    return (
        tuple(dict.fromkeys(contract_errors)),
        tuple(dict.fromkeys(semantic_errors)),
        frozenset(redundant_unit_ids) if not contract_errors else frozenset(),
    )


def _owner_requests(
    partition: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    requests: dict[str, list[str]] = defaultdict(list)
    for unit in partition:
        unit_id = str(unit["unit_id"])
        for route in unit.get("owner_routes") or []:
            owner = str(route["owner"])
            if unit_id not in requests[owner]:
                requests[owner].append(unit_id)
    return {owner: tuple(unit_ids) for owner, unit_ids in requests.items()}


def _expand_partition_routes(
    partition: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give every owner/group route its own typed unit.

    Stage A may identify one indivisible source excerpt that supplies several
    values. Stage B and admission must not share ownership of one unit, so the
    Harness expands only the routing identity while preserving the exact same
    source span and semantic order.
    """

    output: list[dict[str, Any]] = []
    used_ids = {
        str(unit.get("unit_id") or "")
        for unit in partition
    }
    route_sequence = 0
    for raw in partition:
        unit = dict(raw)
        routes = [
            dict(route)
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        ]
        if len(routes) <= 1:
            output.append(unit)
            continue
        base_id = str(unit.get("unit_id") or "unit")
        for route in routes:
            while True:
                route_sequence += 1
                route_id = f"__harness_route_{route_sequence}"
                if route_id not in used_ids:
                    used_ids.add(route_id)
                    break
            expanded = dict(unit)
            expanded["unit_id"] = route_id
            expanded["parent_unit_id"] = base_id
            expanded["owner_routes"] = [route]
            output.append(expanded)
    return output


def _stage_b_prompt(owner: str) -> str:
    return (
        f"You are the Stage B command compiler for the AnyChain owner {owner}. "
        "Return one strict JSON object with actions, bindings, and reason only. "
        "Use only actions and arguments in owner_action_schema. Do not emit an action "
        "owned by another domain or a different group than the unit's owner_route. Bind "
        "every supplied unit_id exactly once. A binding "
        "is {unit_id,action_indexes,disposition,reason}; disposition is action or unresolved. "
        "Indexes are zero-based in this owner-local actions list. Preserve source order and "
        "exact source provenance. Every action object MUST be flat: place type and every "
        "allowed argument in the same object and NEVER emit an arguments object. For example, "
        "{\"type\":\"declared_type\",\"declared_argument\":\"value\"}, not "
        "{\"type\":\"declared_type\",\"arguments\":{...}}. Copy only keys explicitly listed "
        "in that action's allowed_arguments. When allowed_arguments is empty, the complete "
        "valid action object is {\"type\":\"declared_type\"}; source provenance remains in "
        "the binding and semantic unit and is not an action argument. Follow owner_action_schema "
        "exactly: never add an undeclared argument, and emit source_evidence only when that action "
        "declares it. Any emitted source_evidence must be one exact substring of a supplied semantic "
        "unit, never a paraphrase. For a concrete closed-enum selection, use the shortest exact "
        "affirmative source span that semantically selects that value; exclude contrast text and "
        "rejected alternatives from source_evidence. Configuration values remain proposals until the owning workflow "
        "validates and confirms them. Do not copy a value from owner_state "
        "unless the source unit explicitly supplies or confirms it. Do not add inferred "
        "identity, existence, protocol, canonical-name, or evidence-summary arguments to a "
        "selection action; downstream domain validation owns those facts. Questions and "
        "explanations are read-only actions. A concrete request owned by another group has "
        "no valid action in this owner schema: mark that binding unresolved so Stage A can be "
        "corrected, rather than coercing it into a superficially similar action. Ambiguous or "
        "incomplete demands remain unresolved instead of being guessed, except when the unit "
        "is an explicit initial-selection or replacement demand for its routed group and "
        "owner_action_schema declares an action for that same target_group with "
        "incomplete_mutation_intake=true. In that case emit that registered typed intake "
        "action instead of guessing a concrete value or marking the unit unresolved. This "
        "includes closed enums where the source rejects one value but leaves multiple legal "
        "values: exclusion is not a concrete selection, so use the registered intake to ask "
        "the user. Never synthesize an intake that is absent from owner_action_schema. "
        "When a semantic unit has operation=pending_answer, compare its complete exact source "
        "meaning with owner_state.pending_question and every declared option. If exactly one "
        "option is semantically selected, emit the coordinator-owned answer_pending action with "
        "selected_value equal to that option's exact value, omit answer, and include exact "
        "source_evidence from that unit. For valid manual input, emit answer and omit "
        "selected_value. When pending_question.options is empty, selected_value is always "
        "invalid: preserve the complete manual or evidence contribution in answer, or emit "
        "the pending contract's declared manual_action when that action belongs to this "
        "owner. Never reinterpret an array, boolean, id, method parameter, or other literal "
        "inside manual evidence as a declared option. The deterministic pending-question coordinator, not "
        "Stage B, resolves the immutable option.action and dispatches it to its registered owner. "
        "Never emit that specialized option.action directly from the coordinator compiler. "
        "The user does not need to repeat an option label or value verbatim; an unambiguous "
        "natural-language paraphrase, rejection of all listed alternatives, or stated "
        "uncertainty may select the corresponding declared option. If zero or several options "
        "fit, keep the unit unresolved. Never invent an option, and never consume an "
        "independent sibling request as rationale for the pending answer. Respect the pending "
        "question's value_domain: an open researched identity cannot consume a value declared "
        "by another registered closed domain."
    )


def _action_spec_applies(
    spec: Any,
    *,
    groups: frozenset[str],
    operations: frozenset[str],
) -> bool:
    declared_groups = frozenset(
        group
        for group in (spec.target_group, *spec.compiler_groups)
        if group
    )
    return any(
        operation in spec.semantic_operations
        and (
            operation != "domain_request"
            or bool(groups and groups.intersection(declared_groups))
        )
        for operation in operations
    )


def _action_schema_applies(
    row: Mapping[str, Any],
    *,
    groups: frozenset[str],
    operations: frozenset[str],
) -> bool:
    action_type = str(row.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    return bool(
        spec is not None
        and _action_spec_applies(
            spec,
            groups=groups,
            operations=operations,
        )
    )


def _allowed_action_types_for_partition(
    partition: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Expose only actions reachable from this immutable Stage A partition."""

    operations = frozenset(
        str(unit.get("operation") or "")
        for unit in partition
        if str(unit.get("operation") or "") not in {"context", "unresolved"}
    )
    groups = frozenset(
        str(route.get("group") or "")
        for unit in partition
        for route in unit.get("owner_routes") or []
        if isinstance(route, Mapping) and str(route.get("group") or "")
    )
    return frozenset(
        spec.action_type
        for spec in ACTION_SPECS
        if _action_spec_applies(
            spec,
            groups=groups,
            operations=operations,
        )
    )


def _stage_b_payload(
    state: AgentGraphState,
    owner: str,
    groups: frozenset[str],
    partition: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
) -> dict[str, Any]:
    selected = [
        {
            key: unit[key]
            for key in (
                "unit_id",
                "clause_id",
                "source_text",
                "operation",
                "owner_routes",
                "reason",
            )
        }
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    ]
    selected_operations = frozenset(
        str(unit.get("operation") or "")
        for unit in selected
    )
    owner_action_rows = [
        row
        for row in action_schema(owners=frozenset({owner}))
        if _action_schema_applies(
            row,
            groups=groups,
            operations=selected_operations,
        )
    ]
    owner_action_schema = list({
        str(row["type"]): row for row in owner_action_rows
    }.values())
    return {
        "owner": owner,
        "semantic_units": selected,
        "owner_action_schema": owner_action_schema,
        "owner_group_schema": [
            row
            for row in group_schema()
            if row["name"] in groups
        ],
        "registered_semantic_value_domains": list(
            registered_semantic_value_domains()
        ),
        "owner_state": owner_workflow_snapshot(state, owner, groups=groups),
    }


def _compile_owner_document(
    state: AgentGraphState,
    owner: str,
    groups: frozenset[str],
    partition: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[int, ...]]:
    prompt = _stage_b_prompt(owner)
    payload = _stage_b_payload(
        state,
        owner,
        groups,
        partition,
        unit_ids,
    )
    provider = provider_from_config()
    expected_groups = {
        str(unit["unit_id"]): frozenset(
            str(route.get("group") or "")
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        )
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    expected_operations = {
        str(unit["unit_id"]): str(unit.get("operation") or "")
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    expected_sources = {
        str(unit["unit_id"]): str(unit.get("source_text") or "")
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    request_sizes: list[int] = []
    response = ""
    document: dict[str, Any] = {}
    errors: tuple[str, ...] = ()
    for attempt in range(2):
        request_payload = payload
        request_prompt = prompt
        if attempt:
            request_prompt = (
                f"{prompt} This is a bounded contract repair. Preserve every source "
                "demand, but do not preserve a concrete semantic value that the "
                "validator reports as ungrounded. For a reported closed-enum grounding "
                "violation, remove the guessed concrete selection and use the registered "
                "incomplete_mutation_intake for the same routed group when one exists. "
                "Correct only the reported JSON, action-schema, ownership, route, "
                "grounding, or binding-contract violations. Do not turn an otherwise "
                "unresolved binding into an action merely because repair was requested."
            )
            request_payload = {
                **payload,
                "contract_repair": {
                    "rejected_document": response,
                    "validator_errors": list(errors),
                },
            }
        request_sizes.append(_wire_size(request_prompt, request_payload))
        response = request_semantic_compilation(
            provider,
            system_prompt=request_prompt,
            request_payload=request_payload,
            max_tokens=3200,
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        document, errors = _validate_owner_document(
            response,
            owner,
            unit_ids,
            expected_groups=expected_groups,
            expected_operations=expected_operations,
            expected_sources=expected_sources,
            pending_question=dict(state.get("pending_question") or {}),
        )
        if not errors:
            break
    return document, errors, tuple(request_sizes)


def _validate_owner_document(
    text: str,
    owner: str,
    unit_ids: Sequence[str],
    *,
    expected_groups: Mapping[str, frozenset[str]] | None = None,
    expected_operations: Mapping[str, str] | None = None,
    expected_sources: Mapping[str, str] | None = None,
    pending_question: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}, (f"Stage B {owner} did not return strict JSON",)
    if not isinstance(payload, dict) or set(payload) - {"actions", "bindings", "reason"}:
        return {}, (f"Stage B {owner} returned undeclared top-level fields",)
    actions = payload.get("actions")
    bindings = payload.get("bindings")
    errors: list[str] = []
    if not isinstance(actions, list):
        errors.append(f"Stage B {owner} actions is not a list")
        actions = []
    if not isinstance(bindings, list):
        errors.append(f"Stage B {owner} bindings is not a list")
        bindings = []
    normalized_actions: list[dict[str, Any]] = []
    normalized_action_slots: list[dict[str, Any] | None] = [None] * len(actions)
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            errors.append(f"Stage B {owner} action {index} is not an object")
            continue
        try:
            normalized = validate_action_contract(action)
        except ValueError as exc:
            errors.append(f"Stage B {owner} action {index} is invalid: {exc}")
            continue
        if (
            str(normalized.get("type") or "") == "answer_pending"
            and "selected_value" in normalized
            and not pending_option_value_exists(
                normalized.get("selected_value"),
                dict(pending_question or {}),
            )
        ):
            errors.append(
                f"Stage B {owner} action {index} selects a value absent from "
                "the active pending options"
            )
            continue
        spec = ACTION_BY_TYPE.get(str(normalized.get("type") or ""))
        if spec is None or spec.owner != owner:
            errors.append(f"Stage B {owner} emitted an unowned action at {index}")
            continue
        normalized_action_slots[index] = normalized
        normalized_actions.append(normalized)
    seen_units: list[str] = []
    for raw in bindings:
        if not isinstance(raw, Mapping) or set(raw) != _BINDING_KEYS:
            errors.append(f"Stage B {owner} binding contract is invalid")
            continue
        unit_id = str(raw.get("unit_id") or "")
        seen_units.append(unit_id)
        disposition = str(raw.get("disposition") or "")
        indexes = raw.get("action_indexes")
        if disposition not in {"action", "unresolved"}:
            errors.append(f"Stage B {owner} binding disposition is invalid: {unit_id}")
        if not isinstance(indexes, list) or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(actions)
            for index in indexes
        ):
            errors.append(f"Stage B {owner} binding indexes are invalid: {unit_id}")
        if disposition == "action" and not indexes:
            errors.append(f"Stage B {owner} actionable binding has no action: {unit_id}")
        if disposition == "unresolved" and indexes:
            errors.append(f"Stage B {owner} unresolved binding has actions: {unit_id}")
        groups = (expected_groups or {}).get(unit_id, frozenset())
        operation = str((expected_operations or {}).get(unit_id) or "")
        for index in indexes if isinstance(indexes, list) else []:
            if not isinstance(index, int) or index >= len(normalized_action_slots):
                continue
            normalized_action = normalized_action_slots[index]
            if normalized_action is None:
                continue
            spec = ACTION_BY_TYPE[str(normalized_action["type"])]
            if not _action_spec_applies(
                spec,
                groups=groups,
                operations=frozenset({operation}),
            ):
                errors.append(
                    f"Stage B {owner} action {spec.action_type} is outside "
                    f"unit route {unit_id}/{operation}/{sorted(groups)}"
                )
            source = str((expected_sources or {}).get(unit_id) or "")
            evidence = str(normalized_action.get("source_evidence") or "")
            quote = evidence if evidence and evidence in source else source
            for argument in semantic_grounding_arguments(normalized_action):
                argument_schema = ACTION_ARGUMENT_SCHEMAS.get(argument) or {}
                enum_values = argument_schema.get("enum")
                if not isinstance(enum_values, list):
                    continue
                if closed_enum_quote_names_only_competing_values(
                    exact_value=normalized_action.get(argument),
                    enum_values=enum_values,
                    quote=quote,
                ):
                    errors.append(
                        "Stage B closed-enum grounding quote names only "
                        f"competing values: {unit_id}/{spec.action_type}/{argument}"
                    )
    if seen_units != list(unit_ids):
        errors.append(f"Stage B {owner} binding order or cardinality mismatch")
    return {
        "actions": normalized_actions,
        "bindings": [
            dict(binding) for binding in bindings if isinstance(binding, Mapping)
        ],
    }, tuple(dict.fromkeys(errors))


def _merge_owner_documents(
    source_partition: Sequence[Mapping[str, Any]],
    routed_partition: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unit_order = {
        str(unit["unit_id"]): index for index, unit in enumerate(routed_partition)
    }
    action_rows: list[tuple[int, int, str, int, dict[str, Any]]] = []
    for owner, document in documents.items():
        action_units: dict[int, list[str]] = defaultdict(list)
        for binding in document.get("bindings") or []:
            for action_index in binding.get("action_indexes") or []:
                action_units[int(action_index)].append(str(binding["unit_id"]))
        for local_index, action in enumerate(document.get("actions") or []):
            owned_units = action_units.get(local_index, [])
            first_unit = min(
                (unit_order[unit_id] for unit_id in owned_units),
                default=10**9,
            )
            action_rows.append(
                (first_unit, local_index, owner, local_index, dict(action))
            )
    action_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    actions = [row[4] for row in action_rows]
    global_index = {
        (owner, local_index): index
        for index, (_unit, _order, owner, local_index, _action) in enumerate(
            action_rows
        )
    }
    bindings_by_owner = {
        owner: {
            str(binding["unit_id"]): binding
            for binding in document.get("bindings") or []
        }
        for owner, document in documents.items()
    }
    routed_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in routed_partition:
        routed_by_parent[
            str(unit.get("parent_unit_id") or unit["unit_id"])
        ].append(unit)
    semantic_units: list[dict[str, Any]] = []
    for unit in source_partition:
        unit_id = str(unit["unit_id"])
        operation = str(unit["operation"])
        indexes: list[int] = []
        unresolved = operation == "unresolved"
        for routed_unit in routed_by_parent.get(unit_id, (unit,)):
            route = next(iter(routed_unit.get("owner_routes") or []), {})
            owner = str(route.get("owner") or "")
            routed_unit_id = str(routed_unit["unit_id"])
            binding = bindings_by_owner.get(owner, {}).get(routed_unit_id)
            if not binding or str(binding.get("disposition") or "") == "unresolved":
                unresolved = True
                continue
            for local_index in binding.get("action_indexes") or []:
                value = global_index[(owner, int(local_index))]
                if value not in indexes:
                    indexes.append(value)
        disposition = (
            "context"
            if operation == "context"
            else "unresolved"
            if unresolved
            else "action"
        )
        semantic_units.append({
            "unit_id": unit_id,
            "clause_id": str(unit["clause_id"]),
            "start": unit.get("start"),
            "end": unit.get("end"),
            "source_text": str(unit["source_text"]),
            "disposition": disposition,
            "action_indexes": indexes if disposition == "action" else [],
            "reason": str(unit.get("reason") or ""),
        })
    return {
        "actions": actions,
        "semantic_units": semantic_units,
        "reason": "hierarchical Stage A/Stage B compilation",
    }


def _wire_size(prompt: str, payload: Mapping[str, Any]) -> int:
    return len(prompt.encode("utf-8")) + len(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def _with_metrics(
    result: dict[str, Any],
    started: float,
    *,
    request_sizes: Sequence[int],
    stage_a_calls: int,
    stage_b_calls: int,
    admission_calls: int,
    owner_count: int,
    unit_count: int,
) -> dict[str, Any]:
    output = dict(result)
    metrics = HierarchicalPlannerMetrics(
        stage_a_calls=stage_a_calls,
        stage_b_calls=stage_b_calls,
        admission_calls=admission_calls,
        prompt_bytes=sum(request_sizes),
        largest_request_bytes=max(request_sizes, default=0),
        elapsed_ms=int((time.monotonic() - started) * 1000),
        owner_count=owner_count,
        unit_count=unit_count,
    )
    output["planner_metrics"] = metrics.as_dict()
    return output
