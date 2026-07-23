"""Hierarchical semantic planning for current product turns.

Stage A partitions the complete turn and routes exact source units to domain
owners. Stage B compiles only the actions owned by each selected owner. The
result continues through the immutable whole-plan admission boundary until
the graph owns the complete turn receipt in Phase 3.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..llm.providers import provider_from_config
from ..llm.types import LLMTurnTimeoutError
from .action_registry import (
    ACTION_BY_TYPE,
    ACTION_SPECS,
    SEMANTIC_OPERATIONS,
    resolve_action_target_group,
    validate_action_contract,
)
from .context import (
    action_schema,
    group_schema,
    owner_workflow_snapshot,
    workflow_snapshot,
)
from .domains.environment import extract_structured_input_candidates
from .intent import (
    _admitted_action_queue,
    _admitted_plan_requires_pending_contract_adjudication,
    _review_bounded_semantic_candidate,
    _semantic_fulfillment_prompt,
    _unresolved_action_queue,
    prepare_hierarchical_candidate,
)
from .plan_coverage import TurnClause, segment_user_turn, validate_semantic_partition
from .questions import typed_pending_value_candidates
from .semantic_compiler import (
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


def resolve_product_action_queue(
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    """Compile one current turn without exposing the flat action registry."""

    started = time.monotonic()
    clauses = _turn_clauses(state, text)
    if not clauses:
        return _unresolved_action_queue(clauses, ("empty semantic turn",))
    request_sizes: list[int] = []
    stage_a_calls = 0
    stage_b_calls = 0
    admission_calls = 0
    try:
        provider = provider_from_config()
        stage_a_prompt = _stage_a_prompt()
        stage_a_payload = _stage_a_payload(state, text, clauses)
        partition: list[dict[str, Any]] = []
        partition_errors: tuple[str, ...] = ()
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
            request_sizes.append(_wire_size(request_prompt, request_payload))
            stage_a_calls += 1
            previous_output = request_semantic_compilation(
                provider,
                system_prompt=request_prompt,
                request_payload=request_payload,
                max_tokens=2600,
            )
            partition, partition_errors = _validate_partition_document(
                previous_output,
                clauses,
            )
            if not partition_errors:
                break
        if partition_errors:
            return _with_metrics(
                _unresolved_action_queue(clauses, partition_errors),
                started,
                request_sizes=request_sizes,
                stage_a_calls=stage_a_calls,
                stage_b_calls=stage_b_calls,
                admission_calls=admission_calls,
                owner_count=0,
                unit_count=len(partition),
            )
        partition_admission_errors, admission_request_size = (
            _review_stage_a_partition(
                provider,
                stage_a_payload,
                partition,
            )
        )
        request_sizes.append(admission_request_size)
        stage_a_calls += 1
        if partition_admission_errors:
            return _with_metrics(
                _unresolved_action_queue(clauses, partition_admission_errors),
                started,
                request_sizes=request_sizes,
                stage_a_calls=stage_a_calls,
                stage_b_calls=stage_b_calls,
                admission_calls=admission_calls,
                owner_count=0,
                unit_count=len(partition),
            )
        source_partition = partition
        partition = _expand_partition_routes(source_partition)

        requests = _owner_requests(partition)
        owner_inputs: list[tuple[str, tuple[str, ...], frozenset[str]]] = []
        for owner, unit_ids in requests.items():
            owner_groups = frozenset(
                route["group"]
                for unit in partition
                if unit["unit_id"] in unit_ids
                for route in unit["owner_routes"]
                if route["owner"] == owner and route["group"]
            )
            owner_inputs.append((owner, unit_ids, owner_groups))
        owner_documents: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(owner_inputs))) as executor:
            futures = [
                executor.submit(
                    _compile_owner_document,
                    state,
                    owner,
                    owner_groups,
                    partition,
                    unit_ids,
                )
                for owner, unit_ids, owner_groups in owner_inputs
            ]
            owner_results = [
                (owner, unit_ids, future.result())
                for (owner, unit_ids, _groups), future in zip(owner_inputs, futures)
            ]
        for owner, _unit_ids, (document, errors, request_size) in owner_results:
            stage_b_calls += 1
            request_sizes.append(request_size)
            if errors:
                return _with_metrics(
                    _unresolved_action_queue(clauses, errors),
                    started,
                    request_sizes=request_sizes,
                    stage_a_calls=stage_a_calls,
                    stage_b_calls=stage_b_calls,
                    admission_calls=admission_calls,
                    owner_count=len(requests),
                    unit_count=len(partition),
                )
            owner_documents[owner] = document

        candidate = _merge_owner_documents(
            source_partition,
            partition,
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
                owner_count=len(requests),
                unit_count=len(partition),
            )
        allowed_types = frozenset(ACTION_BY_TYPE)
        plan, admission, admission_errors = _review_bounded_semantic_candidate(
            provider,
            candidate_text,
            validation,
            state,
            clauses,
            allowed_action_types=allowed_types,
        )
        admission_calls = 1
        if plan is not None:
            request_sizes.append(
                len(
                    whole_plan_admission_prompt(
                        _semantic_fulfillment_prompt()
                    ).encode("utf-8")
                )
                + len(plan.request_json.encode("utf-8"))
            )
        if (
            admission is None
            or not admission.valid
            or plan is None
            or _admitted_plan_requires_pending_contract_adjudication(
                plan,
                admission,
                state,
                focused_adjudication=bool(state.get("pending_question")),
            )
        ):
            return _with_metrics(
                _unresolved_action_queue(clauses, admission_errors),
                started,
                request_sizes=request_sizes,
                stage_a_calls=stage_a_calls,
                stage_b_calls=stage_b_calls,
                admission_calls=admission_calls,
                owner_count=len(requests),
                unit_count=len(partition),
            )
        result = _admitted_action_queue(plan, admission, state)
        return _with_metrics(
            result,
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=len(requests),
            unit_count=len(partition),
        )
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        return _with_metrics(
            _unresolved_action_queue(
                clauses,
                (f"hierarchical planner failed: {type(exc).__name__}",),
            ),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=0,
            unit_count=0,
        )


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
        "owner_routes is an ordered list of {owner,group}; use an empty list only for context "
        "or unresolved. A unit may route to several owners when one atomic structured block "
        "contains independently owned values. Preserve questions, corrections, contradictions, "
        "pending answers, navigation, multiline evidence, and every sibling demand separately. "
        "When a turn answers the active pending question and also supplies sibling configuration, "
        "emit the exact answer excerpt as one pending_answer unit routed only to coordinator and "
        "emit every sibling as separate domain_request units. Never label a normal domain_request "
        "as a pending answer, and never combine a pending answer with a sibling mutation. "
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


def _stage_a_admission_prompt() -> str:
    return (
        "You are the independent Stage A coverage authority for AnyChain Benchmark Agent. "
        "You are not a planner and must not create product actions. Compare the complete user "
        "clauses with the immutable semantic units and the supplied owner/group purposes. "
        "Return one strict JSON object with exactly unit_verdicts, clause_verdicts, and reason. "
        "unit_verdicts contains exactly one row per supplied unit in order: "
        "{unit_id,verdict:'complete'|'unresolved',reason}. clause_verdicts contains exactly one "
        "row per supplied clause in order: {clause_id,verdict:'complete'|'omitted'|'unresolved',"
        "omitted_owner_routes:[{owner,group}],reason}. A clause is complete only when every "
        "independent present request, question, correction, navigation, value, and evidence "
        "contribution appears in an appropriate semantic unit. Connective or framing prose is "
        "not an omitted demand. If a demand is absent, return omitted and identify its declared "
        "owner/group route; use unresolved when no safe route can be identified. Do not accept "
        "planner reason text as evidence and never invent source text, ids, owners, or groups."
    )


def _review_stage_a_partition(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], int]:
    prompt = _stage_a_admission_prompt()
    payload = {
        "user_text": stage_a_payload["user_text"],
        "clauses": stage_a_payload["clauses"],
        "semantic_units": [dict(unit) for unit in partition],
        "groups": stage_a_payload["groups"],
        "universal_operations": stage_a_payload["universal_operations"],
    }
    response = request_semantic_compilation(
        provider,
        system_prompt=prompt,
        request_payload=payload,
        max_tokens=2200,
    )
    try:
        document = json.loads(response)
    except json.JSONDecodeError:
        return ("Stage A admission did not return strict JSON",), _wire_size(
            prompt,
            payload,
        )
    errors: list[str] = []
    if not isinstance(document, Mapping) or set(document) != {
        "unit_verdicts",
        "clause_verdicts",
        "reason",
    }:
        errors.append("Stage A admission returned an invalid top-level contract")
        document = {}
    unit_verdicts = document.get("unit_verdicts")
    clause_verdicts = document.get("clause_verdicts")
    expected_unit_ids = [str(unit["unit_id"]) for unit in partition]
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
        errors.append("Stage A admission unit verdict order or cardinality mismatch")
        unit_verdicts = []
    if (
        not isinstance(clause_verdicts, list)
        or len(clause_verdicts) != len(expected_clause_ids)
        or not all(isinstance(row, Mapping) for row in clause_verdicts)
        or [str(row.get("clause_id") or "") for row in clause_verdicts]
        != expected_clause_ids
    ):
        errors.append("Stage A admission clause verdict order or cardinality mismatch")
        clause_verdicts = []
    valid_routes = {
        (str(group["owner"]), str(group["name"]))
        for group in stage_a_payload["groups"]
    }
    for row in unit_verdicts:
        if set(row) != {"unit_id", "verdict", "reason"}:
            errors.append("Stage A admission unit verdict contract is invalid")
            continue
        if not str(row.get("reason") or "").strip():
            errors.append("Stage A admission unit verdict has no reason")
        if str(row.get("verdict") or "") != "complete":
            errors.append(
                f"Stage A admission found unresolved unit: {row.get('unit_id')}"
            )
    for row in clause_verdicts:
        if set(row) != {
            "clause_id",
            "verdict",
            "omitted_owner_routes",
            "reason",
        }:
            errors.append("Stage A admission clause verdict contract is invalid")
            continue
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "omitted", "unresolved"}:
            errors.append("Stage A admission clause verdict is invalid")
        if not str(row.get("reason") or "").strip():
            errors.append("Stage A admission clause verdict has no reason")
        routes = row.get("omitted_owner_routes")
        if not isinstance(routes, list):
            errors.append("Stage A admission omitted_owner_routes is not a list")
            routes = []
        for route in routes:
            if not isinstance(route, Mapping) or set(route) != _ROUTE_KEYS:
                errors.append("Stage A admission omitted route contract is invalid")
                continue
            identity = (
                str(route.get("owner") or ""),
                str(route.get("group") or ""),
            )
            if identity not in valid_routes:
                errors.append(
                    f"Stage A admission returned an invalid omitted route: "
                    f"{identity[0]}/{identity[1]}"
                )
        if verdict != "complete":
            errors.append(
                f"Stage A admission found {verdict or 'invalid'} demand in "
                f"{row.get('clause_id')}"
            )
        elif routes:
            errors.append(
                f"Stage A complete clause declares omitted routes: "
                f"{row.get('clause_id')}"
            )
    if not str(document.get("reason") or "").strip():
        errors.append("Stage A admission has no reason")
    return tuple(dict.fromkeys(errors)), _wire_size(prompt, payload)


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
        "in that action's allowed_arguments. source_evidence must be one exact substring of "
        "a supplied semantic unit, never a paraphrase. Do not copy a value from owner_state "
        "unless the source unit explicitly supplies or confirms it. Do not add inferred "
        "identity, existence, protocol, canonical-name, or evidence-summary arguments to a "
        "selection action; downstream domain validation owns those facts. Questions and "
        "explanations are read-only actions. A concrete request owned by another group has "
        "no valid action in this owner schema: mark that binding unresolved so Stage A can be "
        "corrected, rather than coercing it into a superficially similar action. Ambiguous or "
        "incomplete demands remain unresolved instead of being guessed."
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
        "owner_state": owner_workflow_snapshot(state, owner, groups=groups),
    }


def _compile_owner_document(
    state: AgentGraphState,
    owner: str,
    groups: frozenset[str],
    partition: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...], int]:
    prompt = _stage_b_prompt(owner)
    payload = _stage_b_payload(
        state,
        owner,
        groups,
        partition,
        unit_ids,
    )
    response = request_semantic_compilation(
        provider_from_config(),
        system_prompt=prompt,
        request_payload=payload,
        max_tokens=3200,
    )
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
    document, errors = _validate_owner_document(
        response,
        owner,
        unit_ids,
        expected_groups=expected_groups,
        expected_operations=expected_operations,
    )
    return document, errors, _wire_size(prompt, payload)


def _validate_owner_document(
    text: str,
    owner: str,
    unit_ids: Sequence[str],
    *,
    expected_groups: Mapping[str, frozenset[str]] | None = None,
    expected_operations: Mapping[str, str] | None = None,
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
