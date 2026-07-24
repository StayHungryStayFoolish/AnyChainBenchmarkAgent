"""Bounded semantic compilation and immutable whole-plan admission.

The configured model gets one compilation call and one independent admission
call.  Admission can reject a complete candidate, but it cannot edit the
candidate.  The caller may run one repair compilation followed by one final
admission; this module never retries per action or per semantic unit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Collection, Mapping, Sequence

from ..llm.types import LLMMessage, LLMRequest, ensure_turn_active
from .input_values import parse_weight_spec
from .semantic_policy import PENDING_CANDIDATE_SEMANTIC_POLICY


_ADMISSION_TOP_LEVEL_KEYS = frozenset({
    "plan_hash",
    "action_verdicts",
    "unit_verdicts",
    "reason",
})
_ACTION_VERDICT_KEYS = frozenset({
    "action_id",
    "verdict",
    "unit_ids",
    "evidence",
    "grounded_arguments",
    "pending_answer_argument",
    "turn_candidate_verdicts",
    "reason",
})
_ACTION_EVIDENCE_KEYS = frozenset({
    "unit_id",
    "quote",
    "relation",
    "support_relation",
})
_GROUNDED_ARGUMENT_KEYS = frozenset({
    "argument_name",
    "evidence_quote",
})
_TURN_CANDIDATE_VERDICT_KEYS = frozenset({
    "candidate_id",
    "verdict",
    "evidence_quote",
    "reason",
})
_UNIT_VERDICT_KEYS = frozenset({
    "unit_id",
    "verdict",
    "owner_action_ids",
    "evidence_quote",
    "omitted_action_type",
    "reason",
})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _quote_supports_pending_candidate(
    quote: str,
    candidate: Mapping[str, Any],
    pending: Mapping[str, Any],
    source_texts: Sequence[str],
) -> bool:
    """Require reviewer evidence to identify the candidate it adjudicates."""

    text = str(quote or "").strip()
    identity = str(candidate.get("identity") or "")
    value = candidate.get("value")
    if not text or not identity:
        return False
    exact_source_quote = any(text == source for source in source_texts)
    if identity.startswith(("rpc_weights:", "json:")):
        parsed = parse_weight_spec(text)
        candidate_mapping = (
            parse_weight_spec(value)
            if identity.startswith("rpc_weights:")
            else value
        )
        if parsed and _canonical_json(parsed) == _canonical_json(candidate_mapping):
            return True
        if not exact_source_quote or not isinstance(candidate_mapping, Mapping):
            return False
        collective = "\n".join(source_texts).casefold()
        collective_mapping = parse_weight_spec(collective)
        if collective_mapping:
            return _canonical_json(collective_mapping) == _canonical_json(
                candidate_mapping
            )
        for key, item in candidate_mapping.items():
            key_sources = [
                source
                for source in source_texts
                if str(key).casefold() in source.casefold()
            ]
            if not key_sources:
                return False
            numeric_literals = [
                number
                for source in key_sources
                for number in re.findall(r"(?<![\w.])[0-9]+(?:\.[0-9]+)?(?![\w.])", source)
            ]
            if numeric_literals and str(item) not in numeric_literals:
                return False
        return True
    if identity.startswith(("integer:", "number:")):
        literal = identity.split(":", 1)[1]
        literal_match = bool(
            re.search(
                rf"(?<![0-9.]){re.escape(literal)}(?![0-9.])",
                text,
            )
        )
        if literal_match:
            return True
        appears_in_source = any(
            re.search(
                rf"(?<![0-9.]){re.escape(literal)}(?![0-9.])",
                source,
            )
            for source in source_texts
        )
        return exact_source_quote and not appears_in_source
    candidate_literals: list[str] = []
    if identity.startswith(("scalar:", "enum:")):
        candidate_literals.append(str(value))
    if identity.startswith("option:"):
        for option in pending.get("options") or []:
            if not isinstance(option, Mapping) or option.get("value") != value:
                continue
            candidate_literals.extend(
                str(option.get(key) or "")
                for key in ("id", "label", "value")
            )
    for literal in candidate_literals:
        token = literal.strip()
        if token and re.search(
            rf"(?<![\w]){re.escape(token)}(?![\w])",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    appears_in_source = any(
        re.search(
            rf"(?<![\w]){re.escape(literal.strip())}(?![\w])",
            source,
            flags=re.IGNORECASE,
        )
        is not None
        for literal in candidate_literals
        for source in source_texts
        if literal.strip()
    )
    return exact_source_quote and not appears_in_source


def _strict_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("model response has an unterminated JSON fence")
        if lines[0].strip().lower() not in {"```", "```json"}:
            raise ValueError("model response uses an unsupported code fence")
        raw = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not one strict JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("model response is not a JSON object")
    return payload


@dataclass(frozen=True)
class ImmutableSemanticPlan:
    """A content-addressed candidate supplied to the independent reviewer."""

    plan_hash: str
    document_json: str
    request_json: str
    action_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        return json.loads(self.document_json)

    def request_payload(self) -> dict[str, Any]:
        return json.loads(self.request_json)


@dataclass(frozen=True)
class WholePlanAdmission:
    """Strict admission result for one immutable candidate."""

    valid: bool
    errors: tuple[str, ...]
    action_verdicts: tuple[dict[str, Any], ...] = ()
    unit_verdicts: tuple[dict[str, Any], ...] = ()
    response: Mapping[str, Any] | None = None
    request_count: int = 1
    request_sizes: tuple[int, ...] = ()

    def repair_context(self) -> dict[str, Any]:
        return {
            "errors": list(self.errors),
            "action_verdicts": [dict(row) for row in self.action_verdicts],
            "unit_verdicts": [dict(row) for row in self.unit_verdicts],
        }


def request_semantic_compilation(
    provider: Any,
    *,
    system_prompt: str,
    request_payload: Mapping[str, Any],
    max_tokens: int = 3600,
) -> str:
    """Run one call and leave malformed output to the caller's bounded repair."""

    ensure_turn_active()
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
            ),
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    ))
    raw = str(response.text or "")
    try:
        return _canonical_json(_strict_json_object(raw))
    except ValueError:
        return raw


def freeze_semantic_plan(
    document: Mapping[str, Any],
    *,
    action_records: Sequence[Mapping[str, Any]],
    unit_records: Sequence[Mapping[str, Any]],
    review_context: Mapping[str, Any],
) -> ImmutableSemanticPlan:
    """Freeze one validated candidate and its registry-derived review facts."""

    actions = document.get("actions")
    units = document.get("semantic_units")
    if not isinstance(actions, list) or not isinstance(units, list):
        raise ValueError("semantic plan requires actions and semantic_units lists")
    if len(action_records) != len(actions):
        raise ValueError("semantic plan action record count mismatch")
    if len(unit_records) != len(units):
        raise ValueError("semantic plan unit record count mismatch")

    normalized_actions = [dict(row) for row in action_records]
    normalized_units = [dict(row) for row in unit_records]
    action_ids = tuple(str(row.get("action_id") or "") for row in normalized_actions)
    unit_ids = tuple(str(row.get("unit_id") or "") for row in normalized_units)
    if any(not value for value in action_ids) or len(set(action_ids)) != len(action_ids):
        raise ValueError("semantic plan action ids are missing or duplicated")
    if any(not value for value in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("semantic plan unit ids are missing or duplicated")

    for index, row in enumerate(normalized_actions):
        if row.get("action_index") != index or row.get("action") != actions[index]:
            raise ValueError("semantic plan action record is not bound to the immutable action")
        mapped = row.get("unit_ids")
        if not isinstance(mapped, list) or any(str(value) not in unit_ids for value in mapped):
            raise ValueError("semantic plan action record references an unknown unit")
    for index, row in enumerate(normalized_units):
        if row.get("unit_index") != index or row.get("unit") != units[index]:
            raise ValueError("semantic plan unit record is not bound to the immutable unit")
        owners = row.get("owner_action_ids")
        if not isinstance(owners, list) or any(str(value) not in action_ids for value in owners):
            raise ValueError("semantic plan unit record references an unknown action")

    canonical_document = json.loads(_canonical_json(document))
    canonical_context = json.loads(_canonical_json(review_context))
    hash_input = {
        "document": canonical_document,
        "action_records": normalized_actions,
        "unit_records": normalized_units,
        "review_context": canonical_context,
    }
    plan_hash = _content_hash(hash_input)
    request = {
        "plan_hash": plan_hash,
        "immutable_document": canonical_document,
        "actions": normalized_actions,
        "semantic_units": normalized_units,
        "review_context": canonical_context,
    }
    return ImmutableSemanticPlan(
        plan_hash=plan_hash,
        document_json=_canonical_json(canonical_document),
        request_json=_canonical_json(request),
        action_ids=action_ids,
        unit_ids=unit_ids,
    )


def whole_plan_admission_prompt(semantic_policy: str) -> str:
    """Return the one independent immutable whole-plan admission contract."""

    return (
        "You are the independent admission authority for one immutable AnyChain typed intent plan. "
        "You are not a planner. Never create, repair, replace, rename, remove, merge, split, or reorder an action or semantic unit. "
        "Judge only the supplied immutable ids, actions, registry purposes, source units, pending contract, and workflow state. "
        "The supplied review_context.original_request clauses are the authoritative complete user turn. Compare them with the immutable semantic units before admitting coverage; a planner unit cannot hide a sibling demand merely by spanning the same prose. "
        "Return exactly one JSON object with exactly these keys: plan_hash, action_verdicts, unit_verdicts, reason. "
        "Echo plan_hash exactly. Return exactly one action_verdict for every supplied action_id and exactly one unit_verdict for every supplied unit_id; never add an id. "
        "Each action_verdict is {action_id,verdict:'admit'|'reject',unit_ids:[string],evidence:[{unit_id,quote,relation:'direct'|'support',support_relation:string}],grounded_arguments:[{argument_name:string,evidence_quote:string}],pending_answer_argument:string,turn_candidate_verdicts:[{candidate_id:string,verdict:'selected'|'not_selected',evidence_quote:string,reason:string}],reason}. "
        "A pending option may be selected by its number, id, canonical value, label, or a clear natural-language semantic equivalent. Do not require the source to repeat an option number or full label when it directly names the declared value or meaning. "
        "unit_ids must exactly equal that action's supplied immutable unit_ids. An admitted action needs one evidence row for every unit_id, every quote must be a non-empty exact substring of that unit, at least one relation must be direct, and a support row may use only one supplied allowed_support_relation. Direct rows use an empty support_relation. "
        "grounded_arguments must contain exactly one row for every supplied required_value_grounding_argument and no other row. argument_name is the exact required_value_grounding_argument name copied verbatim, never an explanation or value. Its evidence_quote must be a non-empty exact substring of one owned source unit that semantically selects the exact immutable operation_arguments value. Merely naming the argument or dimension, asking to change it without selecting a value, stating a generic benchmark goal, or relying on workflow state does not ground a concrete value. Natural-language equivalents may ground a value only when they unambiguously select that exact value. Actions with no required value-grounding arguments return an empty list. "
        "pending_answer_argument is an opaque manual-candidate id, never an answer value, option id, option label, number, or paraphrase. Declared options are reviewed through the immutable action and pending_choice_contracts; they do not use pending_answer_argument. If an action record supplies an empty pending_value_candidates list, pending_answer_argument must be exactly the empty string even when that action selects a declared option. Only when the supplied active pending question allows manual input and exactly one supplied pending_value_candidate semantically answers that question, set pending_answer_argument to that candidate's exact candidate_id copied verbatim. Each action record also supplies turn_pending_value_candidates scoped to source units owned by that action. For every action with non-empty pending_value_candidates return exactly one turn_candidate_verdict for every supplied turn candidate in supplied order; actions with no pending_value_candidates return an empty list. Each evidence_quote must be a non-empty exact substring of one candidate source unit. For a parser-derived literal it contains the literal; for a semantically normalized number, map, or enum it must be the exact phrase that selects that canonical value. Each reason must explain the source role rather than repeat the verdict. Exactly one row may be selected, and it must have the same contract-owned identity as the immutable operation value selected by pending_answer_argument; every other row is not_selected. Evaluate the selected operation against the complete scoped set, not only its operation argument. When two or more candidates exist, admit one only when the source explicitly distinguishes it as selected and distinguishes every other value as rejected, old, example-only, or otherwise not selected. A comparison, conjunction, disjunction, slash-separated list, or bare sequence is unresolved and must reject the pending answer rather than arbitrarily labeling one selected. A syntax-compatible value for an unrelated interruption is not an answer. A candidate mentioned only as an example, quotation, rejected option, negated operation, correction target, or value the user says not to apply is not an answer. Never invent a candidate id or rewrite the action. "
        "Each unit_verdict is {unit_id,verdict:'complete'|'support'|'context'|'unresolved'|'omitted',owner_action_ids:[string],evidence_quote:string,omitted_action_type:string,reason}. "
        "owner_action_ids must exactly equal the supplied immutable owner_action_ids. complete is valid only when the unit directly expresses a present demand preserved by every registered owner action. support is valid only when the unit does not independently request another action, every owner action cites it with relation=support and a supplied allowed_support_relation, and each owner action has direct evidence in another unit. A supplied immutable disposition=context has no owner and must receive verdict=context when it contains no independent omitted demand, or verdict=omitted when it does; never reinterpret it as complete or support and never invent an owner. context is invalid for any other supplied disposition. unresolved means the request is genuinely unsafe or not expressible. omitted means the unit contains a present independently actionable demand expressible by one action_schema type but missing from the immutable actions; set omitted_action_type to that exact registered type. Never propose its arguments or a replacement action. For every other verdict omitted_action_type is empty. Every evidence_quote is a non-empty exact substring of that unit. "
        "The action evidence and unit verdict are one consistency contract, not independent guesses. For each owned unit: if every owner action cites that unit as direct, its unit verdict is complete; if every owner action cites it as support, its unit verdict is support. Never return complete for a support-cited unit or support for a direct-cited unit. "
        "A mapped action may be individually plausible while its unit still has an omitted demand. Questions, corrections, navigation, configuration, evidence analysis, execution approval, and pending answers are all present demands when explicitly requested. Workflow state and planner reasons are context, never user evidence. "
        "A pending option or manual answer is admitted only when current source evidence satisfies the supplied typed pending contract. "
        + PENDING_CANDIDATE_SEMANTIC_POLICY
        + "A registered group or field owner is admitted only when its declared purpose and exact target preserve the source demand. Structured syntax facts are authoritative only after the immutable compiler assigned that structured unit to the corresponding registered owner, or after a clarification-only compiler unit exposed the same exact atomic clause as a typed candidate for this independent review; examples, negations, consultations, or logs do not become configuration merely because they contain assignments. "
        "A pending-answer owner covers only the answer to the question that existed at turn start. If its owned unit also states any independently actionable value, evidence, mutation, consultation, or navigation for a later step, the unit is complete only when the immutable plan includes every corresponding registered owner action; otherwise return omitted for that unit. Never treat creation of a later typed question as preservation of an explicit value already present in the current source. "
        "Malformed ids, missing rows, duplicate rows, invented quotes, an unsupported support relation, or uncertainty must fail closed. "
        + semantic_policy
    )


def request_whole_plan_admission(
    provider: Any,
    plan: ImmutableSemanticPlan,
    *,
    semantic_policy: str,
    allowed_action_types: Collection[str],
    max_tokens: int = 7200,
    contract_repair: bool = False,
) -> WholePlanAdmission:
    """Run one immutable review with one bounded structural-contract repair."""

    base_prompt = whole_plan_admission_prompt(semantic_policy)
    base_payload = plan.request_payload()
    previous_output = ""
    previous_errors: tuple[str, ...] = ()
    request_sizes: list[int] = []
    admission = WholePlanAdmission(False, ("whole-plan admission was not run",))
    for attempt in range(2 if contract_repair else 1):
        ensure_turn_active()
        prompt = base_prompt
        payload = dict(base_payload)
        if attempt:
            payload["admission_contract_repair"] = {
                "prior_invalid_output": previous_output,
                "validation_errors": list(previous_errors),
                "instruction": (
                    "Return a complete replacement admission document for the "
                    "same immutable plan. Preserve semantic verdicts and correct "
                    "every structural receipt or schema error."
                ),
            }
            prompt = (
                f"{base_prompt} This is a contract-repair attempt. The prior "
                f"admission document was structurally rejected for: "
                f"{'; '.join(previous_errors)}. Return every required row and "
                "opaque id exactly; do not change an explicit semantic rejection "
                "merely to pass validation."
            )
        payload_text = _canonical_json(payload)
        request_sizes.append(
            len(prompt.encode("utf-8")) + len(payload_text.encode("utf-8"))
        )
        response = provider.complete(LLMRequest(
            messages=[
                LLMMessage(role="system", content=prompt),
                LLMMessage(role="user", content=payload_text),
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        ))
        previous_output = str(response.text or "")
        admission = validate_whole_plan_admission(
            previous_output,
            plan,
            allowed_action_types=allowed_action_types,
        )
        if admission.valid or _is_explicit_semantic_rejection(admission):
            break
        previous_errors = admission.errors
    return replace(
        admission,
        request_count=len(request_sizes),
        request_sizes=tuple(request_sizes),
    )


def _is_explicit_semantic_rejection(
    admission: WholePlanAdmission,
) -> bool:
    """Do not retry a valid reviewer decision to reject immutable semantics."""

    response = admission.response
    if not isinstance(response, Mapping):
        return False
    action_rows = response.get("action_verdicts")
    unit_rows = response.get("unit_verdicts")
    return bool(
        isinstance(action_rows, list)
        and any(
            isinstance(row, Mapping)
            and str(row.get("verdict") or "") == "reject"
            for row in action_rows
        )
    ) or bool(
        isinstance(unit_rows, list)
        and any(
            isinstance(row, Mapping)
            and str(row.get("verdict") or "") in {"omitted", "unresolved"}
            for row in unit_rows
        )
    )


def validate_whole_plan_admission(
    response_text: str,
    plan: ImmutableSemanticPlan,
    *,
    allowed_action_types: Collection[str],
) -> WholePlanAdmission:
    """Validate exact reviewer cardinality, ids, grounding, and immutability."""

    try:
        payload = _strict_json_object(response_text)
    except ValueError as exc:
        return WholePlanAdmission(False, (str(exc),))

    errors: list[str] = []
    if set(payload) != _ADMISSION_TOP_LEVEL_KEYS:
        errors.append("whole-plan admission has missing or undeclared top-level keys")
    if str(payload.get("plan_hash") or "") != plan.plan_hash:
        errors.append("whole-plan admission plan_hash mismatch")
    if not str(payload.get("reason") or "").strip():
        errors.append("whole-plan admission has no reason")

    request = plan.request_payload()
    action_records = {
        str(row["action_id"]): row for row in request.get("actions") or []
        if isinstance(row, dict) and str(row.get("action_id") or "")
    }
    unit_records = {
        str(row["unit_id"]): row for row in request.get("semantic_units") or []
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }
    payload = _canonicalize_admission_receipts(
        payload,
        action_records=action_records,
        unit_records=unit_records,
    )
    raw_action_rows = payload.get("action_verdicts")
    raw_unit_rows = payload.get("unit_verdicts")
    action_rows = list(raw_action_rows) if isinstance(raw_action_rows, list) else []
    unit_rows = list(raw_unit_rows) if isinstance(raw_unit_rows, list) else []
    if not isinstance(raw_action_rows, list):
        errors.append("whole-plan admission action_verdicts is not a list")
    if not isinstance(raw_unit_rows, list):
        errors.append("whole-plan admission unit_verdicts is not a list")

    action_counts: dict[str, int] = {}
    valid_action_rows: list[dict[str, Any]] = []
    admitted_evidence_relations: dict[tuple[str, str], str] = {}
    for raw in action_rows:
        if not isinstance(raw, dict):
            errors.append("whole-plan admission contains a non-object action verdict")
            continue
        row = dict(raw)
        if set(row) != _ACTION_VERDICT_KEYS:
            errors.append("whole-plan action verdict has missing or undeclared keys")
        action_id = str(row.get("action_id") or "")
        action_counts[action_id] = action_counts.get(action_id, 0) + 1
        record = action_records.get(action_id)
        if record is None:
            errors.append(f"whole-plan admission forged action id: {action_id or '<missing>'}")
            continue
        expected_units = [str(value) for value in record.get("unit_ids") or []]
        actual_units = [str(value) for value in row.get("unit_ids") or []] if isinstance(row.get("unit_ids"), list) else []
        if actual_units != expected_units:
            errors.append(f"whole-plan action unit ownership mismatch: {action_id}")
        verdict = str(row.get("verdict") or "")
        if verdict not in {"admit", "reject"}:
            errors.append(f"whole-plan action verdict is invalid: {action_id}")
        if not str(row.get("reason") or "").strip():
            errors.append(f"whole-plan action verdict has no reason: {action_id}")
        evidence = row.get("evidence")
        evidence_rows = list(evidence) if isinstance(evidence, list) else []
        if not isinstance(evidence, list):
            errors.append(f"whole-plan action evidence is not a list: {action_id}")
        grounding = row.get("grounded_arguments")
        grounding_rows = list(grounding) if isinstance(grounding, list) else []
        if not isinstance(grounding, list):
            errors.append(f"whole-plan grounded_arguments is not a list: {action_id}")
        expected_grounding = [
            str(value)
            for value in record.get("required_value_grounding_arguments") or []
        ]
        exact_grounding = {
            str(value)
            for value in record.get("exact_source_value_arguments") or []
        }
        operation_arguments = (
            record.get("operation_arguments")
            if isinstance(record.get("operation_arguments"), Mapping)
            else {}
        )
        grounding_counts: dict[str, int] = {}
        owned_sources = [
            str((unit_records.get(unit_id) or {}).get("source_text") or "")
            for unit_id in expected_units
        ]
        for raw_grounding in grounding_rows:
            if not isinstance(raw_grounding, dict):
                errors.append(f"whole-plan grounded argument is not an object: {action_id}")
                continue
            grounding_row = dict(raw_grounding)
            if set(grounding_row) != _GROUNDED_ARGUMENT_KEYS:
                errors.append(f"whole-plan grounded argument has missing or undeclared keys: {action_id}")
            argument = str(grounding_row.get("argument_name") or "")
            grounding_counts[argument] = grounding_counts.get(argument, 0) + 1
            quote = str(grounding_row.get("evidence_quote") or "")
            if argument not in expected_grounding:
                errors.append(f"whole-plan grounded argument is undeclared: {action_id}/{argument or '<missing>'}")
            if not quote or not any(quote in source for source in owned_sources):
                errors.append(f"whole-plan grounded argument evidence is not exact: {action_id}/{argument or '<missing>'}")
            exact_value = str(operation_arguments.get(argument) or "").strip()
            if argument in exact_grounding and exact_value and exact_value not in quote:
                errors.append(
                    f"whole-plan exact grounded argument quote does not contain its immutable value: "
                    f"{action_id}/{argument}"
                )
        if [str(item.get("argument_name") or "") for item in grounding_rows if isinstance(item, dict)] != expected_grounding:
            errors.append(f"whole-plan grounded argument order or cardinality mismatch: {action_id}")
        if any(grounding_counts.get(argument, 0) != 1 for argument in expected_grounding):
            errors.append(f"whole-plan required argument is not grounded exactly once: {action_id}")
        pending_argument = str(row.get("pending_answer_argument") or "")
        pending_candidates = [
            str(value.get("candidate_id") or "")
            for value in record.get("pending_value_candidates") or []
            if isinstance(value, Mapping)
        ]
        if pending_argument and pending_argument not in pending_candidates:
            errors.append(f"whole-plan pending answer argument is undeclared: {action_id}/{pending_argument}")
        immutable_action = record.get("action") if isinstance(record.get("action"), Mapping) else {}
        if (
            verdict == "admit"
            and str(immutable_action.get("type") or "") == "answer_pending"
            and pending_candidates
            and (
                len(pending_candidates) != 1
                or pending_argument != pending_candidates[0]
            )
        ):
            errors.append(
                f"whole-plan admitted pending answer lacks one explicit typed candidate selection: {action_id}"
            )
        raw_turn_verdicts = row.get("turn_candidate_verdicts")
        turn_verdicts = (
            list(raw_turn_verdicts)
            if isinstance(raw_turn_verdicts, list)
            else []
        )
        if not isinstance(raw_turn_verdicts, list):
            errors.append(
                f"whole-plan turn_candidate_verdicts is not a list: {action_id}"
            )
        turn_candidates = [
            dict(value)
            for value in record.get("turn_pending_value_candidates") or []
            if isinstance(value, Mapping)
            and str(value.get("candidate_id") or "")
        ]
        expected_turn_ids = [
            str(value["candidate_id"])
            for value in turn_candidates
        ]
        requires_turn_verdicts = bool(pending_candidates and turn_candidates)
        review_context = plan.request_payload().get("review_context")
        pending_question = (
            review_context.get("pending_question")
            if isinstance(review_context, Mapping)
            and isinstance(review_context.get("pending_question"), Mapping)
            else {}
        )
        actual_turn_ids: list[str] = []
        selected_turn_identities: list[str] = []
        for raw_turn_verdict in turn_verdicts:
            if not isinstance(raw_turn_verdict, dict):
                errors.append(
                    f"whole-plan turn candidate verdict is not an object: {action_id}"
                )
                continue
            turn_row = dict(raw_turn_verdict)
            if set(turn_row) != _TURN_CANDIDATE_VERDICT_KEYS:
                errors.append(
                    f"whole-plan turn candidate verdict has missing or undeclared keys: {action_id}"
                )
            candidate_id = str(turn_row.get("candidate_id") or "")
            actual_turn_ids.append(candidate_id)
            candidate = next(
                (
                    value
                    for value in turn_candidates
                    if str(value["candidate_id"]) == candidate_id
                ),
                None,
            )
            candidate_verdict = str(turn_row.get("verdict") or "")
            if candidate_verdict not in {"selected", "not_selected"}:
                errors.append(
                    f"whole-plan turn candidate verdict is invalid: {action_id}/{candidate_id}"
                )
            evidence_quote = str(turn_row.get("evidence_quote") or "")
            candidate_value = candidate.get("value") if candidate is not None else None
            candidate_source_units = (
                [
                    str(value)
                    for value in candidate.get("source_unit_ids") or []
                ]
                if candidate is not None
                else []
            )
            candidate_source_texts = [
                str(
                    (unit_records.get(unit_id) or {}).get("source_text")
                    or ""
                )
                for unit_id in candidate_source_units
            ]
            if (
                candidate is None
                or not evidence_quote
                or not candidate_source_units
                or any(unit_id not in expected_units for unit_id in candidate_source_units)
                or not any(
                    evidence_quote
                    in str(
                        (unit_records.get(unit_id) or {}).get("source_text")
                        or ""
                    )
                    for unit_id in candidate_source_units
                )
                or not _quote_supports_pending_candidate(
                    evidence_quote,
                    candidate,
                    pending_question,
                    candidate_source_texts,
                )
            ):
                errors.append(
                    f"whole-plan turn candidate evidence is not exact: {action_id}/{candidate_id}"
                )
            if not str(turn_row.get("reason") or "").strip():
                errors.append(
                    f"whole-plan turn candidate verdict has no reason: {action_id}/{candidate_id}"
                )
            if candidate_verdict == "selected" and candidate is not None:
                selected_turn_identities.append(str(candidate.get("identity") or ""))
        if requires_turn_verdicts:
            if actual_turn_ids != expected_turn_ids:
                errors.append(
                    f"whole-plan turn candidate order or cardinality mismatch: {action_id}"
                )
            selected_operation_identity = next(
                (
                    str(value.get("identity") or "")
                    for value in record.get("pending_value_candidates") or []
                    if isinstance(value, Mapping)
                    and str(value.get("candidate_id") or "") == pending_argument
                ),
                None,
            )
            if (
                verdict == "admit"
                and (
                    len(selected_turn_identities) != 1
                    or selected_turn_identities[0] != selected_operation_identity
                )
            ):
                errors.append(
                    f"whole-plan admitted pending answer lacks one matching turn candidate: {action_id}"
                )
        elif turn_verdicts:
            errors.append(
                f"whole-plan non-pending action declares turn candidate verdicts: {action_id}"
            )
        seen_evidence: set[str] = set()
        direct_count = 0
        allowed_support = set(record.get("allowed_support_relations") or [])
        for raw_evidence in evidence_rows:
            if not isinstance(raw_evidence, dict):
                errors.append(f"whole-plan action evidence is not an object: {action_id}")
                continue
            evidence_row = dict(raw_evidence)
            if set(evidence_row) != _ACTION_EVIDENCE_KEYS:
                errors.append(f"whole-plan action evidence has missing or undeclared keys: {action_id}")
            unit_id = str(evidence_row.get("unit_id") or "")
            if unit_id in seen_evidence:
                errors.append(f"whole-plan action evidence duplicates unit id: {action_id}/{unit_id}")
            seen_evidence.add(unit_id)
            if unit_id not in expected_units:
                errors.append(f"whole-plan action evidence forged unit id: {action_id}/{unit_id}")
                continue
            quote = str(evidence_row.get("quote") or "")
            source = str((unit_records.get(unit_id) or {}).get("source_text") or "")
            if not quote or quote not in source:
                errors.append(f"whole-plan action evidence is not exact: {action_id}/{unit_id}")
            relation = str(evidence_row.get("relation") or "")
            support_relation = str(evidence_row.get("support_relation") or "")
            if relation == "direct":
                direct_count += 1
                if support_relation:
                    errors.append(f"whole-plan direct evidence declares support relation: {action_id}/{unit_id}")
            elif relation == "support":
                if support_relation not in allowed_support:
                    errors.append(f"whole-plan support relation is not registered: {action_id}/{unit_id}")
            else:
                errors.append(f"whole-plan action evidence relation is invalid: {action_id}/{unit_id}")
        if verdict == "admit":
            if set(seen_evidence) != set(expected_units) or len(evidence_rows) != len(expected_units):
                errors.append(f"whole-plan admitted action lacks exact evidence cardinality: {action_id}")
            if direct_count < 1:
                errors.append(f"whole-plan admitted action has no direct evidence: {action_id}")
            for evidence_row in evidence_rows:
                if not isinstance(evidence_row, dict):
                    continue
                unit_id = str(evidence_row.get("unit_id") or "")
                relation = str(evidence_row.get("relation") or "")
                if unit_id in expected_units and relation in {"direct", "support"}:
                    admitted_evidence_relations[(action_id, unit_id)] = relation
        valid_action_rows.append(row)

    for action_id in plan.action_ids:
        count = action_counts.get(action_id, 0)
        if count != 1:
            errors.append(f"whole-plan admission requires one action verdict: {action_id} ({count})")
    returned_action_ids = tuple(
        str(row.get("action_id") or "")
        for row in action_rows
        if isinstance(row, dict)
    )
    if returned_action_ids != plan.action_ids:
        errors.append("whole-plan action verdict order does not match the immutable plan")

    unit_counts: dict[str, int] = {}
    valid_unit_rows: list[dict[str, Any]] = []
    allowed_types = {str(value) for value in allowed_action_types}
    for raw in unit_rows:
        if not isinstance(raw, dict):
            errors.append("whole-plan admission contains a non-object unit verdict")
            continue
        row = dict(raw)
        if set(row) != _UNIT_VERDICT_KEYS:
            errors.append("whole-plan unit verdict has missing or undeclared keys")
        unit_id = str(row.get("unit_id") or "")
        unit_counts[unit_id] = unit_counts.get(unit_id, 0) + 1
        record = unit_records.get(unit_id)
        if record is None:
            errors.append(f"whole-plan admission forged unit id: {unit_id or '<missing>'}")
            continue
        expected_owners = [str(value) for value in record.get("owner_action_ids") or []]
        actual_owners = [str(value) for value in row.get("owner_action_ids") or []] if isinstance(row.get("owner_action_ids"), list) else []
        if actual_owners != expected_owners:
            errors.append(f"whole-plan unit owner mismatch: {unit_id}")
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "support", "context", "unresolved", "omitted"}:
            errors.append(f"whole-plan unit verdict is invalid: {unit_id}")
        quote = str(row.get("evidence_quote") or "")
        source = str(record.get("source_text") or "")
        if not quote or quote not in source:
            errors.append(f"whole-plan unit evidence is not exact: {unit_id}")
        if not str(row.get("reason") or "").strip():
            errors.append(f"whole-plan unit verdict has no reason: {unit_id}")
        omitted_type = str(row.get("omitted_action_type") or "")
        if verdict == "omitted":
            if omitted_type not in allowed_types:
                errors.append(f"whole-plan omitted demand is not registry expressible: {unit_id}")
        elif omitted_type:
            errors.append(f"whole-plan non-omitted unit declares an omitted action: {unit_id}")
        disposition = str(record.get("disposition") or "")
        if disposition == "action" and verdict not in {"complete", "support"}:
            errors.append(f"whole-plan action unit is neither complete nor support: {unit_id}")
        elif disposition == "context" and verdict != "context":
            errors.append(f"whole-plan context unit is not context-only: {unit_id}")
        elif disposition == "unresolved":
            errors.append(f"whole-plan candidate retains unresolved unit: {unit_id}")
        if verdict == "complete" and not expected_owners:
            errors.append(f"whole-plan complete unit has no owner: {unit_id}")
        if verdict == "complete" and any(
            admitted_evidence_relations.get((owner, unit_id)) != "direct"
            for owner in expected_owners
        ):
            errors.append(f"whole-plan complete unit is not direct for every owner: {unit_id}")
        if verdict == "support" and not expected_owners:
            errors.append(f"whole-plan support unit has no owner: {unit_id}")
        if verdict == "support" and any(
            admitted_evidence_relations.get((owner, unit_id)) != "support"
            for owner in expected_owners
        ):
            errors.append(f"whole-plan support unit is not support for every owner: {unit_id}")
        if verdict == "context" and expected_owners:
            errors.append(f"whole-plan context unit has an owner: {unit_id}")
        valid_unit_rows.append(row)

    for unit_id in plan.unit_ids:
        count = unit_counts.get(unit_id, 0)
        if count != 1:
            errors.append(f"whole-plan admission requires one unit verdict: {unit_id} ({count})")
    returned_unit_ids = tuple(
        str(row.get("unit_id") or "")
        for row in unit_rows
        if isinstance(row, dict)
    )
    if returned_unit_ids != plan.unit_ids:
        errors.append("whole-plan unit verdict order does not match the immutable plan")

    if any(str(row.get("verdict") or "") != "admit" for row in valid_action_rows):
        errors.append("whole-plan admission rejected one or more immutable actions")
    if any(str(row.get("verdict") or "") not in {"complete", "support", "context"} for row in valid_unit_rows):
        errors.append("whole-plan admission found unresolved or omitted demand")

    return WholePlanAdmission(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        action_verdicts=tuple(valid_action_rows),
        unit_verdicts=tuple(valid_unit_rows),
        response=payload,
    )


def _canonicalize_admission_receipts(
    payload: dict[str, Any],
    *,
    action_records: Mapping[str, Mapping[str, Any]],
    unit_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize uniquely derivable reviewer wire receipts.

    The reviewer remains the semantic admission authority. This boundary only
    repairs redundant receipt fields after an explicit ``admit`` verdict when
    the immutable plan and exact direct evidence leave no choice. Rejections,
    missing rows, ambiguous candidates, and exact-source values remain
    untouched so the strict validator can fail closed.
    """

    action_rows = payload.get("action_verdicts")
    if not isinstance(action_rows, list):
        return payload
    for raw_row in action_rows:
        if not isinstance(raw_row, dict) or str(raw_row.get("verdict") or "") != "admit":
            continue
        action_id = str(raw_row.get("action_id") or "")
        record = action_records.get(action_id)
        if not isinstance(record, Mapping):
            continue
        immutable_action = (
            record.get("action")
            if isinstance(record.get("action"), Mapping)
            else {}
        )
        evidence_rows = [
            item
            for item in raw_row.get("evidence") or []
            if isinstance(item, Mapping)
            and str(item.get("relation") or "") == "direct"
        ]
        exact_direct_quotes = []
        for evidence in evidence_rows:
            unit_id = str(evidence.get("unit_id") or "")
            source = str((unit_records.get(unit_id) or {}).get("source_text") or "")
            quote = str(evidence.get("quote") or "")
            if quote and quote in source and quote not in exact_direct_quotes:
                exact_direct_quotes.append(quote)
        if len(exact_direct_quotes) != 1:
            continue
        owned_sources = [
            str((unit_records.get(str(unit_id)) or {}).get("source_text") or "")
            for unit_id in record.get("unit_ids") or []
        ]
        pending_candidates = [
            dict(item)
            for item in record.get("pending_value_candidates") or []
            if isinstance(item, Mapping)
        ]
        turn_candidates = [
            dict(item)
            for item in record.get("turn_pending_value_candidates") or []
            if isinstance(item, Mapping)
        ]
        if len(pending_candidates) == 1 and len(turn_candidates) == 1:
            operation_candidate = pending_candidates[0]
            turn_candidate = turn_candidates[0]
            operation_identity = str(operation_candidate.get("identity") or "")
            candidate_sources = [
                str(
                    (unit_records.get(str(unit_id)) or {}).get("source_text")
                    or ""
                )
                for unit_id in turn_candidate.get("source_unit_ids") or []
            ]
            supporting_quote = next(
                (
                    str(evidence.get("quote") or "")
                    for evidence in evidence_rows
                    if str(evidence.get("unit_id") or "")
                    in {
                        str(unit_id)
                        for unit_id in turn_candidate.get("source_unit_ids") or []
                    }
                    and _quote_supports_pending_candidate(
                        str(evidence.get("quote") or ""),
                        turn_candidate,
                        {},
                        candidate_sources,
                    )
                ),
                "",
            )
            if (
                operation_identity
                and operation_identity
                == str(turn_candidate.get("identity") or "")
                and supporting_quote
            ):
                raw_row["pending_answer_argument"] = str(
                    operation_candidate.get("candidate_id") or ""
                )
                raw_row["turn_candidate_verdicts"] = [{
                    "candidate_id": str(
                        turn_candidate.get("candidate_id") or ""
                    ),
                    "verdict": "selected",
                    "evidence_quote": supporting_quote,
                    "reason": (
                        "The sole typed turn candidate matches the sole "
                        "immutable operation candidate admitted by exact "
                        "direct evidence."
                    ),
                }]
        exact_arguments = {
            str(value)
            for value in record.get("exact_source_value_arguments") or []
        }
        for grounding in raw_row.get("grounded_arguments") or []:
            if not isinstance(grounding, dict):
                continue
            argument = str(grounding.get("argument_name") or "")
            quote = str(grounding.get("evidence_quote") or "")
            if (
                argument
                and argument not in exact_arguments
                and quote
                and not any(quote in source for source in owned_sources)
            ):
                grounding["evidence_quote"] = exact_direct_quotes[0]
    return payload
