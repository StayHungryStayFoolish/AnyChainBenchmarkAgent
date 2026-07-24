"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..input_values import (
    looks_like_url_value,
    normalize_scalar,
)
from ..localization import localized
from ..routing import next_group_and_reason
from ..state import AgentGraphState
from ..transitions import record_group_invalidations

from agent.knowledge.framework_capabilities import load_framework_capabilities
from agent.planners import question_prompts
from agent.validators.rpc_workload import default_workload


def _adapter_family(state: AgentGraphState) -> str:
    identity = state.get("chain_identity") or {}
    family = normalize_scalar(identity.get("adapter_family")).casefold()
    if family:
        return family
    chain = normalize_scalar(identity.get("canonical")).casefold()
    for row in load_framework_capabilities().get("chains", []):
        if normalize_scalar(row.get("chain")).casefold() == chain:
            return normalize_scalar(row.get("family") or row.get("adapter_family")).casefold()
    return ""


def _case_dict(state: AgentGraphState, case: str) -> dict[str, Any]:
    return state.setdefault("chain_identity", {}) if case == "new_chain" else state.setdefault("custom_rpc", {})


def is_existing_family_lifecycle(identity: Any) -> bool:
    """Return whether chain identity is inside the authoritative Case 2 lifecycle."""

    if not isinstance(identity, dict):
        return False
    return normalize_scalar(identity.get("status")).startswith("existing_family_")


def _jsonrpc_draft(method: str, params: Any) -> dict[str, Any]:
    return {
        "status": "draft",
        "evidence_kind": "jsonrpc_request",
        "transport": "jsonrpc",
        "method": normalize_scalar(method),
        "params": _parameter_descriptors(params),
        "params_json": params,
        "response_summary": "",
        "confidence": "high",
    }


def _json_wire_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _parameter_descriptors(params: Any) -> list[dict[str, Any]]:
    """Describe observed wire values without inventing blockchain semantics."""

    if isinstance(params, list):
        return [
            {
                "index": index,
                "name": f"param[{index}]",
                "json_type": _json_wire_type(value),
                "semantic_type": "unknown",
                "encoding": "unknown",
                "meaning": "unknown",
                "example": value,
                "required": "unknown",
            }
            for index, value in enumerate(params)
        ]
    if isinstance(params, dict):
        return [
            {
                "index": index,
                "name": str(name),
                "json_type": _json_wire_type(value),
                "semantic_type": "unknown",
                "encoding": "unknown",
                "meaning": "unknown",
                "example": value,
                "required": "unknown",
            }
            for index, (name, value) in enumerate(params.items())
        ]
    return []


def _normalize_parameter_descriptors(params: Any, params_json: Any) -> list[dict[str, Any]]:
    observed = _parameter_descriptors(params_json)
    if not isinstance(params, list) or not all(isinstance(item, dict) for item in params):
        return observed
    if not observed:
        return []
    extracted_by_name = {
        normalize_scalar(item.get("name")): item
        for item in params
        if normalize_scalar(item.get("name"))
    }
    output: list[dict[str, Any]] = []
    for index, wire in enumerate(observed):
        candidate = extracted_by_name.get(normalize_scalar(wire.get("name")))
        if candidate is None and isinstance(params_json, list) and index < len(params):
            candidate = params[index]
        row = dict(candidate or {})
        # Observed request facts are authoritative. The model may enrich only
        # semantic metadata; it must not change the actual wire contract.
        row.update({
            "index": index,
            "json_type": wire["json_type"],
            "example": wire["example"],
        })
        # Object keys are part of the wire contract. Positional parameter
        # names are not transmitted, so a model/document-derived semantic
        # name may be retained while the index/type/value stay authoritative.
        row["name"] = (
            wire["name"]
            if isinstance(params_json, dict)
            else normalize_scalar(row.get("name")) or wire["name"]
        )
        row["semantic_type"] = normalize_scalar(row.get("semantic_type")) or "unknown"
        row["encoding"] = normalize_scalar(row.get("encoding")) or "unknown"
        row["meaning"] = normalize_scalar(row.get("meaning")) or "unknown"
        row.setdefault("required", "unknown")
        output.append(row)
    return output


def _merge_rpc_schema_draft(
    *,
    method: str,
    params_json: Any,
    extracted: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge evidence while keeping observed request fields authoritative."""

    previous = previous or {}
    deterministic = _jsonrpc_draft(method, params_json)
    draft = {**previous, **deterministic, **extracted}
    draft.update({
        "status": "draft",
        "method": method,
        "params_json": params_json,
        "transport": "jsonrpc",
    })
    draft["params"] = _normalize_parameter_descriptors(draft.get("params"), params_json)
    if not draft.get("response_summary"):
        draft["response_summary"] = previous.get("response_summary") or "unknown"
    if not isinstance(draft.get("response_fields"), list):
        draft["response_fields"] = list(previous.get("response_fields") or [])
    return draft


def _params_from_draft(draft: dict[str, Any]) -> Any:
    if isinstance(draft.get("params_json"), (list, dict)):
        return draft["params_json"]
    params = draft.get("params")
    if not isinstance(params, list):
        return []
    return [item.get("example") if isinstance(item, dict) and "example" in item else item for item in params]


def _draft_has_params(draft: dict[str, Any]) -> bool:
    if not isinstance(draft, dict) or draft.get("status") == "failed":
        return False
    return isinstance(draft.get("params_json"), (list, dict)) or isinstance(draft.get("params"), list)


def _schema_indicates_rest(draft: dict[str, Any]) -> bool:
    transport = normalize_scalar(draft.get("transport") or draft.get("protocol")).casefold()
    method = normalize_scalar(draft.get("method")).upper()
    return bool(
        transport in {"rest", "http"}
        or normalize_scalar(draft.get("rest_path"))
        or looks_like_url_value(draft.get("endpoint_url"))
        or method.startswith(("GET ", "POST ", "PUT ", "PATCH ", "DELETE "))
        or normalize_scalar(draft.get("evidence_kind")).casefold() in {"rest_endpoint", "rest_path"}
    )


def _schema_indicates_jsonrpc(draft: dict[str, Any]) -> bool:
    transport = normalize_scalar(draft.get("transport") or draft.get("protocol")).casefold()
    method = normalize_scalar(draft.get("method"))
    return transport == "jsonrpc" or bool(re.match(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+$", method))


def _looks_like_rest_method(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        re.match(r"^(?:GET|POST|PUT|PATCH|DELETE)\s+/", text, re.IGNORECASE)
        or text.startswith("/")
        or any(token in text.casefold() for token in (" path parameter", "query parameter", "get blocks by", "post ", "rest api"))
    )


def _schema_confirmation_prompt(language: str, draft: dict[str, Any]) -> str:
    method = normalize_scalar(draft.get("method")) or "<unknown>"
    evidence_kind = normalize_scalar(draft.get("evidence_kind")) or "unknown"
    transport = normalize_scalar(draft.get("transport")) or "unknown"
    params = draft.get("params")
    if isinstance(params, list):
        lines = []
        for item in params:
            if isinstance(item, dict):
                name = item.get("name") or f"param[{item.get('index', '?')}]"
                lines.append(
                    f"- {name}: json_type={item.get('json_type') or item.get('type') or 'unknown'}, "
                    f"semantic_type={item.get('semantic_type') or 'unknown'}, encoding={item.get('encoding') or 'unknown'}, "
                    f"meaning={item.get('meaning') or 'unknown'}, required={item.get('required', 'unknown')}, "
                    f"example={json.dumps(item.get('example'), ensure_ascii=False)}"
                )
            else:
                lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
        params_text = "\n".join(lines) if lines else "- []"
    else:
        params_text = json.dumps(draft.get("params_json", []), ensure_ascii=False)
    response = normalize_scalar(draft.get("response_summary")) or "<unknown>"
    response_fields = draft.get("response_fields") if isinstance(draft.get("response_fields"), list) else []
    response_fields_text = "; ".join(
        f"{item.get('name') or '<unnamed>'}: type={item.get('type') or 'unknown'}, meaning={item.get('meaning') or 'unknown'}"
        for item in response_fields
        if isinstance(item, dict)
    ) or "<unknown>"
    confidence = normalize_scalar(draft.get("confidence")) or "unknown"
    conflicts = draft.get("conflicts") if isinstance(draft.get("conflicts"), list) else []
    conflicts_text = "; ".join(str(item) for item in conflicts) if conflicts else "<none>"
    evidence_summary = normalize_scalar(draft.get("evidence_summary"))
    search_result = draft.get("search_result") if isinstance(draft.get("search_result"), dict) else {}
    if evidence_summary and search_result.get("available") is True:
        evidence_zh = f"\ngoogle_search 补充：{evidence_summary}"
        evidence_en = f"\ngoogle_search evidence: {evidence_summary}"
    elif evidence_summary:
        evidence_zh = f"\n证据摘要：{evidence_summary}"
        evidence_en = f"\nEvidence summary: {evidence_summary}"
    else:
        evidence_zh = ""
        evidence_en = ""
    return localized(
        language,
        f"我从证据中提取到以下 RPC request contract，请确认 request wire facts 和已审阅的参数语义是否正确。\nevidence: {evidence_kind} / transport={transport}\nmethod: `{method}`\nparams:\n{params_text}\nresponse: {response}\nresponse fields: {response_fields_text}\nconfidence: {confidence}\nconflicts: {conflicts_text}{evidence_zh}\n注意：这里仅确认 request contract，不会执行 endpoint probe。回复 `Y` 确认 request；回复 `N` 补充或修正 request/docs 证据。",
        f"I extracted this RPC request contract from the evidence. Confirm the request wire facts and reviewed parameter semantics.\nevidence: {evidence_kind} / transport={transport}\nmethod: `{method}`\nparams:\n{params_text}\nresponse: {response}\nresponse fields: {response_fields_text}\nconfidence: {confidence}\nconflicts: {conflicts_text}{evidence_en}\nThis confirms only the request contract and does not run an endpoint probe. Reply `Y` to confirm the request, or `N` to add/correct request or docs evidence.",
    )


def _workload_default_prompt(state: AgentGraphState) -> str:
    language = str(state.get("language") or "en")
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    rpc_mode = normalize_scalar(state.get("rpc_mode"))
    defaults = default_workload(chain) if chain else {}
    single = normalize_scalar(defaults.get("single")) or "<none>"
    mixed = ", ".join(f"{row.get('method')}={row.get('weight')}" for row in defaults.get("mixed_weighted") or [] if isinstance(row, dict) and row.get("method")) or "<none>"
    summary = question_prompts.workload_defaults_summary(chain, rpc_mode, single, mixed, language=language)
    return localized(language, f"{summary}\n请选择使用默认 workload、添加自定义 RPC method、调整 mixed 权重，或更换链/目标模式。", f"{summary}\nChoose the default workload, add a custom RPC method, adjust mixed weights, or change chain/target mode.")


def _weight_example(methods: list[str]) -> str:
    unique = [method for method in dict.fromkeys(normalize_scalar(item) for item in methods) if method]
    if not unique:
        return "method=100"
    if len(unique) == 1:
        return f"{unique[0]}=100"
    share, remainder = divmod(100, len(unique))
    return ",".join(
        f"{method}={share + (1 if index < remainder else 0)}"
        for index, method in enumerate(unique)
    )


def _format_weights(weights: dict[str, int]) -> str:
    return ", ".join(f"{method}={weight}" for method, weight in weights.items())


def _chain_confirmed(state: AgentGraphState) -> bool:
    return (state.get("chain_identity") or {}).get("status") == "confirmed"


def _next_group(state: AgentGraphState) -> str:
    return next_group_and_reason(state)[0]


def _handoff_stops(state: AgentGraphState) -> bool:
    return (state.get("chain_identity") or {}).get("status") in {"unsupported_family_handoff", "needs_review_handoff"}


def _invalidate_execution(state: AgentGraphState, changed_group: str = "workload_rpc") -> None:
    record_group_invalidations(state, changed_group)


def _invalidate_groups(state: AgentGraphState, *groups: str) -> None:
    invalidated = set(state.get("invalidated_groups") or [])
    invalidated.update(group for group in groups if group)
    state["invalidated_groups"] = sorted(invalidated)


def _result(
    original: AgentGraphState,
    state: AgentGraphState,
    action: ActionProposal | None = None,
    *,
    completion: str = "completed",
    stop: bool = False,
) -> HandlerResult:
    _assert_external_state_unchanged(original, state)
    pending = deepcopy(state.get("pending_question") or {})
    previous_pending = deepcopy(original.get("pending_question") or {})
    previous_invalidated = set(original.get("invalidated_groups") or [])
    current_invalidated = set(state.get("invalidated_groups") or [])
    previous_responses = list(original.get("visible_response") or [])
    previous_errors = list(original.get("action_errors") or [])
    return HandlerResult(
        delta=StateDelta.between(original, state),
        consumed_action_ids=((action.action_id,) if action is not None else ()),
        invalidated_groups=tuple(sorted(current_invalidated - previous_invalidated)),
        reconfigured_groups=tuple(sorted(previous_invalidated - current_invalidated)),
        action_errors=tuple(
            deepcopy(item)
            for item in state.get("action_errors") or []
            if item not in previous_errors
        ),
        visible_results=tuple(
            str(item)
            for item in state.get("visible_response") or []
            if str(item) and item not in previous_responses
        ),
        pending_question=pending or None,
        clear_pending=bool(previous_pending and not pending),
        next_group=normalize_scalar(pending.get("group") or state.get("active_group")),
        completion=completion,  # type: ignore[arg-type]
        stop_after_response=stop,
    )


def _answer_result(
    original: AgentGraphState,
    state: AgentGraphState,
    *,
    completion: str = "completed",
    stop: bool = False,
) -> HandlerResult:
    return _result(
        original,
        state,
        completion=completion,
        stop=stop,
    )


def _assert_external_state_unchanged(original: AgentGraphState, state: AgentGraphState) -> None:
    for key in ("action_queue", "group_history"):
        if deepcopy(original.get(key)) != deepcopy(state.get(key)):
            raise RuntimeError(f"chain_rpc must not mutate coordinator-owned {key}")
