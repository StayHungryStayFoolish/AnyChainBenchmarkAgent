"""ADK callbacks for AnyChain benchmark safety boundaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge.chain_identity import FRAMEWORK_CONTEXT_TOKENS
from knowledge.chain_identity import canonical_chain_aliases
from knowledge.chain_identity import repo_chain_names
from adk_app.tools.workflow_state import propose_chain_identity_resolution
from adk_app.tools.workflow_state import request_unsupported_chain_handoff
from workflows.conversation_state import update_workflow_state as _update_workflow_state
from workflows.group_registry import normalize_group_name as _normalize_group_name

CONFIRMATION_GATED_TOOLS = {
    "run_fake_node_smoke_benchmark",
    "run_quick_assumed_fake_node_smoke",
    "install_dependencies",
    "submit_benchmark_job",
    "stop_job",
}

STATE_WRITE_TOOLS = {"update_workflow_state", "answer_pending_question"}
DOWNSTREAM_SETUP_QUESTION_TOOLS = {
    "propose_disk_device_choice",
    "propose_benchmark_profile_choice",
    "propose_workload_customization_choice",
}

def before_tool_callback(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """Block confirmation-gated benchmark actions until approval is explicit.

    ADK owns tool orchestration, but benchmark execution is still a high-impact
    domain action. This callback is an ADK-native guardrail: it lets the model
    see and select action tools while preventing accidental execution when the
    tool call does not include ``approved=true``.
    """
    _ = tool_context
    tool_name = _tool_name(tool)
    if tool_name in DOWNSTREAM_SETUP_QUESTION_TOOLS:
        blocked = _block_downstream_question_when_target_entities_are_unrecorded(tool_name, args, tool_context)
        if blocked:
            return blocked
    if tool_name == "answer_pending_question":
        state = _context_state(tool_context)
        user_text = str(state.get("user_text") or "").strip()
        answer = str(args.get("answer") or "").strip()
        input_mode = str(state.get("input_mode") or "")
        if input_mode != "pending_answer_applied" and answer.casefold() != user_text.casefold():
            return {
                "status": "blocked",
                "data": {
                    "action": tool_name,
                    "reason": "answer_pending_question must apply the current user answer only",
                },
                "evidence_paths": [],
                "warnings": [
                    "Do not answer a newly-created pending question on the user's behalf. Ask the typed pending question and wait for the next user turn."
                ],
                "next_actions": [
                    "show the typed pending question",
                    "wait for the user's explicit answer",
                ],
                "requires_user_confirmation": True,
            }
    if tool_name == "update_workflow_state":
        blocked = _block_chain_selection_state_write_without_exact_chain(args, tool_context)
        if blocked:
            return blocked
    if tool_name in STATE_WRITE_TOOLS:
        input_mode = _context_state(tool_context).get("input_mode", "")
        if input_mode in {"pasted_evidence", "evidence_question"} and not bool(args.get("user_confirmed_from_evidence")):
            return {
                "status": "blocked",
                "data": {
                    "action": tool_name,
                    "reason": "current input is pasted evidence, not confirmed configuration",
                },
                "evidence_paths": [],
                "warnings": [
                    "Pasted logs, transcripts, tracebacks, or old Agent output cannot directly update workflow state."
                ],
                "next_actions": [
                    "summarize the evidence",
                    "ask one explicit confirmation question before applying extracted values",
                ],
                "requires_user_confirmation": True,
            }
    if tool_name not in CONFIRMATION_GATED_TOOLS:
        return None
    if bool(args.get("approved")):
        return None
    return {
        "status": "needs_confirmation",
        "data": {
            "action": tool_name,
            "summary": _confirmation_summary(tool_name),
        },
        "evidence_paths": [],
        "warnings": [],
        "next_actions": ["ask user for explicit yes/no confirmation before retrying with approved=true"],
        "requires_user_confirmation": True,
    }


def _block_chain_selection_state_write_without_exact_chain(
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    state = _context_state(tool_context)
    workflow_state = state.get("workflow_state") if isinstance(state.get("workflow_state"), dict) else {}
    pending = workflow_state.get("pending_question") if isinstance(workflow_state.get("pending_question"), dict) else {}
    if str(pending.get("id") or "") != "chain_selection":
        return None
    patch = args.get("patch") if isinstance(args.get("patch"), dict) else {}
    chain_value = _chain_value_from_patch(patch)
    if not chain_value:
        return None
    user_text = str(state.get("user_text") or "").strip()
    exact_chain = _exact_supported_chain_from_text(user_text)
    if exact_chain and exact_chain == chain_value:
        return None
    return {
        "status": "blocked",
        "data": {
            "action": "update_workflow_state",
            "reason": "chain_selection requires an exact supported chain alias before persisting chain",
            "user_text": user_text,
            "attempted_chain": chain_value,
        },
        "evidence_paths": [],
        "warnings": [
            "Do not coerce partial or ambiguous chain text into a supported chain. Ask a clarification question or enter unsupported-chain onboarding."
        ],
        "next_actions": [
            "keep the chain_selection pending question active",
            "if the text is a likely typo, ask the user to confirm the exact supported chain",
            "if it is a new chain, ask whether it belongs to an existing adapter family and require endpoint/RPC validation",
        ],
        "requires_user_confirmation": False,
    }


def _chain_value_from_patch(patch: dict[str, Any]) -> str:
    direct = str(patch.get("chain") or "").strip().lower().replace("_", "-")
    if direct:
        return direct
    confirmed = patch.get("confirmed_config")
    if isinstance(confirmed, dict):
        return str(confirmed.get("chain") or confirmed.get("BLOCKCHAIN_NODE") or "").strip().lower().replace("_", "-")
    return ""


def after_model_callback(callback_context: Any, llm_response: Any) -> Any | None:
    """Persist exact benchmark entities if a model turn skipped workflow tools.

    This is an ADK-level contract guard, not a terminal intent parser. It only
    records exact framework entities that are already explicit in the current
    user turn: supported chain aliases, target mode tokens, and benchmark
    profile names. Ambiguous or partial text remains for normal ADK handling.
    """
    state = _context_state(callback_context)
    user_text = str(state.get("user_text") or "").strip()
    if not user_text:
        return None
    workflow_state = state.get("workflow_state") if isinstance(state.get("workflow_state"), dict) else {}
    if _ensure_no_endpoint_handoff_state(user_text, workflow_state, state):
        return None
    identity_gate = _ensure_unknown_chain_identity_gate(user_text, workflow_state, state, llm_response)
    if identity_gate:
        return None
    patch: dict[str, Any] = {}
    explicit_chain = _exact_supported_chain_from_text(user_text)
    explicit_target_mode = _target_mode_from_text(user_text)
    explicit_profile = _benchmark_profile_from_text(user_text)
    explicit_rpc_mode = _rpc_mode_from_text(user_text)
    explicit_mixed_weights = _mixed_weights_from_text(user_text)
    explicit_custom_rpc = _custom_rpc_request_from_text(user_text)
    if explicit_chain and not str(workflow_state.get("chain") or "").strip():
        patch["chain"] = explicit_chain
    if explicit_target_mode and not str(workflow_state.get("target_mode") or "").strip():
        patch["target_mode"] = explicit_target_mode
    if explicit_rpc_mode and not str(workflow_state.get("rpc_mode") or "").strip():
        patch["rpc_mode"] = explicit_rpc_mode
    if explicit_mixed_weights and not isinstance(workflow_state.get("mixed_weights"), dict):
        patch["mixed_weights"] = explicit_mixed_weights
    elif explicit_mixed_weights and not workflow_state.get("mixed_weights"):
        patch["mixed_weights"] = explicit_mixed_weights
    if explicit_custom_rpc:
        patch["active_group"] = "target_samples_fixtures"
        patch["next_blocking_group"] = "target_samples_fixtures"
        patch["workflow_step"] = "custom_rpc_requested"
        patch["allowed_next_actions"] = ["ask:custom_rpc_endpoint_gate"]
    profile = workflow_state.get("benchmark_profile") if isinstance(workflow_state.get("benchmark_profile"), dict) else {}
    if explicit_profile and not str(profile.get("mode") or profile.get("name") or "").strip():
        patch["benchmark_profile"] = {"mode": explicit_profile}
    if not patch:
        return None
    confirmed_patch: dict[str, Any] = {}
    if explicit_chain:
        confirmed_patch.update({"chain": explicit_chain, "BLOCKCHAIN_NODE": explicit_chain})
    if explicit_target_mode:
        confirmed_patch.update({
            "target_mode": explicit_target_mode,
            "TARGET_MODE": explicit_target_mode,
            "use_fake_node": explicit_target_mode == "fake-node",
        })
    if explicit_rpc_mode:
        confirmed_patch.update({"rpc_mode": explicit_rpc_mode, "RPC_MODE": explicit_rpc_mode})
    if explicit_profile:
        confirmed_patch["benchmark_mode_confirmed"] = explicit_profile
    if confirmed_patch:
        patch["confirmed_config"] = confirmed_patch
    if explicit_mixed_weights:
        total = sum(int(value) for value in explicit_mixed_weights.values())
        if total != 100:
            patch["active_group"] = "workload_rpc"
            patch["next_blocking_group"] = "workload_rpc"
            patch["workflow_step"] = "mixed_weights_confirm"
            patch["allowed_next_actions"] = ["ask:mixed_weights_confirm"]
            patch["blockers"] = [f"mixed_weights total must be 100, got {total}"]
    patch.setdefault("active_intent", "benchmark")
    patch.setdefault("active_workflow", "benchmark_setup")
    patch.setdefault("workflow_step", "benchmark_setup")
    _update_workflow_state(
        patch,
        reason="adk_after_model_exact_benchmark_entities",
        session_id=str(state.get("session_id") or "terminal-session"),
    )
    return None


def _ensure_no_endpoint_handoff_state(
    user_text: str,
    workflow_state: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    pending = workflow_state.get("pending_question") if isinstance(workflow_state.get("pending_question"), dict) else {}
    pending_id = str(pending.get("id") or "")
    if pending_id not in {"chain_identity_resolution", "chain_protocol_resolution", "unsupported_chain_endpoint_gate"}:
        return False
    if not _no_endpoint_handoff_request_from_text(user_text):
        return False
    identity = workflow_state.get("chain_identity_candidate") if isinstance(workflow_state.get("chain_identity_candidate"), dict) else {}
    protocol = workflow_state.get("chain_protocol_candidate") if isinstance(workflow_state.get("chain_protocol_candidate"), dict) else {}
    chain = str(workflow_state.get("chain") or protocol.get("chain") or identity.get("normalized") or identity.get("raw") or "").strip()
    family = str(
        protocol.get("confirmed_adapter_family")
        or protocol.get("suggested_adapter_family")
        or identity.get("suggested_adapter_family")
        or ""
    ).strip()
    request_unsupported_chain_handoff(
        chain=chain,
        adapter_family=family,
        reason=user_text,
        language=str(state.get("terminal_language") or "en"),
        session_id=str(state.get("session_id") or "terminal-session"),
    )
    return True


def _ensure_unknown_chain_identity_gate(
    user_text: str,
    workflow_state: dict[str, Any],
    state: dict[str, Any],
    llm_response: Any,
) -> bool:
    """Register chain identity gate when model prose skipped the required tool.

    This is an ADK callback guardrail, not terminal routing. It only runs when
    the active typed question is already chain selection, the user text is not
    an exact supported chain alias, and no chain has been persisted. The model
    has already had a chance to reason about the candidate; its final response
    is retained as evidence summary for the typed gate.
    """
    pending = workflow_state.get("pending_question") if isinstance(workflow_state.get("pending_question"), dict) else {}
    if str(pending.get("id") or "") != "chain_selection":
        return False
    if str(workflow_state.get("chain") or "").strip():
        return False
    if _exact_supported_chain_from_text(user_text):
        return False
    candidate = _unknown_chain_candidate_from_text(user_text)
    if not candidate:
        return False
    if str(workflow_state.get("chain_status") or "") in {"identity_needs_confirmation", "protocol_needs_confirmation"}:
        return False
    model_text = _response_text(llm_response)
    suggested_chain = _exact_supported_chain_from_text(model_text)
    propose_chain_identity_resolution(
        candidate_chain=candidate,
        suggested_supported_chain=suggested_chain,
        suggested_adapter_family=_adapter_family_hint_from_text(model_text),
        evidence_summary=model_text,
        web_research_used=False,
        language=str(state.get("terminal_language") or "en"),
        session_id=str(state.get("session_id") or "terminal-session"),
    )
    return True


def _block_downstream_question_when_target_entities_are_unrecorded(
    tool_name: str,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """Prevent later setup groups from skipping explicit target entities.

    This callback does not route user intent and does not mutate workflow
    state. It only enforces that, when the current user turn explicitly names a
    supported chain or target mode, the model must record those target fields
    before registering downstream questions such as disk/QPS/workload choices.
    """
    state = _context_state(tool_context)
    user_text = str(state.get("user_text") or args.get("source_prompt") or "").strip()
    workflow_state = state.get("workflow_state") if isinstance(state.get("workflow_state"), dict) else {}
    if not user_text:
        return _block_downstream_question_when_group_is_not_active(tool_name, args, workflow_state)
    explicit_chain = _exact_supported_chain_from_text(user_text)
    explicit_target_mode = _target_mode_from_text(user_text)
    explicit_profile = _benchmark_profile_from_text(user_text)
    missing: list[str] = []
    if explicit_chain and not str(workflow_state.get("chain") or "").strip():
        missing.append(f"chain={explicit_chain}")
    if explicit_target_mode and not str(workflow_state.get("target_mode") or "").strip():
        missing.append(f"target_mode={explicit_target_mode}")
    profile = workflow_state.get("benchmark_profile") if isinstance(workflow_state.get("benchmark_profile"), dict) else {}
    if explicit_profile and not str(profile.get("mode") or profile.get("name") or "").strip():
        missing.append(f"benchmark_profile={explicit_profile}")
    if not missing:
        return _block_downstream_question_when_group_is_not_active(tool_name, args, workflow_state)
    return {
        "status": "blocked",
        "data": {
            "action": tool_name,
            "reason": "explicit target entities in this user turn were not recorded before a downstream setup question",
            "required_state": missing,
        },
        "evidence_paths": [],
        "warnings": [
            "Record explicit target entities and benchmark profile in workflow state before asking disk, QPS, workload, or observability questions."
        ],
        "next_actions": [
            "call update_workflow_state with the explicit target fields",
            "then call recompute_next_blocking_group or let the deterministic transition executor ask the next typed question",
        ],
        "requires_user_confirmation": False,
    }


def _block_downstream_question_when_group_is_not_active(
    tool_name: str,
    args: dict[str, Any],
    workflow_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Require downstream setup question tools to run inside their group.

    This keeps ADK from creating a second wizard. The model may still jump to a
    group, but it must first record that jump in workflow state; otherwise the
    deterministic transition executor owns the fallback setup sequence.
    """
    expected_groups = _expected_groups_for_downstream_tool(tool_name, args)
    if not expected_groups:
        return None
    active = _normalize_group_name(workflow_state.get("active_group") or "")
    next_group = _normalize_group_name(workflow_state.get("next_blocking_group") or "")
    if active in expected_groups or next_group in expected_groups:
        return None
    return {
        "status": "blocked",
        "data": {
            "action": tool_name,
            "reason": "downstream setup question tool called outside its workflow group",
            "expected_groups": sorted(expected_groups),
            "active_group": active,
            "next_blocking_group": next_group,
        },
        "evidence_paths": [],
        "warnings": [
            "Downstream setup question tools must not create a parallel wizard. Record the user's target group first, or let the transition executor ask the next fallback question."
        ],
        "next_actions": [
            "call record_group_jump for an explicit user interruption",
            "or stop and let deterministic fallback sequencing render the next typed question",
        ],
        "requires_user_confirmation": False,
    }


def _expected_groups_for_downstream_tool(tool_name: str, args: dict[str, Any]) -> set[str]:
    if tool_name == "propose_disk_device_choice":
        role = str(args.get("device_role") or args.get("role") or "").strip().lower()
        if role == "accounts":
            return {"accounts_disk"}
        return {"ledger_disk"}
    if tool_name == "propose_benchmark_profile_choice":
        return {"qps_profile"}
    if tool_name == "propose_workload_customization_choice":
        return {"workload_rpc"}
    return set()


def _exact_supported_chain_from_text(text: str) -> str:
    tokens = _latin_tokens(text)
    if not tokens:
        return ""
    known = set(repo_chain_names())
    aliases = canonical_chain_aliases()
    candidates: dict[str, str] = {name: name for name in known}
    candidates.update(aliases)
    matches: set[str] = set()
    for alias, chain in candidates.items():
        alias_tokens = _latin_tokens(alias)
        if not alias_tokens:
            continue
        for index in range(0, len(tokens) - len(alias_tokens) + 1):
            if tokens[index : index + len(alias_tokens)] != alias_tokens:
                continue
            extra = [
                item
                for item in [*tokens[:index], *tokens[index + len(alias_tokens) :]]
                if not _is_allowed_exact_entity_context_token(item)
            ]
            if not extra:
                matches.add(chain)
    if len(matches) == 1:
        return next(iter(matches))
    return ""


def _target_mode_from_text(text: str) -> str:
    tokens = set(_latin_tokens(text))
    if tokens.intersection({"fake", "fake-node", "fakenode", "mock"}):
        return "fake-node"
    if tokens.intersection({"real", "real-node", "realnode"}) or "真实" in str(text):
        return "real-node"
    return ""


def _benchmark_profile_from_text(text: str) -> str:
    tokens = set(_latin_tokens(text))
    matches = tokens.intersection({"quick", "standard", "intensive"})
    return next(iter(matches)) if len(matches) == 1 else ""


def _rpc_mode_from_text(text: str) -> str:
    tokens = set(_latin_tokens(text))
    modes = tokens.intersection({"single", "mixed"})
    return next(iter(modes)) if len(modes) == 1 else ""


def _mixed_weights_from_text(text: str) -> dict[str, int]:
    raw = str(text or "")
    pairs = re.findall(r"\b([A-Za-z][A-Za-z0-9_.:/-]*)\b\s*(?:=|:)?\s*([0-9]{1,3})\b", raw)
    if not pairs:
        return {}
    weights: dict[str, int] = {}
    for method, weight in pairs:
        if method.lower() in FRAMEWORK_CONTEXT_TOKENS:
            continue
        try:
            value = int(weight)
        except ValueError:
            continue
        if value < 0 or value > 100:
            continue
        if _looks_like_rpc_method(method):
            weights[method] = value
    return weights


def _custom_rpc_request_from_text(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if any(item in lowered for item in ("不需要自定义", "不要自定义", "不用自定义", "no custom", "not custom", "without custom")):
        return False
    tokens = set(_latin_tokens(raw))
    has_rpc = "rpc" in tokens or "rpc" in lowered
    has_custom = bool(tokens.intersection({"custom", "customize", "customized"})) or "自定义" in raw
    has_method = bool(tokens.intersection({"method", "methods"})) or "方法" in raw
    return has_rpc and has_custom and (has_method or "method" in lowered)


def _no_endpoint_handoff_request_from_text(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    mentions_no_endpoint = (
        ("endpoint" in lowered and any(item in raw for item in ("没有", "无", "暂无", "没", "不要", "不提供")))
        or any(item in lowered for item in ("no endpoint", "without endpoint"))
    )
    mentions_handoff = any(
        item in lowered
        for item in ("handoff", "development doc", "developer doc", "coding ai")
    ) or any(item in raw for item in ("开发文档", "交接", "另一个 AI", "另一个AI", "二次开发"))
    return mentions_no_endpoint and mentions_handoff


def _is_allowed_exact_entity_context_token(token: str) -> bool:
    if token in FRAMEWORK_CONTEXT_TOKENS:
        return True
    if token.isdigit():
        return True
    return token in _known_rpc_method_tokens()


def _looks_like_rpc_method(method: str) -> bool:
    text = str(method or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in FRAMEWORK_CONTEXT_TOKENS:
        return False
    tokens = set(_latin_tokens(lowered))
    known = _known_rpc_method_tokens()
    return bool(tokens and tokens.issubset(known)) or "_" in text or "." in text or lowered.startswith(("eth", "get", "net", "web3"))


_RPC_METHOD_TOKEN_CACHE: set[str] | None = None


def _known_rpc_method_tokens() -> set[str]:
    global _RPC_METHOD_TOKEN_CACHE
    if _RPC_METHOD_TOKEN_CACHE is not None:
        return _RPC_METHOD_TOKEN_CACHE
    root = Path(__file__).resolve().parents[2]
    tokens: set[str] = set()
    for path in (root / "config" / "chains").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        methods = payload.get("rpc_methods") if isinstance(payload, dict) else {}
        _collect_rpc_tokens(methods, tokens)
    _RPC_METHOD_TOKEN_CACHE = tokens
    return tokens


def _collect_rpc_tokens(value: Any, tokens: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _add_rpc_method_tokens(str(key), tokens)
            _collect_rpc_tokens(nested, tokens)
    elif isinstance(value, list):
        for item in value:
            _collect_rpc_tokens(item, tokens)
    elif isinstance(value, str):
        _add_rpc_method_tokens(value, tokens)


def _add_rpc_method_tokens(method: str, tokens: set[str]) -> None:
    for token in _latin_tokens(method):
        if token:
            tokens.add(token)


def _unknown_chain_candidate_from_text(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw or len(raw) > 120 or "\n" in raw:
        return ""
    if re.search(r"https?://|\\{|\\}|\\[|\\]", raw):
        return ""
    tokens = _latin_tokens(raw)
    if not tokens:
        return ""
    framework_tokens = FRAMEWORK_CONTEXT_TOKENS | {
        "i",
        "want",
        "need",
        "to",
        "switch",
        "change",
        "benchmark",
        "test",
        "chain",
        "node",
        "use",
        "with",
        "fake",
        "real",
        "mode",
        "first",
        "check",
        "whether",
        "is",
        "a",
        "the",
    }
    kept = [token for token in tokens if token not in framework_tokens]
    if not kept:
        return ""
    return "-".join(kept[:4])


def _adapter_family_hint_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    for family in ("jsonrpc", "substrate", "tendermint", "rest", "bitcoin_jsonrpc", "hedera_dual"):
        if family.replace("_", "-") in lowered or family in lowered:
            return family
    if "json-rpc" in lowered or "evm" in lowered:
        return "jsonrpc"
    if "grpc" in lowered or "rest" in lowered:
        return "rest"
    return ""


def _response_text(llm_response: Any) -> str:
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", []) or []) if content is not None else []
    text_parts = [str(getattr(part, "text", "") or "") for part in parts if getattr(part, "text", None)]
    if text_parts:
        return "\n".join(item.strip() for item in text_parts if item.strip()).strip()
    text = getattr(llm_response, "text", None)
    if text:
        return str(text).strip()
    return str(llm_response or "").strip()


def _latin_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text or "").lower())


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or tool)


def _context_state(tool_context: Any) -> dict[str, Any]:
    try:
        state = getattr(tool_context, "state", None)
        if isinstance(state, dict):
            return state
        if hasattr(state, "to_dict"):
            payload = state.to_dict()
            if isinstance(payload, dict):
                return payload
        if state is not None:
            return dict(state)
    except Exception:
        return {}
    return {}


def _confirmation_summary(tool_name: str) -> str:
    if tool_name == "run_fake_node_smoke_benchmark":
        return "Run the real benchmark engine in quick fake-node mode with isolated job-local output."
    if tool_name == "run_quick_assumed_fake_node_smoke":
        return "Run a quick fake-node smoke test with explicit smoke-only assumed values."
    if tool_name == "submit_benchmark_job":
        return "Submit a real benchmark job that can generate load against the target blockchain node."
    if tool_name == "install_dependencies":
        return "Install benchmark dependencies. This may modify the host, so explicit approval is required."
    if tool_name == "stop_job":
        return "Stop or interrupt a running benchmark job."
    return f"Run confirmation-gated tool: {tool_name}"
