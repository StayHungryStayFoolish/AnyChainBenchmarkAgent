"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..input_values import (
    adapter_family_hint,
    normalize_scalar,
    normalize_target_mode,
    target_mode_evidence_matches,
)
from ..intent import resolve_unknown_chain_identity
from ..localization import localized
from ..questions import render_question
from ..state import AgentGraphState
from ..transitions import (
    invalidate_for_chain_change,
    invalidate_for_target_mode,
)

from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.llm.search_grounding import run_google_search_grounding
from agent.onboarding.families import SUPPORTED_FAMILIES
from .chain_rpc_questions import _adapter_family_question, _answer_option, _choice
from .chain_rpc_questions import _endpoint_validation_question
from .chain_identity_receipts import emit_chain_identity_resolution_receipt

SUPPORTED_ADAPTER_FAMILIES = frozenset(SUPPORTED_FAMILIES)

def _origin_text(state: AgentGraphState, arguments: Mapping[str, Any]) -> str:
    return str(arguments.get("origin_text") or arguments.get("_origin_text") or state.get("last_user_input") or "")


def _target_mode_is_explicit(state: AgentGraphState, mode: str, arguments: Mapping[str, Any]) -> bool:
    if (
        arguments.get("selection_contract_verified") is True
        or arguments.get("target_mode_semantic_verified") is True
    ):
        return True
    if arguments.get("target_mode_explicit") is not True:
        return False
    return target_mode_evidence_matches(
        mode,
        arguments.get("source_evidence"),
        _origin_text(state, arguments),
    )


def _resolution_from_arguments(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    if not (
        any(normalize_scalar(arguments.get(key)) for key in ("canonical_chain_name", "adapter_family", "possible_known_chain"))
        or isinstance(arguments.get("chain_exists"), bool)
    ):
        return None
    return {
        key: deepcopy(value)
        for key, value in arguments.items()
        if key in {"chain_exists", "canonical_chain_name", "adapter_family", "possible_known_chain", "confidence", "reason", "evidence_summary"}
    }


def research_chain_identity(state: AgentGraphState, raw: str, resolution: dict[str, Any] | None = None) -> dict[str, Any]:
    """Research an unconfigured chain without mutating workflow state."""

    resolved = deepcopy(resolution) if resolution is not None else dict(resolve_unknown_chain_identity(state, raw) or {})
    if (state.get("web_research") or {}).get("google_search_available"):
        result = run_google_search_grounding(
            f"{raw} blockchain network: does it exist, what protocol/RPC API does it use, official documentation"
        )
        resolved["search_result"] = result.as_dict()
        if result.available and result.text_summary:
            resolved["evidence_summary"] = result.text_summary
    return resolved


def _verified_search_summary(resolution: Mapping[str, Any]) -> str:
    """Return text only when this run actually produced search grounding.

    The identity model can offer an evidence summary from its training context.
    That is useful for routing, but it must never be presented as a completed
    `google_search` operation.  Search provenance is therefore carried only by
    the typed result produced by `run_google_search_grounding`.
    """

    search_result = resolution.get("search_result")
    if not isinstance(search_result, Mapping) or search_result.get("available") is not True:
        return ""
    return normalize_scalar(search_result.get("text_summary"))


def _known_chain_proposal(resolution: Mapping[str, Any], known: set[str]) -> str:
    """Return a configured-chain proposal without granting it state authority."""

    for key in ("possible_known_chain", "canonical_chain_name"):
        proposed = canonicalize_chain_scalar(
            normalize_scalar(resolution.get(key)),
            known_chains=known,
        )
        if proposed:
            return proposed
    return ""


def _identity_confirmation_question(state: AgentGraphState) -> dict[str, Any] | None:
    """Rebuild an unresolved identity decision entirely from domain state."""

    identity = state.get("chain_identity") or {}
    status = normalize_scalar(identity.get("status"))
    if status not in {"needs_known_chain_confirmation", "needs_identity_confirmation"}:
        return None
    language = state.get("language", "en")
    raw = normalize_scalar(identity.get("raw"))
    resolved = identity.get("llm_resolution") if isinstance(identity.get("llm_resolution"), Mapping) else {}
    if status == "needs_known_chain_confirmation":
        proposal = normalize_scalar(identity.get("proposed_known_chain")) or _known_chain_proposal(
            resolved,
            set(repo_chain_names()),
        )
        if not proposal:
            return None
        return _choice(
            "chain_identity",
            "unknown_chain_identity_confirm",
            localized(
                language,
                f"`{raw}` 不在当前模板中。模型认为你可能想输入 `{proposal}`。是否确认使用这个链？",
                f"`{raw}` is not a configured template. The model thinks you may mean `{proposal}`. Use this chain?",
            ),
            "unknown_chain_decision",
            [
                _answer_option("known", localized(language, f"使用 `{proposal}`", f"Use `{proposal}`"), "confirm_known_chain", {"chain_identity.status": "confirmed"}),
                _answer_option("reenter", localized(language, "不是，重新输入链名", "No, re-enter chain name"), "reenter_chain", {"chain_identity": {}}),
                _answer_option("protocol", localized(language, "这是另一条真实链，继续确认协议", "This is another real chain; choose protocol"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            ],
            queue_barrier=True,
        )

    adapter_family = normalize_scalar(identity.get("adapter_family") or "unknown")
    proposed_name = normalize_scalar(identity.get("proposed_canonical_name") or resolved.get("canonical_chain_name") or raw)
    if resolved.get("chain_exists") and adapter_family in SUPPORTED_ADAPTER_FAMILIES:
        prompt = localized(
            language,
            f"`{raw}` 不在当前 36 条已配置链中。模型建议的名称是 `{proposed_name}`，协议族为 `{adapter_family}`。是否保留你输入的链名 `{raw}`，并按该协议继续 endpoint/RPC 验证？",
            f"`{raw}` is not one of the configured 36 chains. The model proposes the name `{proposed_name}` and adapter family `{adapter_family}`. Keep your entered chain name `{raw}` and continue endpoint/RPC validation with that family?",
        )
        options = [
            _answer_option("confirm", localized(language, "确认，继续 endpoint/RPC 验证", "Yes, continue endpoint/RPC validation"), "confirm_proposed_protocol", {"chain_identity.status": "existing_family_needs_endpoint"}),
            _answer_option("protocol", localized(language, "我来选择协议族", "I will choose the adapter family"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            _answer_option("reenter", localized(language, "我要重新输入链名", "I want to re-enter the chain name"), "reenter_chain", {"chain_identity": {}}),
        ]
    else:
        prompt = localized(
            language,
            f"`{raw}` 不在当前 36 条已配置链或已知别名中。模型没有可靠确认它是已支持协议链。它是一个真实链名，还是你想更正输入？",
            f"`{raw}` is not one of the configured 36 chains or known aliases. The model could not reliably confirm it as a supported-family chain. Is it a real chain name, or do you want to correct it?",
        )
        options = [
            _answer_option("protocol", localized(language, "真实链，继续确认协议", "Real chain; continue to protocol confirmation"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            _answer_option("reenter", localized(language, "我要重新输入链名", "I want to re-enter the chain name"), "reenter_chain", {"chain_identity": {}}),
        ]
    summary = _verified_search_summary(resolved)
    if summary:
        prompt += localized(language, f" 已用 google_search 核实：{summary}", f" Verified via google_search: {summary}")
    return _choice(
        "chain_identity",
        "unknown_chain_identity_confirm",
        prompt,
        "unknown_chain_decision",
        options,
        queue_barrier=True,
    )


def _apply_chain_candidate(state: AgentGraphState, raw: str, resolution: dict[str, Any] | None = None) -> None:
    known = set(repo_chain_names())
    canonical = canonicalize_chain_scalar(raw, known_chains=known)
    state['pending_question'] = {}
    if canonical:
        previous = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
        if previous and previous != canonical:
            invalidate_for_chain_change(state)
        state["chain_identity"] = {"raw": raw, "canonical": canonical, "status": "confirmed", "case": "known"}
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = canonical
        state['active_group'] = "chain_identity"
        state['visible_response'] = list(state.get("visible_response") or []) + [
            localized(state.get("language", "en"), f"已确认链为 `{canonical}`。", f"Confirmed chain: `{canonical}`.")
        ]
        return
    resolved = research_chain_identity(state, raw, resolution)
    emit_chain_identity_resolution_receipt(
        state,
        candidate=raw,
        resolution=resolved,
        resolver_source="planner_proposal" if resolution is not None else "llm",
        confirmation_required=True,
    )
    adapter_family = normalize_scalar(resolved.get("adapter_family") or "unknown")
    canonical_name = normalize_scalar(resolved.get("canonical_chain_name") or raw)
    possible_known = _known_chain_proposal(resolved, known)
    if possible_known:
        state["chain_identity"] = {
            "raw": raw,
            "canonical": raw,
            "proposed_known_chain": possible_known,
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
            "llm_resolution": resolved,
        }
        state['active_group'] = "chain_identity"
        state['pending_question'] = _identity_confirmation_question(state) or {}
        state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]
        return
    state["chain_identity"] = {
        "raw": raw,
        "canonical": raw,
        "proposed_canonical_name": canonical_name if canonical_name != raw else "",
        "adapter_family": adapter_family,
        "status": "needs_identity_confirmation",
        "case": "unknown",
        "requires_llm_identity_resolution": True,
        "llm_resolution": resolved,
    }
    state['active_group'] = "chain_identity"
    state['pending_question'] = _identity_confirmation_question(state) or {}
    state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]


def _preserve_same_chain(state: AgentGraphState, chain: str) -> None:
    identity = state.setdefault("chain_identity", {})
    newly_confirmed = identity.get("status") != "confirmed"
    if newly_confirmed:
        identity.update({"canonical": chain, "status": "confirmed", "case": "known"})
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = chain
    message = localized(
        state.get("language", "en"),
        f"当前链已经是 `{chain}`。我会继续当前配置流程。",
        f"The current chain is already `{chain}`. I will continue the current configuration flow.",
    )
    pending = {} if newly_confirmed else (state.get("pending_question") or {})
    state['visible_response'] = [message] + ([render_question(pending, state.get("language", "en"))] if pending else [])
    if newly_confirmed:
        state['active_group'] = "provider_deployment"
        state['pending_question'] = {}


def _request_chain_change(
    state: AgentGraphState,
    raw: str,
    arguments: Mapping[str, Any],
    *,
    resolution: dict[str, Any] | None = None,
) -> None:
    known = set(repo_chain_names())
    canonical = canonicalize_chain_scalar(raw, known_chains=known)
    previous = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    requested_mode = normalize_target_mode(arguments.get("target_mode"))
    mode_changed = bool(requested_mode and requested_mode != state.get("target_mode"))
    if not previous:
        _apply_chain_candidate(state, raw, resolution)
        return
    if canonical == previous and not mode_changed:
        _preserve_same_chain(state, previous)
        return
    resolved = None if canonical else research_chain_identity(state, raw, resolution)
    if resolved is not None:
        emit_chain_identity_resolution_receipt(
            state,
            candidate=raw,
            resolution=resolved,
            resolver_source=(
                "planner_proposal" if resolution is not None else "llm"
            ),
            confirmation_required=True,
        )
    candidate_label = canonical or raw
    prompt = localized(
        state.get("language", "en"),
        f"是否确认从 `{previous}` 切换到 `{candidate_label}`" + (f"，并使用 `{requested_mode}` 模式" if requested_mode else "") + "？",
        f"Confirm switching from `{previous}` to `{candidate_label}`" + (f" and using `{requested_mode}` mode" if requested_mode else "") + "?",
    )
    candidate = {
        "raw": raw,
        "canonical": canonical,
        "target_mode": requested_mode,
        "resolution": resolved or {},
        "interrupted_group": state.get("active_group") or "",
    }
    state.setdefault("chain_identity", {})["change_candidate"] = candidate
    state['active_group'] = "chain_identity"
    if canonical:
        options = [
            _answer_option("yes", "Y", True, {"chain_identity.canonical": canonical}),
            _answer_option("no", "N", False, {"chain_identity.canonical": previous}),
        ]
        kind = "yes_no"
    else:
        possible_known = _known_chain_proposal(resolved or {}, known)
        family = normalize_scalar((resolved or {}).get("adapter_family") or "unknown")
        if possible_known:
            if possible_known == previous:
                prompt = localized(state.get("language", "en"), f"`{raw}` 不在当前模板中。模型认为你可能想输入当前链 `{possible_known}`。要保持当前链，还是把 `{raw}` 当作另一条真实链继续确认协议？", f"`{raw}` is not a configured template. The model thinks you may mean the current chain `{possible_known}`. Keep the current chain, or treat `{raw}` as another real chain and choose protocol?")
                known_label = localized(state.get("language", "en"), f"保持当前链 `{possible_known}`", f"Keep current chain `{possible_known}`")
            else:
                prompt = localized(state.get("language", "en"), f"`{raw}` 不在当前模板中。模型认为你可能想输入 `{possible_known}`。你要从 `{previous}` 切换到 `{possible_known}`，还是把 `{raw}` 当作另一条真实链继续确认协议？", f"`{raw}` is not a configured template. The model thinks you may mean `{possible_known}`. Switch from `{previous}` to `{possible_known}`, or treat `{raw}` as another real chain and choose protocol?")
                known_label = localized(state.get("language", "en"), f"切换到 `{possible_known}`", f"Switch to `{possible_known}`")
            options = [
                _answer_option("known", known_label, "confirm_known_chain", {"chain_identity.canonical": possible_known}),
                _answer_option("protocol", localized(state.get("language", "en"), f"`{raw}` 是另一条真实链，继续确认协议", f"`{raw}` is another real chain; choose protocol"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", "N", False, {"chain_identity.canonical": previous}),
            ]
        elif (resolved or {}).get("chain_exists") and family in SUPPORTED_ADAPTER_FAMILIES:
            proposed_name = normalize_scalar((resolved or {}).get("canonical_chain_name") or raw)
            prompt = localized(state.get("language", "en"), f"`{raw}` 不在当前 36 条已配置链中。模型建议的名称是 `{proposed_name}`，协议族为 `{family}`。是否从 `{previous}` 切换到你输入的链 `{raw}`，并按该协议继续 endpoint/RPC 验证？", f"`{raw}` is not one of the configured 36 chains. The model proposes the name `{proposed_name}` and adapter family `{family}`. Switch from `{previous}` to your entered chain `{raw}` and continue endpoint/RPC validation with that family?")
            options = [
                _answer_option("confirm", localized(state.get("language", "en"), "确认，继续 endpoint/RPC 验证", "Yes, continue endpoint/RPC validation"), "confirm_proposed_protocol", {"chain_identity.status": "existing_family_needs_endpoint"}),
                _answer_option("protocol", localized(state.get("language", "en"), "我来选择协议族", "I will choose the adapter family"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", "N", False, {"chain_identity.canonical": previous}),
            ]
        else:
            prompt = localized(state.get("language", "en"), f"`{raw}` 不在当前 36 条已配置链或已知别名中。模型没有可靠确认它是已支持协议链。要从 `{previous}` 切换到这条真实链并继续确认协议，还是取消？", f"`{raw}` is not one of the configured 36 chains or known aliases. The model could not reliably confirm it as a supported-family chain. Switch from `{previous}` to this real chain and choose protocol, or cancel?")
            options = [
                _answer_option("protocol", localized(state.get("language", "en"), "真实链，继续确认协议", "Real chain; continue to protocol confirmation"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", "N", False, {"chain_identity.canonical": previous}),
            ]
        summary = _verified_search_summary(resolved or {})
        if summary:
            prompt += localized(state.get("language", "en"), f" 已用 google_search 核实：{summary}", f" Verified via google_search: {summary}")
        kind = "numbered_choice"
    state['pending_question'] = _choice(
        "chain_identity",
        "chain_change_confirm",
        prompt,
        "chain_change_confirmed",
        options,
        kind=kind,
        queue_barrier=True,
    )
    state["pending_question"]["interrupted_group"] = candidate["interrupted_group"]
    state["pending_question"]["supersedes_action_types"] = ["choose_chain", "change_chain"]
    state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]


def _apply_chain_change_decision(state: AgentGraphState, question: Mapping[str, Any], value: Any) -> None:
    current_identity = state.setdefault("chain_identity", {})
    candidate = current_identity.get("change_candidate") or {}
    if value is False:
        current_identity.pop("change_candidate", None)
        state['active_group'] = normalize_scalar(candidate.get("interrupted_group") or question.get("interrupted_group")) or "chain_identity"
        return
    raw = normalize_scalar(candidate.get("raw"))
    resolution = candidate.get("resolution") if isinstance(candidate.get("resolution"), dict) else {}
    requested_mode = normalize_target_mode(candidate.get("target_mode"))
    if requested_mode:
        previous_mode = normalize_target_mode(state.get("target_mode"))
        state["target_mode"] = requested_mode
        state["workflow_mode"] = "sync_observe" if requested_mode == "sync-observe" else "rpc_benchmark"
        invalidate_for_target_mode(state, previous_mode=previous_mode)
    if value == "confirm_known_chain":
        possible = _known_chain_proposal(resolution, set(repo_chain_names()))
        current = canonicalize_chain_scalar(normalize_scalar(current_identity.get("canonical")), known_chains=set(repo_chain_names()))
        if possible and possible == current:
            current_identity.pop("change_candidate", None)
            state['active_group'] = normalize_scalar(candidate.get("interrupted_group")) or "chain_identity"
            state['visible_response'] = [localized(state.get("language", "en"), f"保持当前链 `{possible}`，已确认的链相关配置未更改。", f"Keeping the current chain `{possible}`; confirmed chain-dependent configuration is unchanged.")]
        else:
            invalidate_for_chain_change(state)
            state["chain_identity"] = {"raw": raw, "canonical": possible, "status": "confirmed", "case": "known"}
            state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = possible
            state['active_group'] = "chain_identity"
            state['visible_response'] = [localized(state.get("language", "en"), f"已确认链为 `{possible}`。", f"Confirmed chain: `{possible}`.")]
        return
    invalidate_for_chain_change(state)
    if value in {"confirm_proposed_protocol", "choose_protocol"}:
        state["chain_identity"] = {
            "raw": raw,
            "canonical": raw,
            "proposed_canonical_name": normalize_scalar(resolution.get("canonical_chain_name") or ""),
            "adapter_family": normalize_scalar(resolution.get("adapter_family") or "unknown"),
            "status": "needs_identity_confirmation" if value == "confirm_proposed_protocol" else "needs_protocol_confirmation",
            "case": "unknown",
            "identity_confirmed": True,
            "llm_resolution": resolution,
        }
        if value == "confirm_proposed_protocol":
            _enter_case_for_adapter_family(state, normalize_scalar(resolution.get("adapter_family") or "unknown"))
        else:
            state['active_group'] = "chain_identity"
            state['pending_question'] = _adapter_family_question(state)
            state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]
        return
    _apply_chain_candidate(state, raw, resolution)


def _apply_unknown_chain_decision(state: AgentGraphState, value: Any, user_text: str) -> None:
    family = ""
    if isinstance(value, dict):
        family = normalize_scalar(value.get("choose_protocol_family"))
    identity = state.setdefault("chain_identity", {})
    if family:
        _convert_known_candidate_to_unknown(state)
        identity = state.setdefault("chain_identity", {})
        identity["identity_confirmed"] = True
        _enter_case_for_adapter_family(state, family)
        return
    if value == "reenter_chain":
        state["chain_identity"] = {}
        return
    if value == "confirm_known_chain":
        canonical = normalize_scalar(identity.get("proposed_known_chain")) or _known_chain_proposal(
            identity.get("llm_resolution") if isinstance(identity.get("llm_resolution"), Mapping) else {},
            set(repo_chain_names()),
        )
        if canonical:
            state["chain_identity"] = {"raw": normalize_scalar(identity.get("raw") or user_text), "canonical": canonical, "status": "confirmed", "case": "known"}
            state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = canonical
            state['active_group'] = "provider_deployment"
        return
    if value == "choose_protocol":
        _convert_known_candidate_to_unknown(state)
        state.setdefault("chain_identity", {})["status"] = "needs_protocol_confirmation"
        state["chain_identity"]["identity_confirmed"] = True
        state['active_group'] = "chain_identity"
        state['pending_question'] = _adapter_family_question(state)
        state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]
        return
    if value == "confirm_proposed_protocol":
        _enter_case_for_adapter_family(state, normalize_scalar(identity.get("adapter_family")))


def _convert_known_candidate_to_unknown(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    raw = normalize_scalar(identity.get("raw"))
    resolution = identity.get("llm_resolution") if isinstance(identity.get("llm_resolution"), dict) else {}
    identity.clear()
    identity.update(
        {
            "raw": raw,
            # The model/search candidate was explicitly rejected. Keep it as
            # evidence only; only a user-confirmed identity may become the
            # canonical chain used by downstream configuration and execution.
            "canonical": raw,
            "adapter_family": normalize_scalar(resolution.get("adapter_family") or "unknown"),
            "status": "needs_protocol_confirmation",
            "case": "unknown",
            "requires_llm_identity_resolution": True,
            "llm_resolution": resolution,
        }
    )


def _enter_case_for_adapter_family(state: AgentGraphState, family: str) -> None:
    identity = state.setdefault("chain_identity", {})
    family = normalize_scalar(family)
    identity["adapter_family"] = family
    if family not in SUPPORTED_ADAPTER_FAMILIES:
        _route_unsupported_family(state)
        return
    identity.update({"status": "existing_family_needs_endpoint", "case": "case2", "identity_confirmed": True})
    state['active_group'] = "endpoint_process"
    state['pending_question'] = _endpoint_validation_question(state) or {}
    state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))] if state["pending_question"] else []


def _confirm_custom_rpc_family(state: AgentGraphState, family: str) -> None:
    identity = state.setdefault("chain_identity", {})
    identity["adapter_family"] = family
    if family not in SUPPORTED_ADAPTER_FAMILIES:
        _route_unsupported_family(state)
        return
    custom = state.setdefault("custom_rpc", {})
    custom.update({"status": "needs_endpoint", "endpoint_ready": False, "job_local_override": True})
    state['active_group'] = "endpoint_process"
    state['visible_response'] = [localized(state.get("language", "en"), f"已更新协议族为 `{family}`。请重新提供可访问的 RPC endpoint 用于验证。", f"Adapter family updated to `{family}`. Provide a reachable RPC endpoint to validate again.")]


def _route_unsupported_family(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    identity.update({"status": "unsupported_family_handoff", "case": "case3"})
    handoff = state.setdefault("secondary_handoff", {})
    handoff.update(
        {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "chain": normalize_scalar(identity.get("canonical") or identity.get("raw")),
            "adapter_family": normalize_scalar(identity.get("adapter_family") or "unsupported"),
            "evidence": list(handoff.get("evidence") or []),
        }
    )
    state['active_group'] = "chain_identity"
    state['pending_question'] = {}
    state['visible_response'] = [localized(state.get("language", "en"), "该链目前不属于已支持协议族。请提供官方协议/RPC 文档、endpoint 文档、request/response 示例；我会生成二次开发交接文档。", "This chain is outside the supported adapter families. Provide official protocol/RPC docs, endpoint docs, and request/response examples; I will generate a secondary-development handoff.")]


def _request_target_mode_change(state: AgentGraphState, mode: str) -> None:
    previous = normalize_target_mode(state.get("target_mode"))
    interrupted = normalize_scalar(state.get("active_group"))
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    chain_zh = f"链 `{chain}` 会保留（如需更换请直接说明要测哪条链）；" if chain else ""
    chain_en = f"Chain `{chain}` will be kept (say which chain you want if it should change); " if chain else ""
    if mode == "sync-observe":
        impact_zh = "环境/机器/磁盘/网络证据会保留；RPC workload、自定义 method/fixture 和 QPS profile 会清空，因为 sync-observe 不走 Vegeta；同步 endpoint、进程和 metrics source 会重新验证。"
        impact_en = "Environment, machine, disk, and network evidence is kept. RPC workload, custom methods/fixtures, and the QPS profile are cleared because sync-observe does not use Vegeta. Its sync endpoint, process, and metrics source are validated separately."
    elif previous == "sync-observe":
        impact_zh = "环境/机器/磁盘/网络证据会保留；sync-observe 专用状态会清空；RPC endpoint、workload 和 QPS 会按新的 benchmark 模式重新确认。"
        impact_en = "Environment, machine, disk, and network evidence is kept. Sync-observe-only state is cleared; the RPC endpoint, workload, and QPS profile are confirmed for the new benchmark mode."
    else:
        impact_zh = "环境/机器/磁盘/网络证据会保留；endpoint、进程和执行证据会按新模式重新确认，避免复用不适用的运行状态。"
        impact_en = "Environment, machine, disk, and network evidence is kept. Endpoint, process, and execution evidence is re-confirmed for the new mode so incompatible runtime state is not reused."
    state["target_mode_change_candidate"] = mode
    state['active_group'] = "target_mode"
    state['pending_question'] = _choice(
        "target_mode",
        "target_mode_change_confirm",
        localized(
            state.get("language", "en"),
            f"是否确认从 `{previous or '<unset>'}` 切换到 `{mode}`？{chain_zh}{impact_zh}",
            f"Confirm switching from `{previous or '<unset>'}` to `{mode}`? {chain_en}{impact_en}",
        ),
        "target_mode_change_confirmed",
        [
            _answer_option("yes", "Y", True, {"target_mode": mode}),
            _answer_option("no", "N", False, {"target_mode": previous}),
        ],
        kind="yes_no",
        queue_barrier=True,
    )
    state["pending_question"]["interrupted_group"] = interrupted
    state["pending_question"]["previous_mode"] = previous
    state["pending_question"]["supersedes_action_types"] = ["choose_target_mode"]
    state['visible_response'] = [render_question(state["pending_question"], state.get("language", "en"))]


def _chain_ambiguity_question(
    state: AgentGraphState,
    primary: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidate = normalize_scalar(primary)
    raw_candidates = arguments.get("chain_candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidate_values = [normalize_scalar(item) for item in raw_candidates if normalize_scalar(item)]
    if candidate and candidate not in candidate_values:
        candidate_values.insert(0, candidate)
    known = set(repo_chain_names())
    semantic_candidates = {
        canonicalize_chain_scalar(item, known_chains=known) or item.casefold()
        for item in candidate_values
    }
    if len(semantic_candidates) < 2:
        return None
    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidate_values[:6]:
        normalized = canonicalize_chain_scalar(raw, known_chains=known)
        key = normalized or raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        if normalized:
            options.append(_answer_option(normalized, localized(state.get("language", "en"), f"使用已支持链 `{normalized}`（来自 `{raw}`）", f"Use configured chain `{normalized}` from `{raw}`"), {"chain_choice": normalized}, {"chain_identity.canonical": normalized}))
        else:
            options.append(_answer_option(raw, localized(state.get("language", "en"), f"`{raw}` 是另一条真实链，进入新链确认", f"`{raw}` is another real chain; enter new-chain confirmation"), {"unknown_chain_choice": raw}, {"chain_identity.raw": raw}))
    options.append(_answer_option("reenter", localized(state.get("language", "en"), "我重新输入链名", "I will re-enter the chain name"), "reenter_chain", {"chain_identity": {}}))
    if len(options) < 2:
        return None
    question = _choice(
        "chain_identity",
        "chain_ambiguity_confirm",
        localized(state.get("language", "en"), "你这句话里有链名不确定或多个候选。请先确认要测试哪条链。", "Your turn contains an uncertain or multiple chain candidates. Confirm which chain to test first."),
        "chain_ambiguity_choice",
        options,
        queue_barrier=True,
    )
    question["supersedes_action_types"] = ["choose_chain", "change_chain"]
    return question
