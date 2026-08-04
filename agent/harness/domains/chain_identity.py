"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..input_values import (
    normalize_scalar,
    normalize_target_mode,
)
from ..advisory import (
    normalize_chain_identity_resolution,
    resolve_unknown_chain_identity,
)
from ..questions import question_text
from ..state import AgentGraphState
from ..transitions import (
    invalidate_for_adapter_family_change,
    invalidate_for_chain_change,
    invalidate_for_target_mode,
)

from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.llm.search_grounding import run_google_search_grounding
from agent.onboarding.families import SUPPORTED_FAMILIES
from .chain_rpc_questions import (
    _adapter_family_question,
    _answer_option,
    _chain_question,
    _chain_selection_question,
    _choice,
)
from .chain_rpc_questions import _endpoint_validation_question
from .chain_identity_receipts import emit_chain_identity_resolution_receipt
from .response_fragments import ResponseCollector, emit

SUPPORTED_ADAPTER_FAMILIES = frozenset(SUPPORTED_FAMILIES)

def _origin_text(state: AgentGraphState, arguments: Mapping[str, Any]) -> str:
    return str(arguments.get("origin_text") or arguments.get("_origin_text") or state.get("last_user_input") or "")


def _target_mode_is_explicit(state: AgentGraphState, mode: str, arguments: Mapping[str, Any]) -> bool:
    return bool(
        arguments.get("selection_contract_verified") is True
        or arguments.get("target_mode_semantic_verified") is True
    )


def _resolution_from_arguments(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    if not (
        any(normalize_scalar(arguments.get(key)) for key in ("canonical_chain_name", "adapter_family", "possible_known_chain"))
        or isinstance(arguments.get("chain_exists"), bool)
        or normalize_scalar(arguments.get("reference_kind"))
    ):
        return None
    return {
        key: deepcopy(value)
        for key, value in arguments.items()
        if key in {
            "reference_kind",
            "chain_exists",
            "canonical_chain_name",
            "adapter_family",
            "possible_known_chain",
            "confidence",
            "reason",
            "evidence_summary",
        }
    }


def research_chain_identity(state: AgentGraphState, raw: str, resolution: dict[str, Any] | None = None) -> dict[str, Any]:
    """Research an unconfigured chain without mutating workflow state."""

    proposal = deepcopy(resolution) if resolution is not None else {}
    if normalize_scalar(proposal.get("reference_kind")) not in {
        "named_identity",
        "generic_reference",
        "uncertain",
    }:
        classified = dict(resolve_unknown_chain_identity(state, raw) or {})
        resolved = {**proposal, **classified}
    else:
        resolved = proposal
    if (
        resolved.get("reference_kind") == "named_identity"
        and (state.get("web_research") or {}).get("google_search_available")
    ):
        result = run_google_search_grounding(
            f"{raw} blockchain network: does it exist, what protocol/RPC API does it use, official documentation"
        )
        resolved["search_result"] = result.as_dict()
        if result.available and result.text_summary:
            resolved["evidence_summary"] = result.text_summary
    return normalize_chain_identity_resolution(resolved)


def _identity_resolver_source(
    resolution: Mapping[str, Any] | None,
) -> str:
    if (
        isinstance(resolution, Mapping)
        and normalize_scalar(resolution.get("reference_kind"))
        in {"named_identity", "generic_reference", "uncertain"}
    ):
        return "planner_proposal"
    return "llm"


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
            question_text(
                "question.chain_rpc.unknown_chain.known_proposal.prompt",
                raw=raw,
                proposal=proposal,
            ),
            "unknown_chain_decision",
            [
                _answer_option("known", question_text("question.chain_rpc.option.use_known_chain", chain=proposal), "confirm_known_chain", {"chain_identity.status": "confirmed"}),
                _answer_option("reenter", question_text("question.chain_rpc.option.reenter_chain"), "reenter_chain", {"chain_identity": {}}),
                _answer_option("protocol", question_text("question.chain_rpc.option.other_real_chain_protocol"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            ],
            queue_barrier=True,
        )

    adapter_family = normalize_scalar(identity.get("adapter_family") or "unknown")
    proposed_name = normalize_scalar(identity.get("proposed_canonical_name") or resolved.get("canonical_chain_name") or raw)
    summary = _verified_search_summary(resolved)
    if resolved.get("chain_exists") and adapter_family in SUPPORTED_ADAPTER_FAMILIES:
        prompt = (
            question_text(
                "question.chain_rpc.unknown_chain.supported_family_grounded.prompt",
                raw=raw,
                proposed_name=proposed_name,
                adapter_family=adapter_family,
                search_summary=summary,
            )
            if summary
            else question_text(
                "question.chain_rpc.unknown_chain.supported_family.prompt",
                raw=raw,
                proposed_name=proposed_name,
                adapter_family=adapter_family,
            )
        )
        options = [
            _answer_option("confirm", question_text("question.chain_rpc.option.continue_endpoint_validation"), "confirm_proposed_protocol", {"chain_identity.status": "existing_family_needs_endpoint"}),
            _answer_option("protocol", question_text("question.chain_rpc.option.choose_adapter_family"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            _answer_option("reenter", question_text("question.chain_rpc.option.reenter_chain"), "reenter_chain", {"chain_identity": {}}),
        ]
    else:
        prompt = (
            question_text(
                "question.chain_rpc.unknown_chain.unresolved_grounded.prompt",
                raw=raw,
                search_summary=summary,
            )
            if summary
            else question_text(
                "question.chain_rpc.unknown_chain.unresolved.prompt",
                raw=raw,
            )
        )
        options = [
            _answer_option("protocol", question_text("question.chain_rpc.option.real_chain_protocol"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
            _answer_option("reenter", question_text("question.chain_rpc.option.reenter_chain"), "reenter_chain", {"chain_identity": {}}),
        ]
    return _choice(
        "chain_identity",
        "unknown_chain_identity_confirm",
        prompt,
        "unknown_chain_decision",
        options,
        queue_barrier=True,
    )


def _apply_chain_candidate(
    state: AgentGraphState,
    raw: str,
    resolution: dict[str, Any] | None = None,
    *,
    responses: ResponseCollector,
) -> None:
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
        emit(
            responses,
            "chain_rpc.response.chain_confirmed",
            arguments={"chain": canonical},
            source=__name__,
        )
        return
    resolved = research_chain_identity(state, raw, resolution)
    emit_chain_identity_resolution_receipt(
        state,
        candidate=raw,
        resolution=resolved,
        resolver_source=_identity_resolver_source(resolution),
        confirmation_required=True,
    )
    if resolved.get("reference_kind") in {"generic_reference", "uncertain"}:
        identity = state.get("chain_identity") or {}
        if identity.get("status") != "confirmed":
            state["chain_identity"] = {}
        state["active_group"] = "chain_identity"
        state["pending_question"] = _chain_question(state)
        return
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


def _preserve_same_chain(
    state: AgentGraphState,
    chain: str,
    *,
    responses: ResponseCollector,
) -> None:
    identity = state.setdefault("chain_identity", {})
    newly_confirmed = identity.get("status") != "confirmed"
    if newly_confirmed:
        identity.update({"canonical": chain, "status": "confirmed", "case": "known"})
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = chain
    emit(
        responses,
        "chain_rpc.response.same_chain",
        arguments={"chain": chain},
        source=__name__,
    )
    if newly_confirmed:
        state['active_group'] = "provider_deployment"
        state['pending_question'] = {}


def _request_chain_change(
    state: AgentGraphState,
    raw: str,
    arguments: Mapping[str, Any],
    *,
    resolution: dict[str, Any] | None = None,
    responses: ResponseCollector,
) -> None:
    known = set(repo_chain_names())
    canonical = canonicalize_chain_scalar(raw, known_chains=known)
    previous = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    requested_mode = normalize_target_mode(arguments.get("target_mode"))
    mode_changed = bool(requested_mode and requested_mode != state.get("target_mode"))
    if not previous:
        _apply_chain_candidate(state, raw, resolution, responses=responses)
        return
    if canonical == previous and not mode_changed:
        _preserve_same_chain(state, previous, responses=responses)
        return
    resolved = None if canonical else research_chain_identity(state, raw, resolution)
    if resolved is not None:
        emit_chain_identity_resolution_receipt(
            state,
            candidate=raw,
            resolution=resolved,
            resolver_source=_identity_resolver_source(resolution),
            confirmation_required=True,
        )
        if resolved.get("reference_kind") in {
            "generic_reference",
            "uncertain",
        }:
            state["active_group"] = "chain_identity"
            state["pending_question"] = _chain_selection_question(state)
            return
    candidate_label = canonical or raw
    prompt = (
        question_text(
            "question.chain_rpc.chain_change.known_with_mode.prompt",
            previous=previous,
            candidate=candidate_label,
            target_mode=requested_mode,
        )
        if requested_mode
        else question_text(
            "question.chain_rpc.chain_change.known.prompt",
            previous=previous,
            candidate=candidate_label,
        )
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
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"chain_identity.canonical": canonical}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"chain_identity.canonical": previous}),
        ]
        kind = "yes_no"
    else:
        possible_known = _known_chain_proposal(resolved or {}, known)
        family = normalize_scalar((resolved or {}).get("adapter_family") or "unknown")
        summary = _verified_search_summary(resolved or {})
        if possible_known:
            if possible_known == previous:
                prompt = (
                    question_text(
                        "question.chain_rpc.chain_change.possible_current_grounded.prompt",
                        raw=raw,
                        possible_known=possible_known,
                        search_summary=summary,
                    )
                    if summary
                    else question_text(
                        "question.chain_rpc.chain_change.possible_current.prompt",
                        raw=raw,
                        possible_known=possible_known,
                    )
                )
                known_label = question_text(
                    "question.chain_rpc.option.keep_known_chain",
                    chain=possible_known,
                )
            else:
                prompt = (
                    question_text(
                        "question.chain_rpc.chain_change.possible_known_grounded.prompt",
                        raw=raw,
                        previous=previous,
                        possible_known=possible_known,
                        search_summary=summary,
                    )
                    if summary
                    else question_text(
                        "question.chain_rpc.chain_change.possible_known.prompt",
                        raw=raw,
                        previous=previous,
                        possible_known=possible_known,
                    )
                )
                known_label = question_text(
                    "question.chain_rpc.option.switch_known_chain",
                    chain=possible_known,
                )
            options = [
                _answer_option("known", known_label, "confirm_known_chain", {"chain_identity.canonical": possible_known}),
                _answer_option("protocol", question_text("question.chain_rpc.option.raw_is_other_chain", raw=raw), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"chain_identity.canonical": previous}),
            ]
        elif (resolved or {}).get("chain_exists") and family in SUPPORTED_ADAPTER_FAMILIES:
            proposed_name = normalize_scalar((resolved or {}).get("canonical_chain_name") or raw)
            prompt = (
                question_text(
                    "question.chain_rpc.chain_change.supported_family_grounded.prompt",
                    raw=raw,
                    previous=previous,
                    proposed_name=proposed_name,
                    adapter_family=family,
                    search_summary=summary,
                )
                if summary
                else question_text(
                    "question.chain_rpc.chain_change.supported_family.prompt",
                    raw=raw,
                    previous=previous,
                    proposed_name=proposed_name,
                    adapter_family=family,
                )
            )
            options = [
                _answer_option("confirm", question_text("question.chain_rpc.option.continue_endpoint_validation"), "confirm_proposed_protocol", {"chain_identity.status": "existing_family_needs_endpoint"}),
                _answer_option("protocol", question_text("question.chain_rpc.option.choose_adapter_family"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"chain_identity.canonical": previous}),
            ]
        else:
            prompt = (
                question_text(
                    "question.chain_rpc.chain_change.unresolved_grounded.prompt",
                    raw=raw,
                    previous=previous,
                    search_summary=summary,
                )
                if summary
                else question_text(
                    "question.chain_rpc.chain_change.unresolved.prompt",
                    raw=raw,
                    previous=previous,
                )
            )
            options = [
                _answer_option("protocol", question_text("question.chain_rpc.option.real_chain_protocol"), "choose_protocol", {"chain_identity.status": "needs_protocol_confirmation"}),
                _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"chain_identity.canonical": previous}),
            ]
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
    state["pending_question"]["supersedes_action_types"] = ["choose_chain"]


def _apply_chain_change_decision(
    state: AgentGraphState,
    question: Mapping[str, Any],
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
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
            emit(
                responses,
                "chain_rpc.response.chain_kept",
                arguments={"chain": possible},
                source=__name__,
            )
        else:
            invalidate_for_chain_change(state)
            state["chain_identity"] = {"raw": raw, "canonical": possible, "status": "confirmed", "case": "known"}
            state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = possible
            state['active_group'] = "chain_identity"
            emit(
                responses,
                "chain_rpc.response.chain_confirmed",
                arguments={"chain": possible},
                source=__name__,
            )
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
            _enter_case_for_adapter_family(
                state,
                normalize_scalar(resolution.get("adapter_family") or "unknown"),
                responses=responses,
            )
        else:
            state['active_group'] = "chain_identity"
            state['pending_question'] = _adapter_family_question(state)
        return
    _apply_chain_candidate(state, raw, resolution, responses=responses)


def _apply_unknown_chain_decision(
    state: AgentGraphState,
    value: Any,
    user_text: str,
    *,
    responses: ResponseCollector,
) -> None:
    family = ""
    if isinstance(value, dict):
        family = normalize_scalar(value.get("choose_protocol_family"))
    identity = state.setdefault("chain_identity", {})
    if family:
        _convert_known_candidate_to_unknown(state)
        identity = state.setdefault("chain_identity", {})
        identity["identity_confirmed"] = True
        _enter_case_for_adapter_family(state, family, responses=responses)
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
        return
    if value == "confirm_proposed_protocol":
        _enter_case_for_adapter_family(
            state,
            normalize_scalar(identity.get("adapter_family")),
            responses=responses,
        )


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


def _enter_case_for_adapter_family(
    state: AgentGraphState,
    family: str,
    *,
    responses: ResponseCollector,
) -> None:
    identity = state.setdefault("chain_identity", {})
    family = normalize_scalar(family)
    previous_family = normalize_scalar(identity.get("adapter_family"))
    if previous_family and previous_family != family:
        invalidate_for_adapter_family_change(state)
    identity["adapter_family"] = family
    if family not in SUPPORTED_ADAPTER_FAMILIES:
        _route_unsupported_family(state, responses=responses)
        return
    identity.update({"status": "existing_family_needs_endpoint", "case": "case2", "identity_confirmed": True})
    state['active_group'] = "endpoint_process"
    state['pending_question'] = _endpoint_validation_question(state) or {}


def _confirm_custom_rpc_family(
    state: AgentGraphState,
    family: str,
    *,
    responses: ResponseCollector,
) -> None:
    identity = state.setdefault("chain_identity", {})
    identity["adapter_family"] = family
    if family not in SUPPORTED_ADAPTER_FAMILIES:
        _route_unsupported_family(state, responses=responses)
        return
    custom = state.setdefault("custom_rpc", {})
    custom.update({"status": "needs_endpoint", "endpoint_ready": False, "job_local_override": True})
    state['active_group'] = "endpoint_process"
    emit(
        responses,
        "chain_rpc.response.adapter_family_updated",
        arguments={"adapter_family": family},
        source=__name__,
    )


def _route_unsupported_family(
    state: AgentGraphState,
    *,
    responses: ResponseCollector,
) -> None:
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
    emit(
        responses,
        "chain_rpc.response.unsupported_family_handoff",
        source=__name__,
    )


def _request_target_mode_change(state: AgentGraphState, mode: str) -> None:
    previous = normalize_target_mode(state.get("target_mode"))
    interrupted = normalize_scalar(state.get("active_group"))
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    if mode == "sync-observe":
        prompt = question_text(
            "question.chain_rpc.target_mode_change.to_sync_observe.prompt",
            previous=previous or "<unset>",
            target_mode=mode,
            retained_chain=chain or "<none>",
        )
    elif previous == "sync-observe":
        prompt = question_text(
            "question.chain_rpc.target_mode_change.from_sync_observe.prompt",
            previous=previous or "<unset>",
            target_mode=mode,
            retained_chain=chain or "<none>",
        )
    else:
        prompt = question_text(
            "question.chain_rpc.target_mode_change.benchmark.prompt",
            previous=previous or "<unset>",
            target_mode=mode,
            retained_chain=chain or "<none>",
        )
    state["target_mode_change_candidate"] = mode
    state['active_group'] = "target_mode"
    state['pending_question'] = _choice(
        "target_mode",
        "target_mode_change_confirm",
        prompt,
        "target_mode_change_confirmed",
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"target_mode": mode}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"target_mode": previous}),
        ],
        kind="yes_no",
        queue_barrier=True,
    )
    state["pending_question"]["interrupted_group"] = interrupted
    state["pending_question"]["previous_mode"] = previous
    state["pending_question"]["supersedes_action_types"] = ["choose_target_mode"]


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
            options.append(_answer_option(
                normalized,
                question_text(
                    "question.chain_rpc.option.ambiguity_known_chain",
                    chain=normalized,
                    raw=raw,
                ),
                {"chain_choice": normalized},
                {"chain_identity.canonical": normalized},
            ))
        else:
            options.append(_answer_option(
                raw,
                question_text(
                    "question.chain_rpc.option.ambiguity_unknown_chain",
                    raw=raw,
                ),
                {"unknown_chain_choice": raw},
                {"chain_identity.raw": raw},
            ))
    options.append(_answer_option(
        "reenter",
        question_text("question.chain_rpc.option.reenter_chain"),
        "reenter_chain",
        {"chain_identity": {}},
    ))
    if len(options) < 2:
        return None
    question = _choice(
        "chain_identity",
        "chain_ambiguity_confirm",
        question_text("question.chain_rpc.ambiguity.prompt"),
        "chain_ambiguity_choice",
        options,
        queue_barrier=True,
    )
    question["supersedes_action_types"] = ["choose_chain"]
    return question
