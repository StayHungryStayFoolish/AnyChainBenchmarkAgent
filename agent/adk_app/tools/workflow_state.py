"""ADK tools for structured workflow state.

These tools give ADK a persistent place to store confirmed facts and workflow
progress. They do not classify natural language or route business intent.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from knowledge.chain_identity import FRAMEWORK_CONTEXT_TOKENS
from knowledge.chain_identity import canonical_chain_aliases
from knowledge.chain_identity import canonicalize_chain_scalar
from knowledge.framework_capabilities import load_framework_capabilities
from workflows.conversation_state import (
    answer_pending_question as _answer_pending_question,
    load_workflow_state as _load_workflow_state,
    recompute_next_blocking_group as _recompute_next_blocking_group,
    record_group_jump as _record_group_jump,
    revert_workflow_state as _revert_workflow_state,
    reset_workflow_state as _reset_workflow_state,
    update_workflow_state as _update_workflow_state,
)

from .read_only import _tool_result


_CURRENT_SESSION_ID: ContextVar[str] = ContextVar("anychain_workflow_tool_session_id", default="")
_CURRENT_STATE_ROOT: ContextVar[str] = ContextVar("anychain_workflow_tool_state_root", default="")
SUPPORTED_ADAPTER_FAMILIES = {
    "jsonrpc",
    "substrate",
    "rest",
    "tendermint",
    "bitcoin_jsonrpc",
    "hedera_dual",
}


@contextmanager
def workflow_tool_session(session_id: str, state_root: str | Path | None = None):
    """Bind ADK workflow-state tools to the current product-terminal session."""
    token_session = _CURRENT_SESSION_ID.set(session_id or "terminal-session")
    token_root = _CURRENT_STATE_ROOT.set(str(state_root or ""))
    try:
        yield
    finally:
        _CURRENT_SESSION_ID.reset(token_session)
        _CURRENT_STATE_ROOT.reset(token_root)


def _effective_session_id(session_id: str = "") -> str:
    return session_id or _CURRENT_SESSION_ID.get() or "terminal-session"


def _effective_state_root() -> str | Path:
    return _CURRENT_STATE_ROOT.get() or ".agent/sessions"


def load_workflow_state(session_id: str = "") -> dict[str, Any]:
    """Load the current structured conversation/workflow state."""
    effective_session_id = _effective_session_id(session_id)
    return _tool_result(
        data=_load_workflow_state(session_id=effective_session_id, state_root=_effective_state_root()),
        next_actions=["continue current workflow", "ask pending question", "validate confirmed config"],
    )


def update_workflow_state(
    patch: dict | None = None,
    reason: str = "",
    session_id: str = "",
    user_confirmed_from_evidence: bool = False,
) -> dict[str, Any]:
    """Persist structured facts inferred by ADK from the conversation.

    Pass explicit fields only. Do not pass raw natural language as a parsing
    shortcut. Set ``user_confirmed_from_evidence`` only after the user has
    explicitly approved applying values extracted from pasted logs, tracebacks,
    or old Agent output.
    """
    _ = user_confirmed_from_evidence
    effective_session_id = _effective_session_id(session_id)
    payload = _update_workflow_state(
        patch or {},
        reason=reason,
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        warnings=[
            *[f"ignored unsupported state key: {key}" for key in payload.get("ignored_keys", [])],
            *payload.get("validation_warnings", []),
        ],
        next_actions=["validate current workflow step", "ask one blocking question", "prepare benchmark run"],
    )


def propose_opening_help_choice(
    language: str = "en",
    latest_job_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the typed opening menu for greetings and capability openings.

    This is intentionally not benchmark setup. It lets a short follow-up such
    as ``2`` deterministically choose a next high-level path without treating
    the original greeting as benchmark intent.
    """
    effective_session_id = _effective_session_id(session_id)
    prompt, options = _opening_help_prompt_and_options(language, latest_job_id)
    payload = _update_workflow_state(
        {
            "active_intent": "opening",
            "active_workflow": "opening",
            "workflow_step": "opening_help_choice",
            "pending_question": {
                "id": "opening_help_choice",
                "kind": "numbered_choice",
                "prompt": prompt,
                "source": "workflow_tool",
                "source_tool": "propose_opening_help_choice",
                "branch": "startup",
                "workflow_step": "opening_help_choice",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": False,
                "blocking": False,
                "options": options,
            },
        },
        reason="propose_opening_help_choice",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "opening_help_choice",
            "options": options,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for a numbered choice or natural-language goal"],
    )


def answer_pending_question(
    answer: str,
    reason: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Apply a short answer to the active typed pending question.

    Use this for user replies such as Y/N, numbered choices, disk names, URLs,
    and "back". It must only act on the current pending_question state.
    """
    effective_session_id = _effective_session_id(session_id)
    payload = _answer_pending_question(
        answer,
        reason=reason,
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("blockers", []),
        next_actions=payload.get("next_actions", ["ask one blocking question", "validate current workflow step"]),
    )


def record_group_jump(
    target_group: str,
    reason: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Switch workflow control to another canonical configuration group.

    Use this after ADK/model has classified a non-structural user turn as a
    request to change groups, for example changing chain during disk setup or
    adjusting QPS during workload setup. This tool does not classify natural
    language; it only applies the chosen group transition.
    """
    effective_session_id = _effective_session_id(session_id)
    payload = _record_group_jump(
        target_group,
        reason=reason,
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    warnings = payload.get("blockers", []) if not payload.get("applied") else []
    return _tool_result(
        status="ok" if payload.get("applied") else "blocked",
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        warnings=warnings,
        next_actions=[
            "validate the selected group",
            "register one typed pending question for the selected group",
            "after the group completes, recompute the next blocking group",
        ],
    )


def recompute_next_blocking_group(
    reason: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Recompute workflow progress after a group completes or is invalidated."""
    effective_session_id = _effective_session_id(session_id)
    payload = _recompute_next_blocking_group(
        reason=reason,
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        next_actions=[
            "if next_blocking_group is set, ask one question for that group",
            "if no next_blocking_group remains, run preflight/smoke gate",
        ],
    )


def propose_benchmark_target_mode_choice(
    source_prompt: str = "",
    language: str = "en",
    explicit_benchmark_intent: bool = False,
    session_id: str = "",
) -> dict[str, Any]:
    """Register the typed fake-node / real-node / explain choice.

    Use this when the user says only that they want to test or benchmark and
    has not provided a target mode. The visible numbered list must be backed by
    this pending question so a later ``1`` or ``2`` has a deterministic state
    transition.
    """
    if not explicit_benchmark_intent:
        return _tool_result(
            status="blocked",
            data={
                "reason": "target mode choice requires explicit benchmark intent in the current user turn",
                "source_prompt": source_prompt,
            },
            warnings=[
                "Do not start benchmark setup from a greeting, capability question, pasted evidence, or completed-job discussion."
            ],
            next_actions=[
                "answer the user's opening or capability question without creating benchmark workflow state",
                "ask whether they want latest-job analysis, fake-node benchmark, real-node benchmark, or supported-chain information",
            ],
        )
    source_lower = (source_prompt or "").strip().lower()
    if "fake-node" in source_lower or "real-node" in source_lower:
        return _tool_result(
            status="blocked",
            data={
                "reason": "target mode is already explicit in the current user turn",
                "source_prompt": source_prompt,
            },
            warnings=[
                "Do not ask the fake-node/real-node target-mode menu when the user already named a target mode."
            ],
            next_actions=[
                "set the explicit target mode in workflow state",
                "if chain is missing, call propose_chain_selection_question with the explicit target mode",
            ],
        )
    effective_session_id = _effective_session_id(session_id)
    prompt, options = _target_mode_prompt_and_options(language)
    payload = _update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "workflow_step": "target_mode_choice",
            "assumed_values": {"source_prompt": source_prompt},
            "pending_question": {
                "id": "target_mode",
                "kind": "numbered_choice",
                "prompt": prompt,
                "field": "target_mode",
                "source": "workflow_tool",
                "source_tool": "propose_benchmark_target_mode_choice",
                "branch": "target_selection",
                "workflow_step": "target_mode_choice",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": options,
            },
        },
        reason="propose_benchmark_target_mode_choice",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "target_mode",
            "options": options,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for a numbered choice or manual target mode"],
    )


def propose_quick_assumed_smoke_confirmation(
    source_prompt: str = "",
    chain: str = "solana",
    rpc_mode: str = "single",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the typed confirmation for quick assumed fake-node smoke.

    Use this before showing a compact "just verify the framework can run"
    confirmation. The tool records the exact pending question so the user's
    next short reply is applied by ``answer_pending_question`` rather than by a
    terminal fallback.
    """
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = (chain or "solana").strip().lower() or "solana"
    normalized_rpc_mode = (rpc_mode or "single").strip().lower() or "single"
    if normalized_rpc_mode not in {"single", "mixed"}:
        normalized_rpc_mode = "single"
    prompt = _quick_assumed_smoke_prompt(
        chain=normalized_chain,
        rpc_mode=normalized_rpc_mode,
        language=language,
    )
    payload = _update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "quick_assumed_smoke",
            "workflow_step": "quick_assumed_smoke_confirm",
            "target_mode": "fake-node",
            "chain": normalized_chain,
            "rpc_mode": normalized_rpc_mode,
            "assumed_for_smoke": True,
            "assumed_values": {
                "source_prompt": source_prompt,
                "scope": "smoke-only",
                "chain": normalized_chain,
                "rpc_mode": normalized_rpc_mode,
            },
            "pending_question": {
                "id": "quick_assumed_smoke_confirm",
                "kind": "yes_no",
                "prompt": prompt,
                "field": "approval.quick_assumed_smoke",
                "source": "workflow_tool",
                "source_tool": "propose_quick_assumed_smoke_confirmation",
                "branch": "target_selection",
                "workflow_step": "quick_assumed_smoke_confirm",
                "manual_input_allowed": False,
                "allow_manual_input": False,
                "blocks_execution": True,
                "blocking": True,
                "next_on_yes": {
                    "workflow_step": "quick_assumed_smoke_approved",
                    "tool": "run_quick_assumed_fake_node_smoke",
                    "state_patch": {
                        "approval": {"quick_assumed_smoke": True},
                    },
                },
                "next_on_no": {
                    "workflow_step": "target_selection",
                    "state_patch": {
                        "approval": {"quick_assumed_smoke": False},
                        "assumed_for_smoke": False,
                    },
                },
            },
        },
        reason="propose_quick_assumed_smoke_confirmation",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "chain": normalized_chain,
            "rpc_mode": normalized_rpc_mode,
            "pending_question_id": "quick_assumed_smoke_confirm",
            "assumed_for_smoke": True,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for Y or N"],
    )


def propose_chain_selection_question(
    target_mode: str = "",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the typed chain-selection gate for benchmark setup."""
    effective_session_id = _effective_session_id(session_id)
    normalized_target = (target_mode or "").strip().lower().replace("_", "-")
    if normalized_target not in {"fake-node", "real-node"}:
        normalized_target = ""
    prompt = _chain_selection_prompt(normalized_target, language)
    capabilities = load_framework_capabilities()
    patch: dict[str, Any] = {
        "active_intent": "benchmark",
        "active_workflow": "benchmark_setup",
        "workflow_step": "chain_selection",
        "pending_question": {
            "id": "chain_selection",
            "kind": "chain",
            "prompt": prompt,
            "field": "chain",
            "known_chains": _known_chain_names(capabilities),
            "chain_aliases": _chain_aliases(),
            "source": "workflow_tool",
            "source_tool": "propose_chain_selection_question",
            "branch": "chain_selection",
            "workflow_step": "chain_selection",
            "manual_input_allowed": True,
            "allow_manual_input": True,
            "blocks_execution": True,
            "blocking": True,
            "next_on_manual": {
                "workflow_step": "validate_chain_template",
                "tool": "validate_chain_template",
            },
        },
    }
    if normalized_target:
        patch["target_mode"] = normalized_target
    payload = _update_workflow_state(
        patch,
        reason="propose_chain_selection_question",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "chain_selection",
            "target_mode": normalized_target,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for a chain name or unsupported-chain request"],
    )


def propose_chain_change_confirmation(
    new_chain: str,
    previous_chain: str = "",
    target_mode: str = "",
    target_mode_explicit: bool = False,
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register a typed confirmation before switching the selected chain.

    Use this when the user interrupts an active setup branch with a different
    supported chain. The chain is not committed until the user replies ``Y``.
    """
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = _canonical_chain_name(new_chain)
    normalized_previous = _canonical_chain_name(previous_chain)
    raw_change = _latest_raw_user_change(effective_session_id)
    raw_supported_chain = _supported_chain_from_raw_text(raw_change) if raw_change else ""
    if raw_supported_chain and (
        not normalized_chain
        or normalized_chain == str(new_chain or "").strip().lower().replace("_", "-")
        or normalized_chain not in set(_known_chain_names(load_framework_capabilities()))
    ):
        normalized_chain = raw_supported_chain
    if raw_change and not _raw_text_supports_chain_change(raw_change, normalized_chain):
        payload = _update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "active_group": "chain_target",
                "next_blocking_group": "chain_target",
                "workflow_step": "chain_identity_clarification",
                "chain": "",
                "chain_status": "ambiguous_or_unknown",
                "confirmed_config": {
                    "chain": "",
                    "BLOCKCHAIN_NODE": "",
                },
                "pending_question": {
                    "id": "chain_selection",
                    "kind": "chain",
                    "prompt": _ambiguous_chain_prompt(raw_change, language),
                    "field": "chain",
                    "known_chains": _known_chain_names(load_framework_capabilities()),
                    "chain_aliases": _chain_aliases(),
                    "source": "workflow_tool",
                    "source_tool": "propose_chain_change_confirmation",
                    "branch": "chain_selection",
                    "workflow_step": "chain_identity_clarification",
                    "manual_input_allowed": True,
                    "allow_manual_input": True,
                    "blocks_execution": True,
                    "blocking": True,
                    "next_on_manual": {
                        "workflow_step": "validate_chain_template",
                        "tool": "validate_chain_template",
                    },
                },
                "blockers": [
                    f"chain candidate is ambiguous or not an exact supported-chain alias: {raw_change}"
                ],
            },
            reason="propose_chain_change_confirmation:ambiguous_raw_chain",
            session_id=effective_session_id,
            state_root=_effective_state_root(),
        )
        return _tool_result(
            status="blocked",
            data={
                "prompt": _ambiguous_chain_prompt(raw_change, language),
                "pending_question_id": "chain_selection",
                "raw_user_text": raw_change,
                "proposed_chain": normalized_chain,
                "status": "ambiguous_or_unknown",
            },
            evidence_paths=[payload.get("state_file", "")],
            warnings=[
                "The proposed supported chain was not backed by an exact user-provided chain alias."
            ],
            next_actions=[
                "show the clarification prompt exactly once",
                "if the user names a supported chain, continue Case 1",
                "if the user names a new chain, run chain identity and protocol-family gates before Case 2 or Case 3",
            ],
        )
    normalized_target = (target_mode or "").strip().lower().replace("_", "-")
    inferred_target = _target_mode_from_raw_text(raw_change) if raw_change else ""
    if normalized_target not in {"fake-node", "real-node"} and inferred_target:
        normalized_target = inferred_target
    explicit_target_mode = normalized_target in {"fake-node", "real-node"} and (
        bool(target_mode_explicit) or bool(inferred_target)
    )
    if explicit_target_mode and raw_change:
        if inferred_target and inferred_target != normalized_target:
            explicit_target_mode = False
        elif not inferred_target and not _raw_text_has_target_mode(raw_change, normalized_target):
            explicit_target_mode = False
    if not explicit_target_mode:
        normalized_target = ""
    prompt = _chain_change_confirmation_prompt(normalized_chain, normalized_previous, normalized_target, language)
    yes_patch = {
        "chain": normalized_chain,
        "target_mode": normalized_target,
        "rpc_mode": "",
        "rpc_methods": [],
        "mixed_weights": {},
        "custom_rpc": [],
        "custom_rpc_methods": [],
        "benchmark_profile": {},
        "approval": {"chain_change": True},
        "allowed_next_actions": [],
        "confirmed_config": {
            "chain": normalized_chain,
            "BLOCKCHAIN_NODE": normalized_chain,
            "TARGET_MODE": normalized_target,
            "RPC_MODE": "",
        },
    }
    yes_transition = {
        "workflow_step": "target_mode_choice",
        "next_question_id": "target_mode",
        "state_patch": yes_patch,
    }
    if explicit_target_mode:
        yes_patch["target_mode"] = normalized_target
        yes_patch["confirmed_config"]["use_fake_node"] = normalized_target == "fake-node"
        yes_transition = {
            "workflow_step": "benchmark_setup",
            "state_patch": yes_patch,
        }
    payload = _update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "workflow_step": "confirm_chain_change",
            "pending_question": {
                "id": "confirm_chain_change",
                "kind": "yes_no",
                "prompt": prompt,
                "field": "approval.chain_change",
                "source": "workflow_tool",
                "source_tool": "propose_chain_change_confirmation",
                "branch": "chain_selection",
                "workflow_step": "confirm_chain_change",
                "manual_input_allowed": False,
                "allow_manual_input": False,
                "blocks_execution": True,
                "blocking": True,
                "next_on_yes": yes_transition,
                "next_on_no": {
                    "workflow_step": "chain_selection",
                    "next_question_id": "chain_selection",
                    "state_patch": {
                        "approval": {"chain_change": False},
                        "pending_question": {},
                    },
                },
            },
        },
        reason="propose_chain_change_confirmation",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "confirm_chain_change",
            "new_chain": normalized_chain,
            "previous_chain": normalized_previous,
            "target_mode": normalized_target,
            "target_mode_explicit": explicit_target_mode,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for Y or N"],
    )


def reset_workflow_state(reason: str = "", session_id: str = "") -> dict[str, Any]:
    """Reset structured workflow state when the user explicitly starts over."""
    payload = _reset_workflow_state(
        reason=reason,
        session_id=_effective_session_id(session_id),
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        next_actions=["start a new workflow"],
    )


def revert_workflow_state(
    steps: int = 1,
    reason: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Revert workflow state when the user asks to go back or correct prior input."""
    effective_session_id = _effective_session_id(session_id)
    payload = _revert_workflow_state(
        steps=steps,
        reason=reason,
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        next_actions=["explain reverted values", "re-run validators", "ask the corrected blocking question"],
    )


def propose_unsupported_chain_endpoint_gate(
    chain: str,
    adapter_family: str = "",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the endpoint gate for an unsupported-chain request."""
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = _unsupported_chain_candidate_name(chain, effective_session_id)
    normalized_family = (adapter_family or "unknown").strip().lower() or "unknown"
    raw_change = _latest_raw_user_change(effective_session_id)
    effective_language = _effective_tool_language(language, raw_change)
    prompt = _unsupported_chain_endpoint_prompt(normalized_chain, normalized_family, effective_language)
    payload = _update_workflow_state(
        {
            "active_intent": "chain_onboarding",
            "active_workflow": "unsupported_chain_onboarding",
            "workflow_step": "unsupported_chain_endpoint_gate",
            "chain": normalized_chain,
            "chain_status": "unsupported_needs_review",
            "rpc_mode": "",
            "rpc_methods": [],
            "mixed_weights": {},
            "custom_rpc": [],
            "custom_rpc_methods": [],
            "confirmed_config": {
                "chain": normalized_chain,
                "BLOCKCHAIN_NODE": normalized_chain,
                "RPC_MODE": "",
            },
            "pending_question": {
                "id": "unsupported_chain_endpoint_gate",
                "kind": "url",
                "prompt": prompt,
                "field": "endpoint_validation.rpc_url",
                "source": "workflow_tool",
                "source_tool": "propose_unsupported_chain_endpoint_gate",
                "branch": "unsupported_chain",
                "workflow_step": "unsupported_chain_endpoint_gate",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": _unsupported_chain_endpoint_options(normalized_chain, effective_language),
                "state_patch_on_valid": {
                    "chain_status": "unsupported_needs_endpoint_validation",
                    "fixture_status": {"status": "needs_review"},
                },
                "next_on_manual": {
                    "workflow_step": "validate_unsupported_chain_endpoint",
                    "tool": "validate_rpc_endpoint",
                },
                "next_on_choice": {"workflow_step": "unsupported_chain_endpoint_choice"},
            },
            "fixture_status": {"status": "needs_endpoint"},
            "blockers": ["unsupported chain requires a reachable RPC endpoint before fixtures or smoke"],
        },
        reason="propose_unsupported_chain_endpoint_gate",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "chain": normalized_chain,
            "adapter_family": normalized_family,
            "pending_question_id": "unsupported_chain_endpoint_gate",
            "status": "needs_endpoint",
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for endpoint URL or handoff request"],
    )


def request_unsupported_chain_handoff(
    chain: str = "",
    adapter_family: str = "",
    reason: str = "",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Move unsupported-chain onboarding to a needs_review handoff state.

    Use this when the user wants a development handoff but does not have a
    validated endpoint/request/response evidence yet. This is a workflow state
    transition; it does not classify natural language by itself.
    """
    effective_session_id = _effective_session_id(session_id)
    state = _load_workflow_state(session_id=effective_session_id, state_root=_effective_state_root())
    identity = state.get("chain_identity_candidate") if isinstance(state.get("chain_identity_candidate"), dict) else {}
    protocol = state.get("chain_protocol_candidate") if isinstance(state.get("chain_protocol_candidate"), dict) else {}
    candidate = (
        _normalize_new_chain_candidate(chain)
        or _normalize_new_chain_candidate(state.get("chain"))
        or _normalize_new_chain_candidate(protocol.get("chain"))
        or _normalize_new_chain_candidate(identity.get("normalized"))
        or _normalize_new_chain_candidate(identity.get("raw"))
        or _unsupported_chain_candidate_name("", effective_session_id)
    )
    family = _normalize_adapter_family(
        adapter_family
        or protocol.get("confirmed_adapter_family")
        or protocol.get("suggested_adapter_family")
        or identity.get("suggested_adapter_family")
    )
    visible_reason = _sanitize_visible_evidence_summary(reason, language)
    payload = _update_workflow_state(
        {
            "active_intent": "chain_onboarding",
            "active_workflow": "unsupported_chain_onboarding",
            "active_group": "chain_target",
            "next_blocking_group": "",
            "workflow_step": "unsupported_chain_handoff_requested",
            "chain": candidate,
            "chain_status": "unsupported_needs_development_handoff",
            "confirmed_config": {
                "chain": candidate,
                "BLOCKCHAIN_NODE": candidate,
            },
            "chain_protocol_candidate": {
                "chain": candidate,
                "suggested_adapter_family": family,
                "confirmed_adapter_family": "" if family == "unknown" else family,
            },
            "fixture_status": {
                "status": "needs_review",
                "reason": "no validated endpoint/request/response evidence",
            },
            "pending_question": {},
            "blockers": [
                f"{candidate} requires endpoint/request/response evidence before fixtures or smoke can run"
            ],
        },
        reason="request_unsupported_chain_handoff",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        status="needs_review",
        data={
            "chain": candidate,
            "adapter_family": family,
            "handoff_status": "needs_review",
            "reason": visible_reason,
            "missing_evidence": [
                "reachable endpoint",
                "live request sample",
                "live response sample",
                "fixture recording",
                "fake-node smoke validation",
            ],
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=["needs_review: endpoint/request/response evidence is missing"],
        next_actions=[
            "call build_onboarding_handoff",
            "show an in-chat needs_review handoff draft",
            "do not claim fixtures, fake-node support, or smoke are ready",
        ],
    )


def request_custom_rpc_handoff(
    chain: str = "",
    methods: list[str] | None = None,
    reason: str = "",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Move custom-RPC onboarding to a needs_review handoff state.

    Use this when the user wants to add custom RPC methods but cannot provide
    a reachable endpoint yet. Endpoint probing, fixture recording, and smoke
    validation remain blocked until real evidence is available.
    """
    effective_session_id = _effective_session_id(session_id)
    state = _load_workflow_state(session_id=effective_session_id, state_root=_effective_state_root())
    candidate = _canonical_chain_name(chain) or _canonical_chain_name(state.get("chain")) or str(state.get("chain") or "selected-chain")
    visible_reason = _sanitize_visible_evidence_summary(reason, language)
    method_list = [str(item).strip() for item in (methods or []) if str(item).strip()]
    payload = _update_workflow_state(
        {
            "active_intent": "custom_rpc_onboarding",
            "active_workflow": "custom_rpc_onboarding",
            "active_group": "target_samples_fixtures",
            "next_blocking_group": "",
            "workflow_step": "custom_rpc_handoff_requested",
            "chain": candidate,
            "fixture_status": {
                "status": "needs_review",
                "reason": "custom RPC has no validated endpoint/request/response evidence",
            },
            "pending_question": {},
            "blockers": [
                f"custom RPC for {candidate} requires endpoint/request/response evidence before fixtures or smoke can run"
            ],
        },
        reason="request_custom_rpc_handoff",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        status="needs_review",
        data={
            "chain": candidate,
            "methods": method_list,
            "handoff_status": "needs_review",
            "reason": visible_reason,
            "missing_evidence": [
                "reachable endpoint",
                "exact RPC method name",
                "parameter order/types and TARGET_* samples",
                "live request sample",
                "live response sample",
                "fixture recording",
                "fake-node smoke validation",
            ],
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=["needs_review: custom RPC endpoint/request/response evidence is missing"],
        next_actions=[
            "call build_onboarding_handoff",
            "show an in-chat needs_review handoff draft",
            "do not claim the custom RPC method is executable",
        ],
    )


def propose_chain_identity_resolution(
    candidate_chain: str,
    suggested_supported_chain: str = "",
    suggested_adapter_family: str = "",
    evidence_summary: str = "",
    web_research_used: bool = False,
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the first typed gate for ambiguous or unknown chain identity.

    Use this before Case 2/3 when the user's chain text is not an exact current
    chain template or alias. This gate confirms what chain the user means. It
    must not route directly into endpoint validation or development handoff.
    """
    effective_session_id = _effective_session_id(session_id)
    raw_change = _latest_raw_user_change(effective_session_id)
    effective_language = _effective_tool_language(language, raw_change or candidate_chain)
    visible_evidence = _sanitize_visible_evidence_summary(evidence_summary, effective_language)
    candidate = _normalize_new_chain_candidate(candidate_chain) or _unsupported_chain_candidate_name(raw_change, effective_session_id)
    suggested_chain = _canonical_chain_name(suggested_supported_chain)
    known = set(_known_chain_names(load_framework_capabilities()))
    if suggested_chain not in known:
        suggested_chain = ""
    family = (suggested_adapter_family or "unknown").strip().lower() or "unknown"
    prompt, options = _chain_identity_resolution_prompt_and_options(
        candidate=candidate,
        suggested_chain=suggested_chain,
        suggested_adapter_family=family,
        evidence_summary=visible_evidence,
        web_research_used=web_research_used,
        language=effective_language,
    )
    default_option = options[0] if suggested_chain and options else {}
    payload = _update_workflow_state(
        {
            "active_intent": "chain_identity_resolution",
            "active_workflow": "benchmark_setup",
            "active_group": "chain_target",
            "next_blocking_group": "chain_target",
            "workflow_step": "chain_identity_resolution",
            "chain": "",
            "chain_status": "identity_needs_confirmation",
            "chain_identity_candidate": {
                "raw": candidate_chain or raw_change,
                "normalized": candidate,
                "suggested_supported_chain": suggested_chain,
                "suggested_adapter_family": family,
                "evidence_summary": visible_evidence,
                "web_research_used": bool(web_research_used),
            },
            "confirmed_config": {
                "chain": "",
                "BLOCKCHAIN_NODE": "",
            },
            "pending_question": {
                "id": "chain_identity_resolution",
                "kind": "numbered_choice",
                "prompt": prompt,
                "field": "chain_identity_resolution",
                "source": "workflow_tool",
                "source_tool": "propose_chain_identity_resolution",
                "branch": "chain_selection",
                "workflow_step": "chain_identity_resolution",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": options,
                "default_option": default_option,
                "next_on_manual": {
                    "workflow_step": "chain_selection",
                    "next_question_id": "chain_selection",
                },
            },
            "blockers": [
                f"chain identity must be confirmed before Case 1/2/3 routing: {candidate}"
            ],
        },
        reason="propose_chain_identity_resolution",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "chain_identity_resolution",
            "candidate_chain": candidate,
            "suggested_supported_chain": suggested_chain,
            "suggested_adapter_family": family,
            "web_research_used": bool(web_research_used),
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for the user to confirm the chain identity path"],
    )


def _unsupported_chain_candidate_name(chain: str, session_id: str) -> str:
    proposed = (chain or "unknown-chain").strip().lower() or "unknown-chain"
    state = _load_workflow_state(session_id=session_id, state_root=_effective_state_root())
    change = state.get("last_user_change") or {}
    raw = str(change.get("raw") or "").strip().lower()
    if raw:
        raw_candidate = _candidate_slug_around_proposed(raw, proposed)
        if raw_candidate:
            return raw_candidate
        raw_candidate = _unsupported_candidate_slug_from_raw(raw)
        if raw_candidate:
            return raw_candidate
    return proposed


def _latest_raw_user_change(session_id: str) -> str:
    state = _load_workflow_state(session_id=session_id, state_root=_effective_state_root())
    change = state.get("last_user_change") or {}
    raw = str(change.get("raw") or "").strip()
    return raw


def _effective_tool_language(language: str, raw_user_text: str = "") -> str:
    lang = (language or "").strip().lower()
    if _contains_cjk(raw_user_text):
        return "zh"
    return lang or "en"


def propose_chain_protocol_resolution(
    candidate_chain: str,
    suggested_adapter_family: str = "",
    evidence_summary: str = "",
    web_research_used: bool = False,
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the second typed gate for a confirmed unknown chain.

    The chain identity must already be confirmed before this tool is used. This
    gate asks whether the chain fits one of AnyChain's existing adapter
    families. Only after this confirmation may the Agent enter Case 2 endpoint
    validation or Case 3 development handoff.
    """
    effective_session_id = _effective_session_id(session_id)
    raw_change = _latest_raw_user_change(effective_session_id)
    effective_language = _effective_tool_language(language, raw_change or candidate_chain)
    visible_evidence = _sanitize_visible_evidence_summary(evidence_summary, effective_language)
    candidate = _normalize_new_chain_candidate(candidate_chain) or _unsupported_chain_candidate_name(raw_change, effective_session_id)
    family = _normalize_adapter_family(suggested_adapter_family)
    prompt, options = _chain_protocol_resolution_prompt_and_options(
        candidate=candidate,
        suggested_adapter_family=family,
        evidence_summary=visible_evidence,
        web_research_used=web_research_used,
        language=effective_language,
    )
    payload = _update_workflow_state(
        {
            "active_intent": "chain_protocol_resolution",
            "active_workflow": "benchmark_setup",
            "active_group": "chain_target",
            "next_blocking_group": "chain_target",
            "workflow_step": "chain_protocol_resolution",
            "chain": candidate,
            "chain_status": "protocol_needs_confirmation",
            "chain_protocol_candidate": {
                "chain": candidate,
                "suggested_adapter_family": family,
                "evidence_summary": visible_evidence,
                "web_research_used": bool(web_research_used),
            },
            "confirmed_config": {
                "chain": candidate,
                "BLOCKCHAIN_NODE": candidate,
            },
            "pending_question": {
                "id": "chain_protocol_resolution",
                "kind": "numbered_choice",
                "prompt": prompt,
                "field": "chain_protocol_resolution",
                "source": "workflow_tool",
                "source_tool": "propose_chain_protocol_resolution",
                "branch": "unsupported_chain",
                "workflow_step": "chain_protocol_resolution",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": options,
                "next_on_manual": {
                    "workflow_step": "chain_protocol_resolution",
                    "tool": "propose_chain_protocol_resolution",
                },
            },
            "blockers": [
                f"protocol family must be confirmed before endpoint validation or handoff: {candidate}"
            ],
        },
        reason="propose_chain_protocol_resolution",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "chain_protocol_resolution",
            "candidate_chain": candidate,
            "suggested_adapter_family": family,
            "web_research_used": bool(web_research_used),
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for the user to confirm the protocol family path"],
    )


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _sanitize_visible_evidence_summary(evidence_summary: str, language: str) -> str:
    """Remove internal third-person phrasing before evidence is shown to users."""
    text = " ".join(str(evidence_summary or "").strip().split())
    if not text:
        return ""
    marker = "VISIBLE_RESPONSE:"
    if marker in text:
        text = text.rsplit(marker, 1)[1].strip()
    if any(
        internal in text
        for internal in (
            "FunctionCall(",
            "function_call=",
            "GenerateContentResponseUsageMetadata",
            "model_version=",
            "grounding_metadata=",
        )
    ):
        return ""
    zh = (language or "en").strip().lower().startswith("zh")
    replacements = (
        (
            ("The user says", "用户补充"),
            ("User says", "用户补充"),
            ("the user says", "用户补充"),
            ("The user stated", "用户补充"),
            ("User stated", "用户补充"),
            ("the user stated", "用户补充"),
            ("The user states", "用户补充"),
            ("User states", "用户补充"),
            ("the user states", "用户补充"),
            ("The user indicated", "用户补充"),
            ("User indicated", "用户补充"),
            ("the user indicated", "用户补充"),
            ("The user described", "用户补充"),
            ("User described", "用户补充"),
            ("the user described", "用户补充"),
            ("The user", "用户补充"),
            ("the user", "用户补充"),
            ("I will provide more info", "提供更多信息"),
            ("I will provide more information", "提供更多信息"),
            ("suggesting", "这表明"),
        )
        if zh
        else (
            ("The user says", "User-provided evidence:"),
            ("User says", "User-provided evidence:"),
            ("the user says", "User-provided evidence:"),
            ("The user stated", "User-provided evidence:"),
            ("User stated", "User-provided evidence:"),
            ("the user stated", "User-provided evidence:"),
            ("The user states", "User-provided evidence:"),
            ("User states", "User-provided evidence:"),
            ("the user states", "User-provided evidence:"),
            ("The user indicated", "User-provided evidence:"),
            ("User indicated", "User-provided evidence:"),
            ("the user indicated", "User-provided evidence:"),
            ("The user described", "User-provided evidence:"),
            ("User described", "User-provided evidence:"),
            ("the user described", "User-provided evidence:"),
            ("The user", "User-provided evidence:"),
            ("the user", "User-provided evidence:"),
            ("I will provide more info", "provide more info"),
            ("I will provide more information", "provide more information"),
        )
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _normalize_new_chain_candidate(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "unknown-chain":
        return ""
    tokens = _latin_name_tokens(text)
    if tokens:
        return "-".join(tokens)
    return re.sub(r"[^a-z0-9._:-]+", "-", text).strip("-") or ""


def _raw_text_supports_chain_change(raw: str, normalized_chain: str) -> bool:
    """Validate a model-proposed supported-chain change against user text.

    This is not intent routing. ADK/model has already proposed a supported
    chain. The tool validates that the proposal is backed by an exact scalar
    chain name or alias in the user's latest text. Partial tokens such as
    ``sola`` must not become ``solana``; longer product names such as
    ``bnb greenfield`` must not collapse to ``bsc`` just because they contain
    ``bnb``.
    """
    target = _canonical_chain_name(normalized_chain)
    if not target:
        return False
    exact_supported = _supported_chain_from_raw_text(raw)
    if exact_supported:
        return exact_supported == target
    scalar = canonicalize_chain_scalar(raw)
    if scalar:
        return scalar == target
    tokens = _latin_name_tokens(raw)
    if not tokens:
        return False
    text = " ".join(tokens)
    candidates = _supported_chain_candidates()
    for alias, chain in candidates.items():
        canonical = _canonical_chain_name(chain)
        if canonical != target:
            continue
        alias_tokens = _latin_name_tokens(alias)
        if not alias_tokens:
            continue
        for index in range(0, len(tokens) - len(alias_tokens) + 1):
            if tokens[index : index + len(alias_tokens)] != alias_tokens:
                continue
            before = tokens[:index]
            after = tokens[index + len(alias_tokens) :]
            extra = [item for item in [*before, *after] if item not in FRAMEWORK_CONTEXT_TOKENS]
            if not extra:
                return True
    return False


def _supported_chain_from_raw_text(raw: str) -> str:
    """Return an exact supported-chain alias from a short natural-language turn.

    This is an entity-validation helper for workflow tools. It accepts only a
    single exact alias/name with no adjacent non-framework name tokens. It must
    not collapse names such as ``bnb greenfield`` to ``bsc`` or ``sola`` to
    ``solana``.
    """
    tokens = _latin_name_tokens(raw)
    if not tokens:
        return ""
    aliases = _supported_chain_candidates()
    matches: set[str] = set()
    for alias, chain in aliases.items():
        canonical = _canonical_chain_name(chain)
        alias_tokens = _latin_name_tokens(alias)
        if not canonical or not alias_tokens:
            continue
        for index in range(0, len(tokens) - len(alias_tokens) + 1):
            if tokens[index : index + len(alias_tokens)] != alias_tokens:
                continue
            before = tokens[:index]
            after = tokens[index + len(alias_tokens) :]
            extra = [item for item in [*before, *after] if item not in FRAMEWORK_CONTEXT_TOKENS]
            if not extra:
                matches.add(canonical)
    if len(matches) == 1:
        return next(iter(matches))
    return ""


def _supported_chain_candidates() -> dict[str, str]:
    """Return exact supported chain names plus aliases.

    This helper validates model-proposed chain entities against framework facts.
    It is not fuzzy matching: every key is either a current chain template name
    or an explicit alias such as ``bnb -> bsc``.
    """
    capabilities = load_framework_capabilities()
    known_names = _known_chain_names(capabilities)
    candidates = {name: name for name in known_names}
    candidates.update(canonical_chain_aliases())
    return candidates


def _raw_text_has_target_mode(raw: str, normalized_target: str) -> bool:
    tokens = set(_latin_name_tokens(raw))
    target = (normalized_target or "").strip().lower()
    if target == "fake-node":
        return bool(tokens.intersection({"fake", "fake-node", "fakenode", "mock"}))
    if target == "real-node":
        return bool(tokens.intersection({"real", "real-node", "realnode", "真实节点"})) or "真实" in str(raw)
    return False


def _target_mode_from_raw_text(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    tokens = set(_latin_name_tokens(text))
    fake_mentioned = bool(tokens.intersection({"fake", "fake-node", "fakenode", "mock"}))
    real_mentioned = bool(tokens.intersection({"real", "real-node", "realnode"})) or "真实" in text
    if real_mentioned and not fake_mentioned:
        return "real-node"
    if real_mentioned and fake_mentioned:
        if any(marker in text for marker in ("不使用 fake", "不用 fake", "不要 fake", "not fake", "without fake", "no fake")):
            return "real-node"
        return ""
    if fake_mentioned:
        if any(marker in text for marker in ("不使用 fake", "不用 fake", "不要 fake", "not fake", "without fake", "no fake")):
            return "real-node"
        return "fake-node"
    return ""


def _ambiguous_chain_prompt(raw: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        return (
            f"`{raw}` 不是当前已支持链的精确名称或别名。\n"
            "请确认你要测试的链名：可以输入已支持链名（例如 `solana`、`ethereum`、`bnb`），"
            "也可以输入一个新链名称并提供协议/官方文档/endpoint，我会进入新增链适配流程。"
        )
    return (
        f"`{raw}` is not an exact supported chain name or alias.\n"
        "Confirm the chain to benchmark: type a supported chain such as `solana`, "
        "`ethereum`, or `bnb`, or type a new chain name with protocol/docs/endpoint "
        "details to enter onboarding."
    )


def _chain_identity_resolution_prompt_and_options(
    *,
    candidate: str,
    suggested_chain: str,
    suggested_adapter_family: str,
    evidence_summary: str,
    web_research_used: bool,
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    zh = (language or "en").strip().lower().startswith("zh")
    family = suggested_adapter_family or "unknown"
    evidence = evidence_summary.strip() or ("已使用 Google Search/官方资料辅助判断。" if web_research_used and zh else "Google Search/official-source evidence was used." if web_research_used else "")
    options: list[dict[str, Any]] = []
    if suggested_chain:
        options.append(
            {
                "id": str(len(options) + 1),
                "label": (f"确认这是已支持链 `{suggested_chain}`" if zh else f"Confirm this is supported chain `{suggested_chain}`"),
                "value": "confirm_supported_chain",
                "state_patch": {
                    "active_intent": "benchmark",
                    "active_workflow": "benchmark_setup",
                    "active_group": "chain_target",
                    "next_blocking_group": "",
                    "chain": suggested_chain,
                    "chain_status": "supported_confirmed",
                    "confirmed_config": {
                        "chain": suggested_chain,
                        "BLOCKCHAIN_NODE": suggested_chain,
                    },
                    "blockers": [],
                },
                "transition": {"workflow_step": "benchmark_setup"},
            }
        )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": (
                f"确认 `{candidate}` 是一个新链，下一步确认协议 family"
                if zh
                else f"Confirm `{candidate}` is a new chain; next confirm its protocol family"
            ),
            "value": "confirm_new_chain_identity",
            "state_patch": {
                "active_intent": "chain_protocol_resolution",
                "active_workflow": "benchmark_setup",
                "active_group": "chain_target",
                "workflow_step": "chain_protocol_resolution",
                "chain": candidate,
                "chain_status": "protocol_needs_confirmation",
                "chain_protocol_candidate": {
                    "chain": candidate,
                    "suggested_adapter_family": family,
                    "evidence_summary": evidence,
                    "web_research_used": bool(web_research_used),
                },
                "confirmed_config": {
                    "chain": candidate,
                    "BLOCKCHAIN_NODE": candidate,
                },
                "pending_question": {
                    "id": "chain_protocol_resolution",
                    "kind": "numbered_choice",
                    "prompt": _chain_protocol_resolution_prompt(
                        candidate=candidate,
                        suggested_adapter_family=family,
                        evidence_summary=evidence,
                        web_research_used=web_research_used,
                        language=language,
                    ),
                    "field": "chain_protocol_resolution",
                    "source": "workflow_tool",
                    "source_tool": "propose_chain_identity_resolution",
                    "branch": "unsupported_chain",
                    "workflow_step": "chain_protocol_resolution",
                    "manual_input_allowed": True,
                    "allow_manual_input": True,
                    "blocks_execution": True,
                    "blocking": True,
                    "options": _chain_protocol_resolution_options(candidate, family, language),
                    "next_on_manual": {"workflow_step": "chain_protocol_resolution"},
                },
                "blockers": [f"protocol family must be confirmed before endpoint validation or handoff: {candidate}"],
            },
            "transition": {"workflow_step": "chain_protocol_resolution"},
        }
    )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": (
                "暂不确认这个链存在；提供官方文档或更多信息"
                if zh
                else "Chain existence is not confirmed; provide docs or more evidence"
            ),
            "value": "needs_chain_identity_evidence",
            "state_patch": {
                "active_intent": "chain_identity_resolution",
                "active_workflow": "benchmark_setup",
                "active_group": "chain_target",
                "workflow_step": "chain_identity_needs_evidence",
                "chain": "",
                "chain_status": "identity_needs_evidence",
                "blockers": [f"{candidate} requires chain identity evidence before protocol routing"],
            },
            "transition": {"workflow_step": "chain_identity_needs_evidence"},
        }
    )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": "输入正确链名" if zh else "Type a corrected chain name",
            "value": "correct_chain_name",
            "state_patch": {
                "chain": "",
                "chain_status": "unknown",
                "confirmed_config": {"chain": "", "BLOCKCHAIN_NODE": ""},
            },
            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
        }
    )
    if zh:
        lines = [
            f"`{candidate}` 不在当前 36 条链中，需要先确认链身份，不能直接进入 benchmark。",
        ]
        if suggested_chain:
            lines.append(f"LLM/资料判断它可能是已支持链 `{suggested_chain}` 的输入错误。")
        else:
            lines.append(f"当前无法把它精确匹配到已支持链；协议候选为 `{family}`。")
        if evidence:
            lines.append(f"依据：{evidence}")
        lines.append("请选择下一步：")
        lines.extend(f"{option['id']}. {option['label']}" for option in options)
        lines.append("回复编号，或直接输入正确链名。")
    else:
        lines = [
            f"`{candidate}` is not one of the current 36 chain templates. Confirm the chain identity before benchmarking.",
        ]
        if suggested_chain:
            lines.append(f"The LLM/source evidence suggests it may be a typo for supported chain `{suggested_chain}`.")
        else:
            lines.append(f"It cannot be exactly matched to a supported chain; adapter-family candidate is `{family}`.")
        if evidence:
            lines.append(f"Evidence: {evidence}")
        lines.append("Choose the next step:")
        lines.extend(f"{option['id']}. {option['label']}" for option in options)
        lines.append("Reply with a number, or type the corrected chain name.")
    return "\n".join(lines), options


def _chain_protocol_resolution_prompt_and_options(
    *,
    candidate: str,
    suggested_adapter_family: str,
    evidence_summary: str,
    web_research_used: bool,
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    return (
        _chain_protocol_resolution_prompt(
            candidate=candidate,
            suggested_adapter_family=suggested_adapter_family,
            evidence_summary=evidence_summary,
            web_research_used=web_research_used,
            language=language,
        ),
        _chain_protocol_resolution_options(candidate, suggested_adapter_family, language),
    )


def _chain_protocol_resolution_prompt(
    *,
    candidate: str,
    suggested_adapter_family: str,
    evidence_summary: str,
    web_research_used: bool,
    language: str,
) -> str:
    zh = (language or "en").strip().lower().startswith("zh")
    family = _normalize_adapter_family(suggested_adapter_family)
    evidence = evidence_summary.strip() or (
        "已使用 Google Search/官方资料辅助判断。"
        if web_research_used and zh
        else "Google Search/official-source evidence was used."
        if web_research_used
        else ""
    )
    if zh:
        lines = [
            f"已确认链身份为 `{candidate}`。现在需要确认它是否属于 AnyChain 已支持的协议 family。",
        ]
        if family != "unknown":
            lines.append(f"协议候选：`{family}`。")
        else:
            lines.append("当前没有可靠的已支持协议候选。")
        if evidence:
            lines.append(f"依据：{evidence}")
        lines.append("请选择下一步：")
    else:
        lines = [
            f"Chain identity is confirmed as `{candidate}`. Now confirm whether it fits an AnyChain supported adapter family.",
        ]
        if family != "unknown":
            lines.append(f"Adapter-family candidate: `{family}`.")
        else:
            lines.append("No reliable supported adapter-family candidate is available yet.")
        if evidence:
            lines.append(f"Evidence: {evidence}")
        lines.append("Choose the next step:")
    options = _chain_protocol_resolution_options(candidate, family, language)
    lines.extend(f"{option['id']}. {option['label']}" for option in options)
    if zh:
        lines.append("回复编号，或直接输入正确的协议 family / 官方文档信息。")
    else:
        lines.append("Reply with a number, or type the correct protocol family / official-doc evidence.")
    return "\n".join(lines)


def _chain_protocol_resolution_options(candidate: str, adapter_family: str, language: str) -> list[dict[str, Any]]:
    zh = (language or "en").strip().lower().startswith("zh")
    family = _normalize_adapter_family(adapter_family)
    options: list[dict[str, Any]] = []
    if family != "unknown":
        endpoint_prompt = _unsupported_chain_endpoint_prompt(candidate, family, language)
        options.append(
            {
                "id": str(len(options) + 1),
                "label": (
                    f"确认属于 `{family}` family，继续 endpoint/RPC 验证"
                    if zh
                    else f"Confirm `{family}` family; continue endpoint/RPC validation"
                ),
                "value": "confirm_existing_family",
                "state_patch": {
                    "active_intent": "chain_onboarding",
                    "active_workflow": "unsupported_chain_onboarding",
                    "active_group": "chain_target",
                    "workflow_step": "unsupported_chain_endpoint_gate",
                    "chain": candidate,
                    "chain_status": "unsupported_needs_endpoint_validation",
                    "confirmed_config": {
                        "chain": candidate,
                        "BLOCKCHAIN_NODE": candidate,
                        "RPC_MODE": "",
                    },
                    "chain_protocol_candidate": {
                        "chain": candidate,
                        "suggested_adapter_family": family,
                        "confirmed_adapter_family": family,
                    },
                    "fixture_status": {"status": "needs_endpoint"},
                    "blockers": ["new chain requires reachable endpoint and RPC request/response evidence before fixtures or smoke"],
                    "pending_question": {
                        "id": "unsupported_chain_endpoint_gate",
                        "kind": "url",
                        "prompt": endpoint_prompt,
                        "field": "endpoint_validation.rpc_url",
                        "source": "workflow_tool",
                        "source_tool": "propose_chain_protocol_resolution",
                        "branch": "unsupported_chain",
                        "workflow_step": "unsupported_chain_endpoint_gate",
                        "manual_input_allowed": True,
                        "allow_manual_input": True,
                        "blocks_execution": True,
                        "blocking": True,
                        "options": _unsupported_chain_endpoint_options(candidate, language),
                        "state_patch_on_valid": {
                            "chain_status": "unsupported_needs_endpoint_validation",
                            "fixture_status": {"status": "needs_review"},
                        },
                        "next_on_manual": {
                            "workflow_step": "validate_unsupported_chain_endpoint",
                            "tool": "validate_rpc_endpoint",
                        },
                        "next_on_choice": {"workflow_step": "unsupported_chain_endpoint_choice"},
                    },
                },
                "transition": {"workflow_step": "unsupported_chain_endpoint_gate"},
            }
        )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": (
                "不属于现有协议，生成 needs_review 二次开发交接"
                if zh
                else "Not in an existing family; generate a needs_review development handoff"
            ),
            "value": "unsupported_family_handoff",
            "state_patch": {
                "active_intent": "chain_onboarding",
                "active_workflow": "unsupported_chain_onboarding",
                "active_group": "chain_target",
                "workflow_step": "unsupported_chain_handoff_requested",
                "chain": candidate,
                "chain_status": "unsupported_needs_development_handoff",
                "chain_protocol_candidate": {
                    "chain": candidate,
                    "suggested_adapter_family": family,
                    "confirmed_adapter_family": "",
                },
                "fixture_status": {"status": "needs_review", "reason": "adapter family unsupported or unconfirmed"},
                "blockers": [f"{candidate} requires adapter-family development before fixtures or smoke can run"],
            },
            "transition": {"workflow_step": "unsupported_chain_handoff_requested", "tool": "build_onboarding_handoff"},
        }
    )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": "输入正确协议 family 或补充官方文档" if zh else "Type the correct protocol family or add official-doc evidence",
            "value": "correct_protocol_family",
            "state_patch": {
                "chain_status": "protocol_needs_confirmation",
            },
            "transition": {"workflow_step": "chain_protocol_resolution"},
        }
    )
    options.append(
        {
            "id": str(len(options) + 1),
            "label": "更换链" if zh else "Change chain instead",
            "value": "change_chain",
            "state_patch": {
                "chain": "",
                "chain_status": "unknown",
                "confirmed_config": {"chain": "", "BLOCKCHAIN_NODE": ""},
            },
            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
        }
    )
    return options


def _normalize_adapter_family(value: str) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    if family in {"evm", "ethereum", "eth", "json_rpc", "json-rpc"}:
        family = "jsonrpc"
    if family in SUPPORTED_ADAPTER_FAMILIES:
        return family
    return "unknown"


def _unsupported_chain_endpoint_options(chain: str, language: str = "en") -> list[dict[str, Any]]:
    zh = (language or "en").strip().lower().startswith("zh")
    return [
        {
            "id": "1",
            "label": "我有 endpoint，现在提供" if zh else "I have an endpoint and will paste it now",
            "value": "provide_endpoint",
            "transition": {"workflow_step": "unsupported_chain_endpoint_gate"},
        },
        {
            "id": "2",
            "label": "暂时没有 endpoint，生成 needs_review 开发交接" if zh else "No endpoint now; generate a needs_review development handoff",
            "value": "needs_review_handoff",
            "state_patch": {
                "chain_status": "unsupported_needs_development_handoff",
                "fixture_status": {"status": "needs_review", "reason": "no endpoint provided"},
                "blockers": [f"{chain} requires endpoint/request/response evidence before fixtures or smoke can run"],
            },
            "transition": {"workflow_step": "unsupported_chain_handoff_requested", "tool": "build_onboarding_handoff"},
        },
        {
            "id": "3",
            "label": "更换链" if zh else "Change chain instead",
            "value": "change_chain",
            "state_patch": {
                "chain": "",
                "chain_status": "unknown",
                "confirmed_config": {"chain": "", "BLOCKCHAIN_NODE": ""},
            },
            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
        },
    ]


def _candidate_slug_around_proposed(raw: str, proposed: str) -> str:
    proposed_tokens = _latin_name_tokens(proposed)
    raw_tokens = _latin_name_tokens(raw)
    if not proposed_tokens or not raw_tokens:
        return ""
    width = len(proposed_tokens)
    for idx in range(0, len(raw_tokens) - width + 1):
        if raw_tokens[idx : idx + width] != proposed_tokens:
            continue
        start = idx
        if idx > 0 and raw_tokens[idx - 1] not in FRAMEWORK_CONTEXT_TOKENS:
            start = idx - 1
        end = idx + width
        if end < len(raw_tokens) and raw_tokens[end] not in FRAMEWORK_CONTEXT_TOKENS:
            end += 1
        candidate = "-".join(raw_tokens[start:end]).strip("-")
        return candidate or "-".join(proposed_tokens)
    return ""


def _unsupported_candidate_slug_from_raw(raw: str) -> str:
    tokens = [
        token
        for token in _latin_name_tokens(raw)
        if token not in FRAMEWORK_CONTEXT_TOKENS
    ]
    if not tokens:
        return ""
    aliases = canonical_chain_aliases()
    supported_alias_positions = [
        index
        for index, token in enumerate(tokens)
        if token in aliases
    ]
    if supported_alias_positions:
        index = supported_alias_positions[0]
        start = index
        while start > 0 and tokens[start - 1] not in aliases:
            start -= 1
        end = index + 1
        while end < len(tokens) and tokens[end] not in aliases:
            end += 1
        candidate = "-".join(tokens[start:end]).strip("-")
        if candidate:
            return candidate
    return "-".join(tokens).strip("-")


def _latin_name_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text or "").lower())



def propose_custom_rpc_endpoint_gate(
    chain: str,
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the endpoint gate for a custom RPC method request."""
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = (chain or "").strip().lower() or "selected-chain"
    prompt = _custom_rpc_endpoint_prompt(normalized_chain, language)
    payload = _update_workflow_state(
        {
            "active_intent": "custom_rpc_onboarding",
            "active_workflow": "custom_rpc_onboarding",
            "workflow_step": "custom_rpc_endpoint_gate",
            "chain": normalized_chain,
            "pending_question": {
                "id": "custom_rpc_endpoint_gate",
                "kind": "url",
                "prompt": prompt,
                "field": "endpoint_validation.rpc_url",
                "source": "workflow_tool",
                "source_tool": "propose_custom_rpc_endpoint_gate",
                "branch": "custom_rpc",
                "workflow_step": "custom_rpc_endpoint_gate",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "state_patch_on_valid": {
                    "fixture_status": {"status": "needs_review"},
                },
                "next_on_manual": {
                    "workflow_step": "validate_custom_rpc_endpoint",
                    "tool": "validate_rpc_endpoint",
                },
            },
            "fixture_status": {"status": "needs_endpoint"},
            "blockers": ["custom RPC requires a reachable endpoint before method validation or fixture recording"],
        },
        reason="propose_custom_rpc_endpoint_gate",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "chain": normalized_chain,
            "pending_question_id": "custom_rpc_endpoint_gate",
            "status": "needs_endpoint",
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for endpoint URL or explain needs_review if unavailable"],
    )


def propose_real_node_endpoint_gate(
    chain: str,
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the typed LOCAL_RPC_URL gate for real-node benchmarks."""
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = (chain or "selected chain").strip().lower()
    prompt = _real_node_endpoint_prompt(normalized_chain, language)
    payload = _update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "chain": normalized_chain,
            "target_mode": "real-node",
            "workflow_step": "real_node_local_rpc_url",
            "missing_fields": ["local_rpc_url"],
            "pending_question": {
                "id": "real_node_local_rpc_url",
                "kind": "url",
                "prompt": prompt,
                "field": "LOCAL_RPC_URL",
                "source": "workflow_tool",
                "source_tool": "propose_real_node_endpoint_gate",
                "branch": "real_node_endpoint",
                "workflow_step": "real_node_local_rpc_url",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "state_patch_on_valid": {
                    "target_mode": "real-node",
                    "chain": normalized_chain,
                },
                "next_on_manual": {
                    "workflow_step": "validate_real_node_endpoint",
                    "tool": "validate_rpc_endpoint",
                },
            },
        },
        reason="propose_real_node_endpoint_gate",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "chain": normalized_chain,
            "pending_question_id": "real_node_local_rpc_url",
            "status": "needs_local_rpc_url",
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for LOCAL_RPC_URL or explain blockers if unavailable"],
    )


def propose_disk_device_choice(
    device_role: str,
    prompt: str,
    device_options: list[str] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Register a typed disk-device choice for ledger or accounts volumes."""
    effective_session_id = _effective_session_id(session_id)
    role = (device_role or "").strip().lower()
    if role not in {"ledger", "accounts"}:
        role = "ledger"
    question_id = "disk_accounts_choice" if role == "accounts" else "disk_ledger_choice"
    field = "ACCOUNTS_DEVICE" if role == "accounts" else "LEDGER_DEVICE"
    options = _device_choice_options(device_options or [], include_none=(role == "accounts"))
    visible_prompt = _device_choice_prompt(role, options, prompt)
    payload = _update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "workflow_step": question_id,
            "pending_question": {
                "id": question_id,
                "kind": "device",
                "prompt": visible_prompt,
                "field": field,
                "source": "workflow_tool",
                "source_tool": "propose_disk_device_choice",
                "branch": "environment_confirmation",
                "workflow_step": question_id,
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": role == "ledger",
                "blocking": role == "ledger",
                "options": options,
                "next_on_manual": {"workflow_step": f"{role}_device_manual_review"},
                "next_on_choice": {"workflow_step": f"{role}_device_confirmed"},
            },
        },
        reason=f"propose_disk_device_choice:{role}",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": visible_prompt,
            "pending_question_id": question_id,
            "options": options,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for device choice, manual device, or back"],
    )


def propose_benchmark_profile_choice(
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the quick / standard / intensive benchmark profile choice."""
    effective_session_id = _effective_session_id(session_id)
    prompt, options = _benchmark_profile_prompt_and_options(language)
    payload = _update_workflow_state(
        {
            "workflow_step": "benchmark_profile_choice",
            "pending_question": {
                "id": "benchmark_profile_choice",
                "kind": "numbered_choice",
                "prompt": prompt,
                "source": "workflow_tool",
                "source_tool": "propose_benchmark_profile_choice",
                "branch": "benchmark_profile",
                "workflow_step": "benchmark_profile_choice",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": options,
            },
        },
        reason="propose_benchmark_profile_choice",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "benchmark_profile_choice",
            "options": options,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for profile choice or manual override"],
    )


def propose_proceed_discovery_confirmation(
    prompt: str = "",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register a typed yes/no gate before discovery, preflight, and smoke."""
    effective_session_id = _effective_session_id(session_id)
    visible_prompt = _proceed_discovery_prompt(language, summary=prompt)
    payload = _update_workflow_state(
        {
            "workflow_step": "proceed_discovery_confirmation",
            "pending_question": {
                "id": "proceed_discovery",
                "kind": "yes_no",
                "prompt": visible_prompt,
                "source": "workflow_tool",
                "source_tool": "propose_proceed_discovery_confirmation",
                "branch": "benchmark_setup",
                "workflow_step": "proceed_discovery_confirmation",
                "manual_input_allowed": False,
                "allow_manual_input": False,
                "blocks_execution": True,
                "blocking": True,
                "next_on_yes": {
                    "workflow_step": "discovery_requested",
                    "tool": "run_environment_discovery",
                },
                "next_on_no": {
                    "workflow_step": "configuration_paused",
                    "tool": "ask_configuration_change",
                },
            },
        },
        reason="propose_proceed_discovery_confirmation",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": visible_prompt,
            "pending_question_id": "proceed_discovery",
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for Y or N"],
    )


def propose_workload_customization_choice(
    chain: str = "",
    rpc_mode: str = "mixed",
    language: str = "en",
    session_id: str = "",
) -> dict[str, Any]:
    """Register the default workload / custom RPC / weight adjustment choice."""
    effective_session_id = _effective_session_id(session_id)
    normalized_chain = (chain or "selected chain").strip().lower()
    prompt, options = _workload_customization_prompt_and_options(normalized_chain, rpc_mode, language)
    payload = _update_workflow_state(
        {
            "workflow_step": "workload_customization_choice",
            "pending_question": {
                "id": "workload_customization_choice",
                "kind": "numbered_choice",
                "prompt": prompt,
                "source": "workflow_tool",
                "source_tool": "propose_workload_customization_choice",
                "branch": "rpc_workload",
                "workflow_step": "workload_customization_choice",
                "manual_input_allowed": True,
                "allow_manual_input": True,
                "blocks_execution": True,
                "blocking": True,
                "options": options,
            },
        },
        reason="propose_workload_customization_choice",
        session_id=effective_session_id,
        state_root=_effective_state_root(),
    )
    return _tool_result(
        data={
            "prompt": prompt,
            "pending_question_id": "workload_customization_choice",
            "options": options,
        },
        evidence_paths=[payload.get("state_file", "")],
        warnings=payload.get("validation_warnings", []),
        next_actions=["show the prompt exactly once", "wait for workload choice or manual override"],
    )


def get_workflow_state_tools() -> list:
    """Return ADK workflow-state tool callables."""
    return [
        load_workflow_state,
        update_workflow_state,
        propose_opening_help_choice,
        answer_pending_question,
        record_group_jump,
        recompute_next_blocking_group,
        propose_benchmark_target_mode_choice,
        propose_quick_assumed_smoke_confirmation,
        propose_chain_selection_question,
        propose_chain_identity_resolution,
        propose_chain_protocol_resolution,
        request_unsupported_chain_handoff,
        request_custom_rpc_handoff,
        propose_chain_change_confirmation,
        propose_unsupported_chain_endpoint_gate,
        propose_custom_rpc_endpoint_gate,
        propose_real_node_endpoint_gate,
        propose_disk_device_choice,
        propose_benchmark_profile_choice,
        propose_proceed_discovery_confirmation,
        propose_workload_customization_choice,
        revert_workflow_state,
        reset_workflow_state,
    ]


def _opening_help_prompt_and_options(language: str, latest_job_id: str = "") -> tuple[str, list[dict[str, Any]]]:
    normalized_language = (language or "en").strip().lower()
    latest_label = latest_job_id.strip()
    if normalized_language.startswith("zh"):
        if not latest_label:
            prompt = (
                "你好，我是 AnyChain Benchmark Agent。当前没有历史任务可分析。你想让我帮你做什么？\n"
                "1. 启动 fake-node 测试（无需真实节点，验证框架闭环）\n"
                "2. 启动 real-node 测试（需要真实 LOCAL_RPC_URL）\n"
                "3. 了解支持的链、RPC method 和二次开发方式\n"
                "回复 `1`、`2` 或 `3`，也可以直接输入你的目标。"
            )
            return prompt, _opening_help_benchmark_options(language, offset=0)
        prompt = (
            "你好，我是 AnyChain Benchmark Agent。你想让我帮你做什么？\n"
            f"1. 查看或分析最近一次任务（{latest_label}）\n"
            "2. 启动 fake-node 测试（无需真实节点，验证框架闭环）\n"
            "3. 启动 real-node 测试（需要真实 LOCAL_RPC_URL）\n"
            "4. 了解支持的链、RPC method 和二次开发方式\n"
            "回复 `1`、`2`、`3` 或 `4`，也可以直接输入你的目标。"
        )
        options = [
            {
                "id": "1",
                "label": f"查看或分析最近一次任务（{latest_label}）",
                "value": "latest_job",
                "state_patch": {"active_intent": "job_resume", "workflow_step": "job_resume"},
                "transition": {"workflow_step": "job_resume", "tool": "latest_job"},
            },
            {
                "id": "2",
                "label": "启动 fake-node 测试（无需真实节点，验证框架闭环）",
                "value": "fake-node",
                "state_patch": {
                    "active_intent": "benchmark",
                    "active_workflow": "benchmark_setup",
                    "target_mode": "fake-node",
                    "workflow_step": "benchmark_setup",
                },
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
            {
                "id": "3",
                "label": "启动 real-node 测试（需要真实 LOCAL_RPC_URL）",
                "value": "real-node",
                "state_patch": {
                    "active_intent": "benchmark",
                    "active_workflow": "benchmark_setup",
                    "target_mode": "real-node",
                    "workflow_step": "benchmark_setup",
                },
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
            {
                "id": "4",
                "label": "了解支持的链、RPC method 和二次开发方式",
                "value": "capabilities",
                "state_patch": {"active_intent": "framework_capability_question", "workflow_step": "explain_capabilities"},
                "transition": {"workflow_step": "explain_capabilities", "tool": "load_framework_context"},
            },
        ]
        return prompt, options
    if not latest_label:
        prompt = (
            "Hi, I am AnyChain Benchmark Agent. No previous job is available to analyze. What would you like to do?\n"
            "1. Start a fake-node benchmark (no real node required; validates the framework loop)\n"
            "2. Start a real-node benchmark (requires a real LOCAL_RPC_URL)\n"
            "3. Learn supported chains, RPC methods, and extension paths\n"
            "Reply `1`, `2`, or `3`, or type your goal directly."
        )
        return prompt, _opening_help_benchmark_options(language, offset=0)
    prompt = (
        "Hi, I am AnyChain Benchmark Agent. What would you like to do?\n"
        f"1. View or analyze the latest job ({latest_label})\n"
        "2. Start a fake-node benchmark (no real node required; validates the framework loop)\n"
        "3. Start a real-node benchmark (requires a real LOCAL_RPC_URL)\n"
        "4. Learn supported chains, RPC methods, and extension paths\n"
        "Reply `1`, `2`, `3`, or `4`, or type your goal directly."
    )
    options = [
        {
            "id": "1",
            "label": f"View or analyze the latest job ({latest_label})",
            "value": "latest_job",
            "state_patch": {"active_intent": "job_resume", "workflow_step": "job_resume"},
            "transition": {"workflow_step": "job_resume", "tool": "latest_job"},
        },
        {
            "id": "2",
            "label": "Start a fake-node benchmark (no real node required; validates the framework loop)",
            "value": "fake-node",
            "state_patch": {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "workflow_step": "benchmark_setup",
            },
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": "3",
            "label": "Start a real-node benchmark (requires a real LOCAL_RPC_URL)",
            "value": "real-node",
            "state_patch": {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "real-node",
                "workflow_step": "benchmark_setup",
            },
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": "4",
            "label": "Learn supported chains, RPC methods, and extension paths",
            "value": "capabilities",
            "state_patch": {"active_intent": "framework_capability_question", "workflow_step": "explain_capabilities"},
            "transition": {"workflow_step": "explain_capabilities", "tool": "load_framework_context"},
        },
    ]
    return prompt, options


def _opening_help_benchmark_options(language: str, offset: int = 0) -> list[dict[str, Any]]:
    normalized_language = (language or "en").strip().lower()
    zh = normalized_language.startswith("zh")
    fake_id = str(offset + 1)
    real_id = str(offset + 2)
    caps_id = str(offset + 3)
    return [
        {
            "id": fake_id,
            "label": "启动 fake-node 测试（无需真实节点，验证框架闭环）" if zh else "Start a fake-node benchmark (no real node required; validates the framework loop)",
            "value": "fake-node",
            "state_patch": {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "workflow_step": "benchmark_setup",
            },
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": real_id,
            "label": "启动 real-node 测试（需要真实 LOCAL_RPC_URL）" if zh else "Start a real-node benchmark (requires a real LOCAL_RPC_URL)",
            "value": "real-node",
            "state_patch": {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "real-node",
                "workflow_step": "benchmark_setup",
            },
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": caps_id,
            "label": "了解支持的链、RPC method 和二次开发方式" if zh else "Learn supported chains, RPC methods, and extension paths",
            "value": "capabilities",
            "state_patch": {"active_intent": "framework_capability_question", "workflow_step": "explain_capabilities"},
            "transition": {"workflow_step": "explain_capabilities", "tool": "load_framework_context"},
        },
    ]


def _target_mode_prompt_and_options(language: str) -> tuple[str, list[dict[str, Any]]]:
    normalized_language = (language or "en").strip().lower()
    if normalized_language.startswith("zh"):
        prompt = (
            "你想怎么测试？\n"
            "1. fake-node 模式 — 不需要真实节点，验证框架闭环流程\n"
            "2. real-node 模式 — 使用真实节点 RPC，测试真实节点性能\n"
            "3. 先了解 AnyChain Benchmark Agent 支持什么\n"
            "回复 `1`、`2` 或 `3`，也可以直接输入其他目标。"
        )
        options = [
            {
                "id": "1",
                "label": "fake-node 模式 — 不需要真实节点，验证框架闭环流程",
                "value": "fake-node",
                "state_patch": {"target_mode": "fake-node", "workflow_step": "benchmark_setup"},
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
            {
                "id": "2",
                "label": "real-node 模式 — 使用真实节点 RPC，测试真实节点性能",
                "value": "real-node",
                "state_patch": {"target_mode": "real-node", "workflow_step": "benchmark_setup"},
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
            {
                "id": "3",
                "label": "先了解 AnyChain Benchmark Agent 支持什么",
                "value": "explain",
                "state_patch": {"active_intent": "framework_capability_question", "workflow_step": "explain_capabilities"},
                "transition": {"workflow_step": "explain_capabilities", "tool": "load_framework_context"},
            },
        ]
        return prompt, options
    prompt = (
        "How do you want to test?\n"
        "1. fake-node mode — no real node required; verify the framework closed loop\n"
        "2. real-node mode — use a real node RPC endpoint and benchmark real performance\n"
        "3. First explain what AnyChain Benchmark Agent supports\n"
        "Reply `1`, `2`, or `3`, or type your own target."
    )
    options = [
        {
            "id": "1",
            "label": "fake-node mode — no real node required; verify the framework closed loop",
            "value": "fake-node",
            "state_patch": {"target_mode": "fake-node", "workflow_step": "benchmark_setup"},
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": "2",
            "label": "real-node mode — use a real node RPC endpoint and benchmark real performance",
            "value": "real-node",
            "state_patch": {"target_mode": "real-node", "workflow_step": "benchmark_setup"},
            "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
        },
        {
            "id": "3",
            "label": "First explain what AnyChain Benchmark Agent supports",
            "value": "explain",
            "state_patch": {"active_intent": "framework_capability_question", "workflow_step": "explain_capabilities"},
            "transition": {"workflow_step": "explain_capabilities", "tool": "load_framework_context"},
        },
    ]
    return prompt, options


def _chain_selection_prompt(target_mode: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        mode_text = f"（当前目标模式：{target_mode}）" if target_mode else ""
        return (
            f"你想测试哪条链？{mode_text}\n"
            "请输入链名，例如 `solana`、`ethereum`、`bitcoin`。"
            "如果不是当前 36 条链，也可以直接输入链名，我会进入新增链/协议适配流程。"
        )
    mode_text = f" (current target mode: {target_mode})" if target_mode else ""
    return (
        f"Which chain do you want to benchmark?{mode_text}\n"
        "Type a chain name such as `solana`, `ethereum`, or `bitcoin`. "
        "If it is not one of the current 36 chains, type the chain name to enter the chain onboarding flow."
    )


def _known_chain_names(capabilities: dict[str, Any]) -> list[str]:
    names = []
    for item in list((capabilities or {}).get("chains") or []):
        if isinstance(item, dict):
            name = str(item.get("chain") or "").strip().lower()
            if name:
                names.append(name)
    return sorted(set(names))


def _chain_aliases() -> dict[str, str]:
    return canonical_chain_aliases()


def _chain_change_confirmation_prompt(new_chain: str, previous_chain: str, target_mode: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        previous_text = f"从 `{previous_chain}` " if previous_chain else ""
        if not target_mode:
            return (
                f"是否确认{previous_text}切换到 `{new_chain}`？\n"
                "回复 `Y` 确认切换并选择 fake-node/real-node；回复 `N` 返回链选择。"
            )
        return (
            f"是否确认{previous_text}切换到 `{new_chain}`，并继续使用 `{target_mode}` 模式？\n"
            "回复 `Y` 确认切换；回复 `N` 返回链选择。"
        )
    previous_text = f"from `{previous_chain}` " if previous_chain else ""
    if not target_mode:
        return (
            f"Confirm switching {previous_text}to `{new_chain}`?\n"
            "Reply `Y` to switch and choose fake-node/real-node, or `N` to return to chain selection."
        )
    return (
        f"Confirm switching {previous_text}to `{new_chain}` and continuing in `{target_mode}` mode?\n"
        "Reply `Y` to switch, or `N` to return to chain selection."
    )


def _canonical_chain_name(value: str) -> str:
    lowered = str(value or "").strip().lower().replace("_", "-")
    return canonicalize_chain_scalar(lowered) or lowered


def _device_choice_options(device_options: list[str], include_none: bool = False) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for index, raw in enumerate(device_options, start=1):
        label = str(raw or "").strip()
        if not label:
            continue
        if label.strip().lower() in {"manual", "manual input", "custom", "custom path", "手动", "手动输入", "自定义", "自定义路径"}:
            continue
        value = _device_value_from_label(label)
        options.append(
            {
                "id": str(index),
                "label": label,
                "value": value,
                "state_patch": {},
            }
        )
    if include_none:
        options.append(
            {
                "id": str(len(options) + 1),
                "label": "无 accounts 设备",
                "value": "",
                "state_patch": {"confirmed_config": {"ACCOUNTS_DEVICE": ""}},
            }
        )
    return options


def _device_choice_prompt(role: str, options: list[dict[str, Any]], source_prompt: str = "") -> str:
    language = "zh" if any("\u3400" <= ch <= "\u9fff" for ch in str(source_prompt or "")) else "en"
    device_name = "ACCOUNTS_DEVICE" if role == "accounts" else "LEDGER_DEVICE"
    if language == "zh":
        lines = [f"请选择 {device_name}，回复编号或直接输入设备路径。"]
        for option in options:
            oid = str(option.get("id") or "").strip()
            label = str(option.get("label") or option.get("value") or "").strip()
            if oid and label:
                lines.append(f"{oid}. {label}")
        lines.append("输入 back 返回上一步。")
        return "\n".join(lines)
    lines = [f"Choose {device_name}; reply with a number or enter a device path."]
    for option in options:
        oid = str(option.get("id") or "").strip()
        label = str(option.get("label") or option.get("value") or "").strip()
        if oid and label:
            lines.append(f"{oid}. {label}")
    lines.append("Enter back to return to the previous step.")
    return "\n".join(lines)


def _device_value_from_label(label: str) -> str:
    text = label.strip()
    for token in text.replace("(", " ").replace(")", " ").replace("，", " ").replace(",", " ").split():
        if token.startswith("/dev/"):
            return token
    first = text.split()[0] if text.split() else text
    if first.startswith("/dev/"):
        return first
    if first:
        return f"/dev/{first}" if not first.startswith("/") else first
    return text


def _quick_assumed_smoke_prompt(chain: str, rpc_mode: str, language: str) -> str:
    normalized_language = (language or "en").strip().lower()
    if normalized_language.startswith("zh"):
        return (
            "我可以用 quick fake-node smoke 快速验证 Agent 和框架闭环是否能跑通。\n"
            f"- 链: {chain}\n"
            f"- RPC 模式: {rpc_mode}\n"
            "- 目标: fake-node，不连接真实 RPC\n"
            "- 未确认的磁盘、网络、进程名和容量上限会使用 smoke-only 假设值\n"
            "- 标记: assumed_for_smoke=true；该结果不能代表真实节点性能\n"
            "回复 `Y` 接受并直接运行 smoke，或回复 `N` 返回配置流程。"
        )
    return (
        "I can run a quick fake-node smoke test to verify the Agent and framework closed loop.\n"
        f"- Chain: {chain}\n"
        f"- RPC mode: {rpc_mode}\n"
        "- Target: fake-node; no real RPC endpoint is used\n"
        "- Unconfirmed disk, network, process-name, and capacity values use smoke-only assumptions\n"
        "- Marker: assumed_for_smoke=true; this does not represent real node performance\n"
        "Reply `Y` to accept and run smoke, or `N` to return to configuration."
    )


def _benchmark_profile_prompt_and_options(language: str) -> tuple[str, list[dict[str, Any]]]:
    profiles = {
        "quick": {"initial": 1, "max": 1, "step": 1, "duration_seconds": 10},
        "standard": {"initial": 100, "max": 5000, "step": 100, "duration_seconds": 60},
        "intensive": {"initial": 100, "max": 50000, "step": 500, "duration_seconds": 300},
    }
    if (language or "en").strip().lower().startswith("zh"):
        prompt = (
            "请选择 benchmark 模式：\n"
            "1. quick — 快速验证/冒烟测试，耗时短\n"
            "2. standard — 标准性能测试（默认，推荐）\n"
            "3. intensive — 长时间瓶颈搜索测试\n"
            "回复 `1`、`2` 或 `3`，也可以输入要调整的 QPS 参数。"
        )
        labels = {
            "quick": "quick — 快速验证/冒烟测试，耗时短",
            "standard": "standard — 标准性能测试（默认，推荐）",
            "intensive": "intensive — 长时间瓶颈搜索测试",
        }
    else:
        prompt = (
            "Choose a benchmark profile:\n"
            "1. quick — short smoke validation\n"
            "2. standard — standard performance test (recommended default)\n"
            "3. intensive — long bottleneck search\n"
            "Reply `1`, `2`, or `3`, or type QPS adjustments."
        )
        labels = {
            "quick": "quick — short smoke validation",
            "standard": "standard — standard performance test (recommended default)",
            "intensive": "intensive — long bottleneck search",
        }
    options = []
    for index, name in enumerate(["quick", "standard", "intensive"], start=1):
        options.append(
            {
                "id": str(index),
                "label": labels[name],
                "value": name,
                "state_patch": {
                    "benchmark_profile": {"name": name, **profiles[name]},
                    "workflow_step": "benchmark_profile_confirmed",
                },
                "transition": {
                    "workflow_step": "benchmark_profile_confirmed",
                    "validator": "validate_required_config",
                },
            }
        )
    return prompt, options


def _proceed_discovery_prompt(language: str, summary: str = "") -> str:
    summary = _compact_discovery_summary(summary)
    if (language or "en").strip().lower().startswith("zh"):
        base = "是否继续进行环境发现、preflight 和 smoke？回复 `Y` 继续，或回复 `N` 暂停并调整配置。"
    else:
        base = "Continue with environment discovery, preflight, and smoke? Reply `Y` to continue, or `N` to pause and adjust configuration."
    return "\n".join(item for item in [summary, base] if item).strip()


def _compact_discovery_summary(summary: str) -> str:
    text = " ".join(str(summary or "").strip().split())
    if not text:
        return ""
    return text[:240]


def _workload_customization_prompt_and_options(
    chain: str,
    rpc_mode: str,
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    if (language or "en").strip().lower().startswith("zh"):
        prompt = (
            f"{chain} {rpc_mode} 默认 workload 已确认。请选择下一步：\n"
            "1. 使用默认值继续\n"
            "2. 添加自定义 RPC 方法\n"
            "3. 调整 mixed 权重\n"
            "4. 更换链或目标模式\n"
            "回复 `1`、`2`、`3` 或 `4`，也可以手动说明要怎么改。"
        )
        labels = {
            "use_defaults": "使用默认值继续",
            "add_custom_rpc": "添加自定义 RPC 方法",
            "adjust_weights": "调整 mixed 权重",
            "change_chain_or_mode": "更换链或目标模式",
        }
    else:
        prompt = (
            f"{chain} {rpc_mode} default workload is confirmed. Choose the next step:\n"
            "1. Continue with defaults\n"
            "2. Add a custom RPC method\n"
            "3. Adjust mixed weights\n"
            "4. Change chain or target mode\n"
            "Reply `1`, `2`, `3`, or `4`, or describe the change manually."
        )
        labels = {
            "use_defaults": "Continue with defaults",
            "add_custom_rpc": "Add a custom RPC method",
            "adjust_weights": "Adjust mixed weights",
            "change_chain_or_mode": "Change chain or target mode",
        }
    return prompt, [
        {
            "id": "1",
            "label": labels["use_defaults"],
            "value": "use_defaults",
            "state_patch": {
                "workflow_step": "workload_default_confirmed",
                "confirmed_config": {
                    "chain_template_reviewed": True,
                    "rpc_workload_confirmed": True,
                    "rpc_param_samples_confirmed": True,
                    "mixed_weights_confirmed": True,
                },
            },
            "transition": {"workflow_step": "workload_default_confirmed", "tool": "validate_rpc_workload"},
        },
        {
            "id": "2",
            "label": labels["add_custom_rpc"],
            "value": "add_custom_rpc",
            "state_patch": {"workflow_step": "custom_rpc_requested", "fixture_status": {"status": "needs_endpoint"}},
            "transition": {"workflow_step": "custom_rpc_endpoint_gate", "next_question_id": "custom_rpc_endpoint_gate"},
        },
        {
            "id": "3",
            "label": labels["adjust_weights"],
            "value": "adjust_weights",
            "state_patch": {"workflow_step": "mixed_weight_adjustment_requested"},
            "transition": {"workflow_step": "mixed_weights_adjust", "next_question_id": "mixed_weights_confirm"},
        },
        {
            "id": "4",
            "label": labels["change_chain_or_mode"],
            "value": "change_chain_or_mode",
            "state_patch": {"workflow_step": "chain_selection"},
            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
        },
    ]


def _unsupported_chain_endpoint_prompt(chain: str, adapter_family: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        return (
            f"{chain} 不在当前 36 条链中，初步归类为 `{adapter_family}` family。\n"
            "请提供一个可访问的 LOCAL_RPC_URL 或 public RPC endpoint 用于验证。"
            "如果暂时没有 endpoint，可以直接说明没有 endpoint，并要求生成 needs_review 开发交接文档。"
        )
    return (
        f"{chain} is not one of the current 36 chain templates and is tentatively classified as `{adapter_family}` family.\n"
        "Provide a reachable LOCAL_RPC_URL or public RPC endpoint for validation. "
        "If no endpoint is available, say so and ask for a needs_review development handoff."
    )


def _custom_rpc_endpoint_prompt(chain: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        return (
            f"为 {chain} 增加自定义 RPC method 前，必须先有一个可访问的 endpoint。"
            "请提供 LOCAL_RPC_URL 或 public RPC endpoint；如果没有 endpoint，"
            "我只能给出 needs_review 的缺失项清单，不能声明 fixture、fake-node 或 smoke 已可用。"
        )
    return (
        f"Before adding a custom RPC method for {chain}, a reachable endpoint is mandatory. "
        "Provide LOCAL_RPC_URL or a public RPC endpoint. If no endpoint is available, "
        "I can only provide a needs_review missing-evidence checklist; fixtures, fake-node support, and smoke are not ready."
    )


def _real_node_endpoint_prompt(chain: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        return (
            f"{chain} real-node benchmark 需要先验证被测节点 endpoint。\n"
            "请提供 LOCAL_RPC_URL（例如 `http://127.0.0.1:8899`）。"
            "如果暂时没有 LOCAL_RPC_URL，请直接说明没有；我会列出阻塞项，并可以帮你切换到 fake-node smoke。"
        )
    return (
        f"A {chain} real-node benchmark must validate the target endpoint first.\n"
        "Provide LOCAL_RPC_URL, for example `http://127.0.0.1:8899`. "
        "If LOCAL_RPC_URL is not available yet, say so to list blockers or switch to fake-node smoke."
    )
