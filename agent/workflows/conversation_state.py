"""File-backed workflow state for the ADK product Agent.

This module stores structured conversation facts. It intentionally does not
parse natural language, classify intent, or route business workflows. ADK and
the configured model infer intent and entities, then update this state through
typed patches. Validators decide whether execution can continue.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge.chain_identity import canonical_chain_aliases
from knowledge.chain_identity import canonicalize_chain_scalar
from knowledge.chain_identity import repo_chain_names
from validators.rpc_workload import default_workload
from workflows.group_registry import GROUP_FIELD_MAP
from workflows.group_registry import GROUP_ORDER
from workflows.group_registry import group_for_field as registry_group_for_field
from workflows.group_registry import normalize_group_name as registry_normalize_group_name


SCHEMA_VERSION = 1
DEFAULT_SESSION_ID = "terminal-session"
DEFAULT_STATE_ROOT = Path(".agent/sessions")
DEFAULT_GROUP_ORDER = GROUP_ORDER
GROUPS = set(DEFAULT_GROUP_ORDER)
DEVICE_VALUE_RE = re.compile(
    r"^(?:/dev/)?(?:(?:sd|vd|xvd)[a-z][0-9]*|nvme[0-9]+n[0-9]+p?[0-9]*|dm-[0-9]+|md[0-9]+|mapper/[A-Za-z0-9_.:+-]+|disk/by-[A-Za-z0-9_.:+-]+/[A-Za-z0-9_.:@+-]+)$"
)
NUMERIC_VALUE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:\s*(?:gib|gb|tib|tb|mib/s|mb/s|gbps|iops))?$", re.IGNORECASE)
STORAGE_TYPE_RE = re.compile(r"^(?:hyperdisk-[a-z0-9-]+|pd-[a-z0-9-]+|local-ssd|nvme|ssd|hdd|standard|gp[0-9]+|io[0-9]+|st[0-9]+|sc[0-9]+)$", re.IGNORECASE)
LOCATION_VALUE_RE = re.compile(r"^(?:global|(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*-)[A-Za-z][A-Za-z0-9_-]{1,63})$")
SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}$")
PROCESS_NAMES_RE = re.compile(r"^[A-Za-z0-9_./:+-]+(?:[ ,]+[A-Za-z0-9_./:+-]+)*$")

ALLOWED_PATCH_KEYS = {
    "language",
    "active_intent",
    "active_workflow",
    "workflow_step",
    "target_mode",
    "chain",
    "chain_status",
    "chain_identity_candidate",
    "chain_protocol_candidate",
    "rpc_mode",
    "rpc_methods",
    "mixed_weights",
    "custom_rpc",
    "custom_rpc_methods",
    "benchmark_profile",
    "confirmed_config",
    "inferred_config",
    "missing_fields",
    "pending_question",
    "assumed_values",
    "assumed_for_smoke",
    "observability",
    "endpoint_validation",
    "fixture_status",
    "blockers",
    "allowed_next_actions",
    "preflight_result",
    "smoke_result",
    "approval",
    "evidence_buffer",
    "last_input_mode",
    "last_user_change",
    "latest_plan_file",
    "latest_job_id",
    "history_summary",
    "active_group",
    "group_progress",
    "confirmed_fields",
    "invalidated_fields",
    "evidence",
    "interruption_stack",
    "last_completed_group",
    "next_blocking_group",
}

PENDING_ANSWER_TYPES = {
    "yes_no",
    "numbered_choice",
    "manual_value",
    "url",
    "device",
    "qps_profile",
    "rpc_weight",
    "chain",
    "rpc_method",
    "multi_select",
    "confirmation",
    "evidence_apply",
    "free_text",
}

YES_VALUES = {"y", "yes"}
NO_VALUES = {"n", "no"}
BACK_VALUES = {"back", "previous", "undo", "返回", "上一步", "回退", "撤回", "回到上一步"}

PENDING_REQUIRED_KEYS = {"id", "prompt", "kind"}
PENDING_ALLOWED_KEYS = {
    "id",
    "kind",
    "prompt",
    "expected_answer",
    "field",
    "fields",
    "current_value",
    "source",
    "reason",
    "manual_input_allowed",
    "allow_manual_input",
    "options",
    "default",
    "default_choice",
    "default_option",
    "recommended",
    "recommended_choice",
    "recommended_option",
    "next_on_answer",
    "next_on_yes",
    "next_on_no",
    "next_on_choice",
    "next_on_manual",
    "next_on_back",
    "next_on_cancel",
    "validation_tool",
    "validation_args",
    "state_patch_on_valid",
    "workflow_step",
    "branch",
    "source_tool",
    "requires_revalidation",
    "clear_on_answer",
    "blocking",
    "blocks_execution",
    "evidence_required",
    "artifact_paths",
    "known_chains",
    "chain_aliases",
    "created_at",
}

PENDING_QUESTION_REGISTRY = {
    "dependency_install": {"kind": "yes_no", "blocks_execution": True},
    "previous_job_resume": {"kind": "yes_no", "blocks_execution": False},
    "opening_help_choice": {"kind": "numbered_choice", "blocks_execution": False},
    "target_mode": {"kind": "numbered_choice", "blocks_execution": True},
    "chain_selection": {"kind": "manual_value", "blocks_execution": True},
    "chain_identity_resolution": {"kind": "numbered_choice", "blocks_execution": True},
    "chain_protocol_resolution": {"kind": "numbered_choice", "blocks_execution": True},
    "confirm_chain_change": {"kind": "yes_no", "blocks_execution": True},
    "quick_assumed_smoke_confirm": {"kind": "yes_no", "blocks_execution": True},
    "confirm_assumed_smoke": {"kind": "yes_no", "blocks_execution": True},
    "real_node_local_rpc_url": {"kind": "url", "blocks_execution": True},
    "real_node_mainnet_rpc_url": {"kind": "url", "blocks_execution": True},
    "cloud_region": {"kind": "manual_value", "blocks_execution": True},
    "cloud_zone": {"kind": "manual_value", "blocks_execution": True},
    "machine_type": {"kind": "manual_value", "blocks_execution": True},
    "disk_ledger_choice": {"kind": "device", "blocks_execution": True},
    "disk_accounts_exists": {"kind": "yes_no", "blocks_execution": False},
    "disk_accounts_choice": {"kind": "device", "blocks_execution": True},
    "data_vol_type": {"kind": "manual_value", "blocks_execution": True},
    "data_vol_size": {"kind": "manual_value", "blocks_execution": True},
    "data_vol_max_iops": {"kind": "manual_value", "blocks_execution": True},
    "data_vol_max_throughput": {"kind": "manual_value", "blocks_execution": True},
    "accounts_vol_type": {"kind": "manual_value", "blocks_execution": True},
    "accounts_vol_size": {"kind": "manual_value", "blocks_execution": True},
    "accounts_vol_max_iops": {"kind": "manual_value", "blocks_execution": True},
    "accounts_vol_max_throughput": {"kind": "manual_value", "blocks_execution": True},
    "volume_baseline_confirm": {"kind": "yes_no", "blocks_execution": True},
    "network_interface": {"kind": "manual_value", "blocks_execution": True},
    "network_interface_confirm": {"kind": "yes_no", "blocks_execution": True},
    "network_max_bandwidth_gbps": {"kind": "manual_value", "blocks_execution": True},
    "network_bandwidth_confirm": {"kind": "yes_no", "blocks_execution": True},
    "blockchain_process_names": {"kind": "manual_value", "blocks_execution": True},
    "process_names_confirm": {"kind": "yes_no", "blocks_execution": True},
    "rpc_mode_choice": {"kind": "numbered_choice", "blocks_execution": True},
    "default_workload_confirm": {"kind": "yes_no", "blocks_execution": True},
    "workload_confirm": {"kind": "yes_no", "blocks_execution": True},
    "workload_customization_choice": {"kind": "numbered_choice", "blocks_execution": True},
    "mixed_weights_confirm": {"kind": "yes_no", "blocks_execution": True},
    "custom_rpc_add": {"kind": "yes_no", "blocks_execution": False},
    "custom_rpc_endpoint_gate": {"kind": "url", "blocks_execution": True},
    "custom_rpc_endpoint_required": {"kind": "url", "blocks_execution": True},
    "custom_rpc_method_name": {"kind": "free_text", "blocks_execution": True},
    "custom_rpc_params_sample": {"kind": "manual_value", "blocks_execution": True},
    "custom_rpc_fixture_recording": {"kind": "confirmation", "blocks_execution": True},
    "benchmark_profile_choice": {"kind": "numbered_choice", "blocks_execution": True},
    "benchmark_profile_confirm": {"kind": "yes_no", "blocks_execution": True},
    "benchmark_profile_adjust_item": {"kind": "numbered_choice", "blocks_execution": True},
    "benchmark_profile_adjust_value": {"kind": "manual_value", "blocks_execution": True},
    "observability_mode_choice": {"kind": "numbered_choice", "blocks_execution": True},
    "observability_ports_confirm": {"kind": "yes_no", "blocks_execution": True},
    "advanced_threshold_review": {"kind": "yes_no", "blocks_execution": False},
    "proceed_discovery": {"kind": "yes_no", "blocks_execution": True},
    "preflight_fix_confirm": {"kind": "yes_no", "blocks_execution": True},
    "smoke_run_confirm": {"kind": "yes_no", "blocks_execution": True},
    "real_benchmark_submit_confirm": {"kind": "yes_no", "blocks_execution": True},
    "job_follow_logs": {"kind": "yes_no", "blocks_execution": False},
    "apply_pasted_evidence": {"kind": "evidence_apply", "blocks_execution": True},
    "unsupported_chain_family_confirm": {"kind": "yes_no", "blocks_execution": True},
    "unsupported_chain_endpoint_gate": {"kind": "url", "blocks_execution": True},
    "unsupported_chain_endpoint_required": {"kind": "url", "blocks_execution": True},
    "unsupported_chain_handoff_confirm": {"kind": "confirmation", "blocks_execution": True},
}


def default_workflow_state(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "language": "en",
        "active_intent": "",
        "active_workflow": "",
        "workflow_step": "",
        "target_mode": "",
        "chain": "",
        "chain_status": "unknown",
        "chain_identity_candidate": {},
        "chain_protocol_candidate": {},
        "rpc_mode": "",
        "rpc_methods": [],
        "mixed_weights": {},
        "custom_rpc": [],
        "custom_rpc_methods": [],
        "benchmark_profile": {},
        "confirmed_config": {},
        "inferred_config": {},
        "missing_fields": [],
        "pending_question": {},
        "assumed_values": {},
        "assumed_for_smoke": False,
        "observability": {},
        "endpoint_validation": {},
        "fixture_status": {},
        "blockers": [],
        "allowed_next_actions": [],
        "preflight_result": {},
        "smoke_result": {},
        "approval": {},
        "evidence_buffer": {},
        "last_input_mode": "",
        "last_user_change": "",
        "latest_plan_file": "",
        "latest_job_id": "",
        "history_summary": "",
        "active_group": "",
        "group_progress": _normalize_group_progress({}),
        "confirmed_fields": [],
        "invalidated_fields": [],
        "evidence": [],
        "interruption_stack": [],
        "last_completed_group": "",
        "next_blocking_group": "",
        "history": [],
        "revision": 0,
        "created_at": now,
        "updated_at": now,
    }


def workflow_state_path(
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> Path:
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return Path(state_root) / safe_session / "conversation_state.json"


def load_workflow_state(
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    path = workflow_state_path(session_id=session_id, state_root=state_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_workflow_state(session_id=session_id)
    return _normalize_state(payload, session_id=session_id)


def save_workflow_state(
    state: dict[str, Any],
    session_id: str | None = None,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> Path:
    normalized = _normalize_state(state, session_id=session_id or state.get("session_id") or DEFAULT_SESSION_ID)
    path = workflow_state_path(session_id=normalized["session_id"], state_root=state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_workflow_state(
    patch: dict[str, Any],
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Apply a structured patch and persist state.

    The patch must be explicit fields inferred by ADK. Unknown keys are ignored
    and returned as warnings so they cannot silently become runtime behavior.
    """
    current = load_workflow_state(session_id=session_id, state_root=state_root)
    clean_patch = {key: deepcopy(value) for key, value in (patch or {}).items() if key in ALLOWED_PATCH_KEYS}
    ignored = sorted(set((patch or {}).keys()) - ALLOWED_PATCH_KEYS)
    validation_warnings: list[str] = []
    before_update = deepcopy(current)
    _push_history(current)

    dict_keys = {
        "confirmed_config",
        "inferred_config",
        "mixed_weights",
        "assumed_values",
        "benchmark_profile",
        "observability",
        "endpoint_validation",
        "fixture_status",
        "preflight_result",
        "smoke_result",
        "approval",
        "evidence_buffer",
        "group_progress",
        "chain_identity_candidate",
        "chain_protocol_candidate",
    }
    list_keys = {
        "rpc_methods",
        "custom_rpc",
        "custom_rpc_methods",
        "missing_fields",
        "blockers",
        "allowed_next_actions",
        "confirmed_fields",
        "invalidated_fields",
        "evidence",
        "interruption_stack",
    }

    for key, value in clean_patch.items():
        if key == "confirmed_config":
            current[key], warnings = _merge_confirmed_config(current.get(key), value)
            validation_warnings.extend(warnings)
        elif key in dict_keys:
            current[key] = _merge_dict(current.get(key), value)
        elif key in list_keys:
            current[key] = list(value or [])
        elif key == "pending_question":
            normalized_question, warnings = normalize_pending_question(value)
            current[key] = normalized_question
            validation_warnings.extend(warnings)
        elif key in {"active_group", "last_completed_group", "next_blocking_group"}:
            current[key] = _normalize_group_name(value)
        else:
            current[key] = _normalize_scalar_field(key, value)

    _sync_top_level_runtime_fields(current)

    changed_groups = _changed_groups_from_patch(before_update, current, clean_patch)
    if changed_groups:
        _apply_group_invalidations(current, changed_groups, clean_patch)

    current["revision"] = int(current.get("revision") or 0) + 1
    current["updated_at"] = _now()
    if reason:
        current["last_update_reason"] = reason
    path = save_workflow_state(current, session_id=session_id, state_root=state_root)
    return {
        "state": current,
        "state_file": str(path),
        "ignored_keys": ignored,
        "validation_warnings": validation_warnings,
    }


def normalize_pending_question(value: Any) -> tuple[dict[str, Any], list[str]]:
    """Return a canonical pending-question object and validation warnings.

    Pending questions are the contract for short replies such as ``Y``, ``N``,
    ``1``, a disk name, or a URL. This function validates state shape only; it
    never parses user text or routes business intent.
    """
    if not value:
        return {}, []
    if not isinstance(value, dict):
        return {}, ["pending_question must be an object"]
    question = dict(value)
    warnings: list[str] = []
    unknown = sorted(set(question) - PENDING_ALLOWED_KEYS)
    if unknown:
        warnings.append(f"pending_question ignored unsupported keys: {', '.join(unknown)}")
        for key in unknown:
            question.pop(key, None)
    if "kind" not in question and "expected_answer" in question:
        question["kind"] = question.get("expected_answer")
    if "expected_answer" not in question and "kind" in question:
        question["expected_answer"] = question.get("kind")
    registry = PENDING_QUESTION_REGISTRY.get(str(question.get("id", "")).strip())
    if registry:
        question.setdefault("kind", registry["kind"])
        question.setdefault("expected_answer", registry["kind"])
        question.setdefault("blocks_execution", registry["blocks_execution"])
    missing = sorted(key for key in PENDING_REQUIRED_KEYS if not str(question.get(key, "")).strip())
    if missing:
        warnings.append(f"pending_question missing required keys: {', '.join(missing)}")
    kind = str(question.get("kind") or question.get("expected_answer") or "").strip()
    if kind not in PENDING_ANSWER_TYPES:
        warnings.append(
            "pending_question kind must be one of: "
            + ", ".join(sorted(PENDING_ANSWER_TYPES))
        )
        if kind:
            question["expected_answer_original"] = kind
        kind = "free_text"
    question["id"] = str(question.get("id", "")).strip()
    question["prompt"] = str(question.get("prompt", "")).strip()
    question["kind"] = kind
    question["expected_answer"] = kind
    question["field"] = str(question.get("field", "")).strip()
    question["fields"] = [str(item).strip() for item in list(question.get("fields") or []) if str(item).strip()]
    question["current_value"] = deepcopy(question.get("current_value"))
    question["source"] = str(question.get("source", "")).strip()
    question["reason"] = str(question.get("reason", "")).strip()
    manual_default = question.get("manual_input_allowed", question.get("allow_manual_input", True))
    question["manual_input_allowed"] = bool(manual_default)
    question["allow_manual_input"] = question["manual_input_allowed"]
    question["options"] = _normalize_options(question.get("options", []))
    default_option, default_warnings = _normalize_default_option(question, question["options"])
    if default_option:
        question["default_option"] = default_option
    else:
        question.pop("default_option", None)
    warnings.extend(default_warnings)
    for alias in ("default", "default_choice", "recommended", "recommended_choice", "recommended_option"):
        question.pop(alias, None)
    question["next_on_answer"] = dict(question.get("next_on_answer") or {})
    question["next_on_yes"] = _normalize_transition(question.get("next_on_yes"))
    question["next_on_no"] = _normalize_transition(question.get("next_on_no"))
    question["next_on_choice"] = _normalize_transition(question.get("next_on_choice"))
    question["next_on_manual"] = _normalize_transition(question.get("next_on_manual"))
    question["next_on_back"] = _normalize_transition(question.get("next_on_back"))
    question["next_on_cancel"] = _normalize_transition(question.get("next_on_cancel"))
    question["validation_tool"] = str(question.get("validation_tool", "")).strip()
    question["validation_args"] = dict(question.get("validation_args") or {})
    question["state_patch_on_valid"] = dict(question.get("state_patch_on_valid") or {})
    question["workflow_step"] = str(question.get("workflow_step", "")).strip()
    question["branch"] = str(question.get("branch", "")).strip()
    question["source_tool"] = str(question.get("source_tool", "")).strip()
    question["requires_revalidation"] = bool(question.get("requires_revalidation", True))
    question["clear_on_answer"] = bool(question.get("clear_on_answer", True))
    blocks_execution = bool(question.get("blocks_execution", question.get("blocking", True)))
    question["blocking"] = blocks_execution
    question["blocks_execution"] = blocks_execution
    question["evidence_required"] = bool(question.get("evidence_required", False))
    question["artifact_paths"] = [str(item).strip() for item in list(question.get("artifact_paths") or []) if str(item).strip()]
    question["created_at"] = str(question.get("created_at") or _now())
    return question, warnings


def revert_workflow_state(
    steps: int = 1,
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Restore an earlier structured state snapshot.

    This supports user corrections such as "go back" or "I gave the wrong
    disk". ADK should still explain what changed and re-run validators before
    execution.
    """
    current = load_workflow_state(session_id=session_id, state_root=state_root)
    history = list(current.get("history") or [])
    count = max(1, int(steps or 1))
    if not history:
        path = save_workflow_state(current, session_id=session_id, state_root=state_root)
        return {
            "state": current,
            "state_file": str(path),
            "reverted": False,
            "message": "no previous workflow state snapshot",
        }
    selected = history[-count] if len(history) >= count else history[0]
    restored = _normalize_state(selected, session_id=session_id)
    restored["history"] = history[: max(0, len(history) - count)]
    restored["revision"] = int(current.get("revision") or 0) + 1
    restored["updated_at"] = _now()
    if reason:
        restored["last_update_reason"] = reason
    path = save_workflow_state(restored, session_id=session_id, state_root=state_root)
    return {
        "state": restored,
        "state_file": str(path),
        "reverted": True,
        "message": f"reverted {min(count, len(history))} workflow state snapshot(s)",
    }


def reset_workflow_state(
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    state = default_workflow_state(session_id=session_id)
    if reason:
        state["last_update_reason"] = reason
    path = save_workflow_state(state, session_id=session_id, state_root=state_root)
    return {
        "state": state,
        "state_file": str(path),
    }


def record_group_jump(
    target_group: str,
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Pause the current group and switch to another workflow group.

    This is the canonical state transition for user interruptions such as
    changing chain during disk configuration or asking to adjust QPS while a
    workload question is active. It does not parse user prose and it does not
    decide which group to choose; ADK/model must choose ``target_group`` first.
    """
    group = _normalize_group_name(target_group)
    if not group:
        current = load_workflow_state(session_id=session_id, state_root=state_root)
        return {
            "state": current,
            "state_file": str(workflow_state_path(session_id=session_id, state_root=state_root)),
            "applied": False,
            "blockers": [f"unknown workflow group: {target_group}"],
        }

    current = load_workflow_state(session_id=session_id, state_root=state_root)
    progress = _normalize_group_progress(current.get("group_progress") or {})
    active = _normalize_group_name(current.get("active_group") or "")
    pending = normalize_pending_question(current.get("pending_question") or {})[0]
    stack = _normalize_interruption_stack(current.get("interruption_stack") or [])
    if active and active != group:
        stack.append(
            {
                "group": active,
                "pending_question": pending,
                "reason": reason or f"jump_to_{group}",
                "created_at": _now(),
            }
        )
    if active:
        progress[active]["status"] = "in_progress"
    progress[group]["status"] = "in_progress"
    current["active_group"] = group
    current["next_blocking_group"] = group
    current["interruption_stack"] = _normalize_interruption_stack(stack)
    current["group_progress"] = progress
    current["pending_question"] = {}
    current["workflow_step"] = group
    current["revision"] = int(current.get("revision") or 0) + 1
    current["updated_at"] = _now()
    current["last_update_reason"] = reason or f"record_group_jump:{group}"
    path = save_workflow_state(current, session_id=session_id, state_root=state_root)
    return {
        "state": current,
        "state_file": str(path),
        "applied": True,
        "target_group": group,
        "paused_group": active,
    }


def recompute_next_blocking_group(
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Recompute the next blocking group from canonical group progress."""
    current = load_workflow_state(session_id=session_id, state_root=state_root)
    progress = _normalize_group_progress(current.get("group_progress") or {})
    next_group = _first_invalidated_or_incomplete_group(progress)
    current["next_blocking_group"] = next_group
    if not current.get("active_group"):
        current["active_group"] = next_group
    current["revision"] = int(current.get("revision") or 0) + 1
    current["updated_at"] = _now()
    current["last_update_reason"] = reason or "recompute_next_blocking_group"
    path = save_workflow_state(current, session_id=session_id, state_root=state_root)
    return {
        "state": current,
        "state_file": str(path),
        "next_blocking_group": next_group,
    }


def _normalize_state(payload: dict[str, Any], session_id: str) -> dict[str, Any]:
    state = default_workflow_state(session_id=session_id)
    for key in state:
        if key in payload:
            state[key] = deepcopy(payload[key])
    state["schema_version"] = SCHEMA_VERSION
    state["session_id"] = str(payload.get("session_id") or session_id or DEFAULT_SESSION_ID)
    state["revision"] = int(state.get("revision") or 0)
    state["confirmed_config"] = dict(state.get("confirmed_config") or {})
    state["inferred_config"] = dict(state.get("inferred_config") or {})
    state["chain_identity_candidate"] = dict(state.get("chain_identity_candidate") or {})
    state["chain_protocol_candidate"] = dict(state.get("chain_protocol_candidate") or {})
    state["mixed_weights"] = dict(state.get("mixed_weights") or {})
    state["benchmark_profile"] = dict(state.get("benchmark_profile") or {})
    state["pending_question"] = normalize_pending_question(state.get("pending_question") or {})[0]
    state["assumed_values"] = dict(state.get("assumed_values") or {})
    state["assumed_for_smoke"] = bool(state.get("assumed_for_smoke"))
    state["observability"] = dict(state.get("observability") or {})
    state["endpoint_validation"] = dict(state.get("endpoint_validation") or {})
    state["fixture_status"] = dict(state.get("fixture_status") or {})
    state["preflight_result"] = dict(state.get("preflight_result") or {})
    state["smoke_result"] = dict(state.get("smoke_result") or {})
    state["approval"] = dict(state.get("approval") or {})
    state["evidence_buffer"] = dict(state.get("evidence_buffer") or {})
    state["group_progress"] = _normalize_group_progress(state.get("group_progress") or {})
    state["confirmed_fields"] = _dedupe_text_list(state.get("confirmed_fields") or [])
    state["invalidated_fields"] = _dedupe_text_list(state.get("invalidated_fields") or [])
    state["evidence"] = list(state.get("evidence") or [])
    state["interruption_stack"] = _normalize_interruption_stack(state.get("interruption_stack") or [])
    state["active_group"] = _normalize_group_name(state.get("active_group") or "")
    state["last_completed_group"] = _normalize_group_name(state.get("last_completed_group") or "")
    state["next_blocking_group"] = _normalize_group_name(state.get("next_blocking_group") or "")
    state["last_input_mode"] = str(state.get("last_input_mode") or "")
    state["rpc_methods"] = list(state.get("rpc_methods") or [])
    state["custom_rpc"] = list(state.get("custom_rpc") or [])
    state["custom_rpc_methods"] = list(state.get("custom_rpc_methods") or [])
    state["missing_fields"] = list(state.get("missing_fields") or [])
    state["blockers"] = list(state.get("blockers") or [])
    state["allowed_next_actions"] = list(state.get("allowed_next_actions") or [])
    state["history"] = list(state.get("history") or [])
    return state


def _merge_dict(existing: Any, incoming: Any) -> dict[str, Any]:
    merged = dict(existing or {})
    for key, value in dict(incoming or {}).items():
        if value is None or value == "":
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _changed_groups_from_patch(before: dict[str, Any], after: dict[str, Any], patch: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    existing_evidence = _has_existing_execution_evidence(before)

    def add(group: str) -> None:
        normalized = _normalize_group_name(group)
        if normalized and normalized not in groups:
            groups.append(normalized)

    for key in patch:
        if key == "pending_question":
            continue
        group = _group_for_state_key(key)
        if group and before.get(key) != after.get(key) and (_has_meaningful_value(before.get(key)) or existing_evidence):
            add(group)

    if isinstance(patch.get("confirmed_config"), dict):
        before_config = before.get("confirmed_config") if isinstance(before.get("confirmed_config"), dict) else {}
        after_config = after.get("confirmed_config") if isinstance(after.get("confirmed_config"), dict) else {}
        for key in patch["confirmed_config"]:
            key_text = str(key or "").strip()
            if not key_text:
                continue
            if (
                before_config.get(key_text) != after_config.get(key_text)
                and (_has_meaningful_value(before_config.get(key_text)) or existing_evidence)
            ):
                add(_group_for_field(key_text))

    return groups


def _has_existing_execution_evidence(state: dict[str, Any]) -> bool:
    return any(
        _has_meaningful_value(state.get(key))
        for key in ("endpoint_validation", "fixture_status", "preflight_result", "smoke_result", "approval")
    )


def _has_meaningful_value(value: Any) -> bool:
    if value in (None, "", False):
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _group_for_state_key(key: str) -> str:
    mapping = {
        "target_mode": "chain_target",
        "chain": "chain_target",
        "chain_status": "chain_target",
        "rpc_mode": "workload_rpc",
        "rpc_methods": "workload_rpc",
        "mixed_weights": "workload_rpc",
        "custom_rpc": "workload_rpc",
        "custom_rpc_methods": "workload_rpc",
        "benchmark_profile": "qps_profile",
        "observability": "observability",
        "endpoint_validation": "chain_auxiliary_endpoints",
        "fixture_status": "target_samples_fixtures",
        "preflight_result": "preflight_smoke_execution",
        "smoke_result": "preflight_smoke_execution",
        "approval": "preflight_smoke_execution",
        "evidence_buffer": "error_evidence_analysis",
        "evidence": "error_evidence_analysis",
        "latest_plan_file": "report_artifact_analysis",
        "latest_job_id": "report_artifact_analysis",
    }
    return mapping.get(str(key or "").strip(), "")


def _group_for_field(field: str) -> str:
    field_text = str(field or "").strip()
    group = registry_group_for_field(field_text)
    if group:
        return group
    lower_group = registry_group_for_field(field_text.lower())
    if lower_group:
        return lower_group
    if field_text.startswith(("QUICK_", "STANDARD_", "INTENSIVE_")):
        return "qps_profile"
    if field_text.startswith("TARGET_"):
        return "target_samples_fixtures"
    return ""


def _apply_group_invalidations(state: dict[str, Any], changed_groups: list[str], patch: dict[str, Any]) -> None:
    changed = [_normalize_group_name(group) for group in changed_groups]
    changed = [group for group in changed if group]
    if not changed:
        return
    changed_set = set(changed)
    invalidated = set(state.get("invalidated_fields") or [])
    progress = _normalize_group_progress(state.get("group_progress") or {})
    earliest_index = min(DEFAULT_GROUP_ORDER.index(group) for group in changed if group in GROUPS)

    for group in changed:
        progress[group]["status"] = "in_progress"

    for group in DEFAULT_GROUP_ORDER[earliest_index + 1 :]:
        progress[group]["status"] = "invalidated"
        progress[group]["invalidated_fields"] = _dedupe_text_list(
            list(progress[group].get("invalidated_fields") or []) + [",".join(changed)]
        )

    if any(DEFAULT_GROUP_ORDER.index(group) < DEFAULT_GROUP_ORDER.index("preflight_smoke_execution") for group in changed):
        for key in ("preflight_result", "smoke_result", "approval"):
            if key not in patch:
                state[key] = {}
            invalidated.add(key)

    if changed_set.intersection({"chain_target", "chain_auxiliary_endpoints", "workload_rpc", "target_samples_fixtures"}):
        for key in ("endpoint_validation", "fixture_status"):
            if key not in patch:
                state[key] = {}
            invalidated.add(key)

    if changed_set.intersection({"chain_target", "workload_rpc"}):
        invalidated.update({"rpc_methods", "mixed_weights", "target_samples", "fixtures"})
    if "qps_profile" in changed_set:
        invalidated.add("benchmark_profile")
    if "observability" in changed_set:
        invalidated.add("observability")
    if changed_set.intersection({"ledger_disk", "accounts_disk"}):
        invalidated.update({"disk_baseline", "resource_attribution"})
    if "network" in changed_set:
        invalidated.add("network_baseline")

    state["group_progress"] = progress
    state["invalidated_fields"] = _dedupe_text_list(sorted(invalidated))
    state["next_blocking_group"] = DEFAULT_GROUP_ORDER[earliest_index]


def _first_invalidated_or_incomplete_group(progress: dict[str, dict[str, Any]]) -> str:
    for group in DEFAULT_GROUP_ORDER:
        status = str((progress.get(group) or {}).get("status") or "").strip()
        if status != "complete":
            return group
    return ""


def _normalize_group_name(value: Any) -> str:
    return registry_normalize_group_name(value)


def _normalize_group_progress(value: Any) -> dict[str, dict[str, Any]]:
    result = {group: {"status": "pending", "confirmed_fields": [], "invalidated_fields": []} for group in DEFAULT_GROUP_ORDER}
    if not isinstance(value, dict):
        return result
    for raw_group, raw_progress in value.items():
        group = _normalize_group_name(raw_group)
        if not group or not isinstance(raw_progress, dict):
            continue
        current = dict(result[group])
        status = str(raw_progress.get("status") or current["status"]).strip().lower()
        current["status"] = status if status in {"pending", "in_progress", "complete", "invalidated", "blocked"} else "pending"
        current["confirmed_fields"] = _dedupe_text_list(raw_progress.get("confirmed_fields") or [])
        current["invalidated_fields"] = _dedupe_text_list(raw_progress.get("invalidated_fields") or [])
        evidence = raw_progress.get("evidence")
        if isinstance(evidence, list):
            current["evidence"] = list(evidence)
        elif isinstance(evidence, dict):
            current["evidence"] = dict(evidence)
        result[group] = current
    return result


def _normalize_interruption_stack(value: Any) -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = []
    for item in list(value or []):
        if not isinstance(item, dict):
            continue
        group = _normalize_group_name(item.get("group") or item.get("active_group") or "")
        if not group:
            continue
        entry = {
            "group": group,
            "pending_question": normalize_pending_question(item.get("pending_question") or {})[0],
            "reason": str(item.get("reason") or "").strip(),
            "created_at": str(item.get("created_at") or _now()),
        }
        stack.append(entry)
    return stack[-20:]


def _dedupe_text_list(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(value or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _merge_confirmed_config(existing: Any, incoming: Any) -> tuple[dict[str, Any], list[str]]:
    """Merge confirmed runtime config through the same scalar normalizers.

    ADK may update confirmed_config directly after interpreting a user turn.
    Those patches must not bypass the normalization used by
    answer_pending_question, otherwise the workflow can split into "field set"
    and "pending question still active" states.
    """
    merged = dict(existing or {})
    warnings: list[str] = []
    for key, value in dict(incoming or {}).items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if value is None or value == "":
            merged.pop(key_text, None)
            continue
        merged[key_text] = _normalize_confirmed_config_value(key_text, value)
    _sync_confirmed_config_aliases(merged)

    ledger = _first_config_value(merged, ("LEDGER_DEVICE", "ledger_device"))
    accounts = _first_config_value(merged, ("ACCOUNTS_DEVICE", "accounts_device"))
    has_accounts = _first_config_value(merged, ("has_accounts_device", "HAS_ACCOUNTS_DEVICE"))
    ledger = _normalize_answer_value_for_field("LEDGER_DEVICE", ledger)
    accounts = _normalize_answer_value_for_field("ACCOUNTS_DEVICE", accounts)
    if _truthy(has_accounts) and ledger and accounts and ledger == accounts:
        merged.pop("ACCOUNTS_DEVICE", None)
        merged.pop("accounts_device", None)
        warnings.append(
            "ignored ACCOUNTS_DEVICE because it matched LEDGER_DEVICE while a separate accounts/state disk was requested"
        )
    return merged, warnings


def _sync_confirmed_config_aliases(config: dict[str, Any]) -> None:
    pairs = (
        ("chain", "BLOCKCHAIN_NODE"),
        ("rpc_mode", "RPC_MODE"),
        ("target_mode", "TARGET_MODE"),
        ("ledger_device", "LEDGER_DEVICE"),
        ("accounts_device", "ACCOUNTS_DEVICE"),
        ("data_vol_type", "DATA_VOL_TYPE"),
        ("data_vol_size", "DATA_VOL_SIZE"),
        ("data_vol_max_iops", "DATA_VOL_MAX_IOPS"),
        ("data_vol_max_throughput", "DATA_VOL_MAX_THROUGHPUT"),
        ("accounts_vol_type", "ACCOUNTS_VOL_TYPE"),
        ("accounts_vol_size", "ACCOUNTS_VOL_SIZE"),
        ("accounts_vol_max_iops", "ACCOUNTS_VOL_MAX_IOPS"),
        ("accounts_vol_max_throughput", "ACCOUNTS_VOL_MAX_THROUGHPUT"),
        ("cloud_region", "CLOUD_REGION"),
        ("cloud_zone", "CLOUD_ZONE"),
        ("machine_type", "MACHINE_TYPE"),
        ("network_interface", "NETWORK_INTERFACE"),
        ("network_max_bandwidth_gbps", "NETWORK_MAX_BANDWIDTH_GBPS"),
    )
    for logical_key, env_key in pairs:
        logical_value = config.get(logical_key)
        env_value = config.get(env_key)
        if env_value not in (None, "") and logical_value in (None, ""):
            config[logical_key] = _normalize_confirmed_config_value(logical_key, env_value)
        elif logical_value not in (None, "") and env_value in (None, ""):
            config[env_key] = _normalize_confirmed_config_value(env_key, logical_value)
    process_names = config.get("blockchain_process_names") or config.get("BLOCKCHAIN_PROCESS_NAMES")
    if process_names not in (None, ""):
        normalized = _normalize_confirmed_config_value("BLOCKCHAIN_PROCESS_NAMES", process_names)
        config["blockchain_process_names"] = normalized
        config["BLOCKCHAIN_PROCESS_NAMES"] = normalized
        if isinstance(normalized, list):
            config.setdefault("BLOCKCHAIN_PROCESS_NAMES_STR", " ".join(str(item) for item in normalized if str(item)))


def _clear_fake_node_process_names(config: dict[str, Any]) -> None:
    process_names = config.get("blockchain_process_names") or config.get("BLOCKCHAIN_PROCESS_NAMES")
    normalized = _normalize_confirmed_config_value("BLOCKCHAIN_PROCESS_NAMES", process_names)
    process_name_text = str(config.get("BLOCKCHAIN_PROCESS_NAMES_STR") or "").strip()
    if normalized == ["fake-node"] or process_name_text == "fake-node":
        for key in ("blockchain_process_names", "BLOCKCHAIN_PROCESS_NAMES", "BLOCKCHAIN_PROCESS_NAMES_STR"):
            config.pop(key, None)


def _sync_top_level_runtime_fields(state: dict[str, Any]) -> None:
    """Keep execution config aliases in sync with authoritative workflow fields."""
    config = state.get("confirmed_config")
    if not isinstance(config, dict):
        config = {}
        state["confirmed_config"] = config

    chain = str(state.get("chain") or "").strip()
    if chain:
        config["chain"] = _normalize_confirmed_config_value("chain", chain)
        config["BLOCKCHAIN_NODE"] = _normalize_confirmed_config_value("BLOCKCHAIN_NODE", chain)

    target_mode = str(state.get("target_mode") or "").strip().lower().replace("_", "-")
    if target_mode in {"fake-node", "real-node"}:
        config["target_mode"] = target_mode
        config["TARGET_MODE"] = target_mode
        config["use_fake_node"] = target_mode == "fake-node"
        if target_mode == "fake-node":
            config.setdefault("blockchain_process_names", ["fake-node"])
            config.setdefault("BLOCKCHAIN_PROCESS_NAMES", ["fake-node"])
            config.setdefault("BLOCKCHAIN_PROCESS_NAMES_STR", "fake-node")
        else:
            _clear_fake_node_process_names(config)

    rpc_mode = str(state.get("rpc_mode") or "").strip().lower()
    if rpc_mode:
        config["rpc_mode"] = _normalize_confirmed_config_value("rpc_mode", rpc_mode)
        config["RPC_MODE"] = _normalize_confirmed_config_value("RPC_MODE", rpc_mode)

    profile = state.get("benchmark_profile")
    if isinstance(profile, dict):
        mode = str(profile.get("mode") or profile.get("name") or "").strip().lower()
        if mode:
            config["benchmark_mode_confirmed"] = mode

    observability = state.get("observability")
    if isinstance(observability, dict):
        mode = str(observability.get("mode") or "").strip().lower()
        if mode:
            config["observability_choice_confirmed"] = mode
            config["OBSERVABILITY_STACK_MODE"] = mode

    _sync_confirmed_config_aliases(config)


def _normalize_confirmed_config_value(key: str, value: Any) -> Any:
    if key in {"chain", "BLOCKCHAIN_NODE", "blockchain_node"}:
        return _canonical_chain_name(value)
    if key in {"rpc_mode", "RPC_MODE"}:
        return _normalize_scalar_field("rpc_mode", value)
    return _normalize_answer_value_for_field(key, value)


def _first_config_value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value or "").strip().lower()
    return lowered in {"1", "true", "yes", "y"}


def _normalize_scalar_field(key: str, value: Any) -> Any:
    if value is None:
        return ""
    if key == "target_mode":
        lowered = str(value).strip().lower().replace("_", "-")
        aliases = {
            "fake-node": "fake-node",
            "real-node": "real-node",
        }
        return aliases.get(lowered, lowered)
    if key == "rpc_mode":
        lowered = str(value).strip().lower()
        return lowered if lowered in {"single", "mixed"} else lowered
    if key == "chain":
        return _canonical_chain_name(value)
    if key in {"chain_status", "active_intent", "active_workflow", "workflow_step"}:
        return str(value).strip()
    return value


def _canonical_chain_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return canonicalize_chain_scalar(text) or text


def _normalize_options(value: Any) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return options
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("value") or index).strip()
            option = dict(item)
            option["label"] = label
            option["value"] = item.get("value", label)
        else:
            label = str(item).strip()
            option = {"label": label, "value": label}
        option.setdefault("id", str(index))
        options.append(option)
    return options


def _normalize_default_option(question: dict[str, Any], options: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    raw = None
    for key in ("default_option", "default_choice", "default", "recommended_option", "recommended_choice", "recommended"):
        if key in question and question.get(key) not in (None, "", False):
            raw = question.get(key)
            break
    if raw is None:
        return {}, []
    if raw is True:
        return {}, ["pending_question default_option must identify an option; boolean true is ambiguous"]
    if isinstance(raw, dict):
        for key in ("id", "value", "label"):
            if raw.get(key) not in (None, ""):
                matched = _match_option(str(raw[key]), options)
                if matched is not None:
                    return matched, []
        return {}, ["pending_question default_option object did not match any option"]
    matched = _match_option(str(raw), options)
    if matched is None:
        return {}, [f"pending_question default_option did not match any option: {raw}"]
    return matched, []


def _normalize_transition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    transition: dict[str, Any] = {}
    for key in (
        "workflow_step",
        "action",
        "tool",
        "validator",
        "tool_order",
        "next_question_id",
        "required_gate",
        "blocker_on_fail",
        "message",
    ):
        raw = value.get(key)
        if raw is not None and raw != "":
            transition[key] = raw
    if "state_patch" in value and isinstance(value["state_patch"], dict):
        transition["state_patch"] = dict(value["state_patch"])
    if "tool_order" in transition and isinstance(transition["tool_order"], str):
        transition["tool_order"] = [transition["tool_order"]]
    if "clear_pending_question" in value:
        transition["clear_pending_question"] = bool(value["clear_pending_question"])
    return transition


def answer_pending_question(
    answer: str,
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Apply a short answer to the current typed pending question.

    This is intentionally state-driven. Bare answers such as ``Y``, ``N``, ``1``,
    disk names, or URLs are only meaningful when a pending question explicitly
    accepts that answer kind.
    """
    state = load_workflow_state(session_id=session_id, state_root=state_root)
    question = state.get("pending_question") or {}
    if not question:
        return {
            "applied": False,
            "answer_kind": "unbound",
            "blockers": ["no active pending_question; ask what the user wants to confirm"],
            "state": state,
            "state_file": str(workflow_state_path(session_id=session_id, state_root=state_root)),
        }

    answer_kind, selected, blockers = _classify_pending_answer(answer, question)
    if answer_kind == "back":
        reverted = revert_workflow_state(
            steps=1,
            reason=reason or f"answer_pending_question:{question.get('id')}:back",
            session_id=session_id,
            state_root=state_root,
        )
        return {
            **reverted,
            "applied": bool(reverted.get("reverted")),
            "answer_kind": "back",
            "question_id": question.get("id", ""),
            "selected": selected,
            "blockers": [] if reverted.get("reverted") else [reverted.get("message", "no previous state")],
        }
    if blockers:
        return {
            "applied": False,
            "answer_kind": answer_kind,
            "question_id": question.get("id", ""),
            "selected": selected,
            "blockers": blockers,
            "state": state,
            "state_file": str(workflow_state_path(session_id=session_id, state_root=state_root)),
        }
    semantic_blockers = _semantic_answer_blockers(state, question, answer_kind, selected)
    if semantic_blockers:
        return {
            "applied": False,
            "answer_kind": answer_kind,
            "question_id": question.get("id", ""),
            "selected": selected,
            "blockers": semantic_blockers,
            "state": state,
            "state_file": str(workflow_state_path(session_id=session_id, state_root=state_root)),
        }

    transition = _select_transition(question, answer_kind, selected)
    patch = _patch_for_answer(state, question, answer_kind, selected, transition)
    result = update_workflow_state(
        patch,
        reason=reason or f"answer_pending_question:{question.get('id')}:{answer_kind}",
        session_id=session_id,
        state_root=state_root,
    )
    next_actions = list(result["state"].get("allowed_next_actions") or [])
    return {
        **result,
        "applied": True,
        "answer_kind": answer_kind,
        "question_id": question.get("id", ""),
        "selected": selected,
        "transition": transition,
        "blockers": [],
        "next_actions": next_actions,
    }


def _classify_pending_answer(answer: str, question: dict[str, Any]) -> tuple[str, Any, list[str]]:
    raw = _normalize_user_answer_for_question(answer, question)
    lowered = raw.lower()
    if not raw:
        return "invalid", None, ["empty answer for pending_question"]
    if lowered in BACK_VALUES:
        return "back", raw, []

    kind = str(question.get("kind") or question.get("expected_answer") or "free_text")
    option = _match_target_mode_option(raw, question.get("options") or []) if str(question.get("id") or "") == "target_mode" else None
    option = option or _match_option(raw, question.get("options") or [])
    if option is not None:
        return "choice", option, []

    if kind == "yes_no":
        if lowered in YES_VALUES:
            return "yes", True, []
        if lowered in NO_VALUES:
            return "no", False, []
        return "invalid", raw, ["expected yes or no for pending_question"]

    if kind in {"numbered_choice", "multi_select", "device"}:
        if lowered in YES_VALUES:
            default_option = question.get("default_option") or {}
            if default_option:
                return "choice", dict(default_option), []
            return "invalid", raw, ["yes requires pending_question.default_option for this choice"]
        if lowered in NO_VALUES:
            return "invalid", raw, ["no is not valid for this numbered choice; choose an option number or type a custom value"]
        if kind == "device" and question.get("manual_input_allowed"):
            if DEVICE_VALUE_RE.fullmatch(raw):
                return "manual", raw, []
            return "invalid", raw, ["expected a Linux block device such as sdb, nvme1n1, or /dev/sdd"]
        if question.get("manual_input_allowed") and not raw.isdigit():
            return "manual", raw, []
        return "invalid", raw, ["answer does not match pending_question options"]

    if kind == "url":
        if raw.startswith(("http://", "https://", "ws://", "wss://")):
            return "manual", raw, []
        return "invalid", raw, ["expected a URL with http://, https://, ws://, or wss://"]

    if kind == "confirmation":
        if lowered in YES_VALUES:
            return "yes", True, []
        if lowered in NO_VALUES:
            return "no", False, []
        if question.get("manual_input_allowed"):
            return "manual", raw, []
        return "invalid", raw, ["expected confirmation yes/no for pending_question"]

    if kind == "chain":
        if raw.isdigit():
            return "invalid", raw, ["expected a chain name, not a numbered choice"]
        known_chain = _known_chain_answer(raw, question)
        if known_chain:
            return "manual", known_chain, []
        return "invalid", raw, ["chain is not one of the currently supported chain templates; use natural language to start onboarding or correct the chain name"]

    if kind in {"manual_value", "qps_profile", "rpc_weight", "chain", "rpc_method", "evidence_apply", "free_text"}:
        if lowered in YES_VALUES:
            current_value = question.get("current_value")
            if current_value not in (None, ""):
                return "manual", current_value, []
            return "invalid", raw, ["no detected/current value is available; enter the explicit value"]
        if lowered in NO_VALUES:
            return "invalid", raw, ["enter the explicit value or use back/previous to change the previous step"]

    if kind in {"manual_value", "qps_profile", "rpc_weight", "chain", "rpc_method", "evidence_apply", "free_text"}:
        return "manual", raw, []

    return "invalid", raw, [f"unsupported pending_question kind: {kind}"]


def _semantic_answer_blockers(
    state: dict[str, Any],
    question: dict[str, Any],
    answer_kind: str,
    selected: Any,
) -> list[str]:
    qid = str(question.get("id") or "")
    field = str(question.get("field") or "")
    if answer_kind == "manual":
        field_blockers = _manual_field_blockers(question, selected)
        if field_blockers:
            return field_blockers
    if qid != "disk_accounts_choice" and field != "ACCOUNTS_DEVICE":
        if field == "mixed_weights" or qid == "mixed_weights_confirm":
            weights = _parse_rpc_weight_pairs(selected)
            if not weights:
                return ["mixed_weights must use method=weight pairs, for example eth_blockNumber=70, eth_getBalance=30"]
            total = sum(int(value) for value in weights.values())
            if total != 100:
                return [f"mixed_weights total must be 100, got {total}"]
        return []
    if answer_kind not in {"choice", "manual"}:
        return []
    value = selected.get("value") if isinstance(selected, dict) else selected
    accounts = _normalize_answer_value_for_field("ACCOUNTS_DEVICE", value)
    confirmed = state.get("confirmed_config") or {}
    ledger = confirmed.get("LEDGER_DEVICE") or confirmed.get("ledger_device")
    ledger = _normalize_answer_value_for_field("LEDGER_DEVICE", ledger)
    if accounts and ledger and accounts == ledger:
        return [
            "ACCOUNTS_DEVICE must be different from LEDGER_DEVICE when a separate accounts/state disk is configured. "
            "Reply back and choose N if accounts/state uses the same disk, or choose a different accounts device."
        ]
    return []


def _manual_field_blockers(question: dict[str, Any], selected: Any) -> list[str]:
    fields = list(question.get("fields") or [])
    field = str(question.get("field") or "").strip()
    if field:
        fields.insert(0, field)
    if not fields:
        return []
    blockers: list[str] = []
    for field_name in fields:
        normalized = _normalize_answer_value_for_field(field_name, selected)
        reason = _field_value_blocker(field_name, normalized)
        if reason:
            blockers.append(reason)
    return blockers


def _field_value_blocker(field_name: str, value: Any) -> str:
    key = str(field_name or "").strip()
    text = str(value or "").strip()
    if key in {"LEDGER_DEVICE", "ACCOUNTS_DEVICE", "ledger_device", "accounts_device"}:
        if DEVICE_VALUE_RE.fullmatch(text):
            return ""
        return f"{key} must be a Linux block device such as sdb, nvme1n1, or /dev/sdd"
    if key in {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE", "data_vol_type", "accounts_vol_type"}:
        if STORAGE_TYPE_RE.fullmatch(text) and not text.isdigit():
            return ""
        return f"{key} must be a storage type such as hyperdisk-balanced, pd-ssd, local-ssd, ssd, or nvme"
    if key in {
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_SIZE",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
        "QUICK_INITIAL_QPS",
        "QUICK_MAX_QPS",
        "QUICK_QPS_STEP",
        "QUICK_DURATION_SECONDS",
        "STANDARD_INITIAL_QPS",
        "STANDARD_MAX_QPS",
        "STANDARD_QPS_STEP",
        "STANDARD_DURATION_SECONDS",
        "INTENSIVE_INITIAL_QPS",
        "INTENSIVE_MAX_QPS",
        "INTENSIVE_QPS_STEP",
        "INTENSIVE_DURATION_SECONDS",
        "data_vol_size",
        "data_vol_max_iops",
        "data_vol_max_throughput",
        "accounts_vol_size",
        "accounts_vol_max_iops",
        "accounts_vol_max_throughput",
        "network_max_bandwidth_gbps",
    }:
        if NUMERIC_VALUE_RE.fullmatch(text):
            return ""
        return f"{key} must be numeric, with optional expected unit"
    if key in {"CLOUD_REGION", "CLOUD_ZONE", "cloud_region", "cloud_zone"}:
        if LOCATION_VALUE_RE.fullmatch(text):
            return ""
        return f"{key} must be a cloud location token such as global, asia-east1, or asia-east1-c"
    if key in {"MACHINE_TYPE", "machine_type", "NETWORK_INTERFACE", "network_interface"}:
        if SIMPLE_IDENTIFIER_RE.fullmatch(text):
            return ""
        return f"{key} must be a simple identifier"
    if key in {"BLOCKCHAIN_PROCESS_NAMES", "BLOCKCHAIN_PROCESS_NAMES_STR", "blockchain_process_names"}:
        if PROCESS_NAMES_RE.fullmatch(text):
            return ""
        return f"{key} must be one or more process/service tokens"
    if key in {"LOCAL_RPC_URL", "MAINNET_RPC_URL", "local_rpc_url", "mainnet_rpc_url"}:
        if text.startswith(("http://", "https://", "ws://", "wss://")):
            return ""
        return f"{key} must be a URL"
    return ""


def _known_chain_answer(value: Any, question: dict[str, Any]) -> str:
    return canonicalize_chain_scalar(
        value,
        known_chains=_known_chains_from_question(question),
        aliases=_chain_aliases_from_question(question),
    )


def _known_chains_from_question(question: dict[str, Any]) -> set[str]:
    values = question.get("known_chains")
    if not isinstance(values, list) or not values:
        values = _repo_chain_names()
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _chain_aliases_from_question(question: dict[str, Any]) -> dict[str, str]:
    aliases = canonical_chain_aliases()
    custom = question.get("chain_aliases")
    if isinstance(custom, dict):
        for key, value in custom.items():
            key_text = str(key or "").strip().lower()
            value_text = str(value or "").strip().lower()
            if key_text and value_text:
                aliases[key_text] = value_text
    return aliases


def _repo_chain_names() -> list[str]:
    return repo_chain_names(Path(__file__).resolve().parents[2])


def _match_option(answer: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = answer.strip().lower()
    for index, option in enumerate(options, start=1):
        candidates = {
            str(index).lower(),
            str(option.get("id", "")).strip().lower(),
            str(option.get("label", "")).strip().lower(),
            str(option.get("value", "")).strip().lower(),
        }
        if lowered in {item for item in candidates if item}:
            return dict(option)
    return None


def _match_target_mode_option(answer: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    mode = _target_mode_from_answer_text(answer)
    if not mode:
        return None
    for option in options:
        if str(option.get("value") or "").strip().lower() == mode:
            return dict(option)
    return None


def _target_mode_from_answer_text(answer: str) -> str:
    text = str(answer or "").strip().lower().replace("_", "-")
    if not text:
        return ""
    if _looks_like_composite_target_mode_request(text):
        return ""
    fake_mentioned = any(token in text for token in ("fake-node", "fakenode", "fake node", "mock"))
    real_mentioned = any(token in text for token in ("real-node", "realnode", "real node", "真实节点", "真实"))
    fake_negated = any(
        marker in text
        for marker in (
            "不使用 fake",
            "不用 fake",
            "不要 fake",
            "not fake",
            "without fake",
            "no fake",
        )
    )
    if fake_negated:
        return "real-node"
    if real_mentioned and not fake_mentioned:
        return "real-node"
    if fake_mentioned and not real_mentioned:
        return "fake-node"
    return ""


def _looks_like_composite_target_mode_request(text: str) -> bool:
    if not text:
        return False
    if any(marker in text for marker in ("切换", "改成", "换成", "change to", "switch to")):
        return False
    if any(marker in text for marker in ("smoke", "假设", "快速验证", "确认 agent", "确认框架", "can run")):
        return True
    return len(text.split()) > 8


def _select_transition(question: dict[str, Any], answer_kind: str, selected: Any) -> dict[str, Any]:
    if answer_kind == "yes":
        return dict(question.get("next_on_yes") or question.get("next_on_answer") or {})
    if answer_kind == "no":
        return dict(question.get("next_on_no") or question.get("next_on_cancel") or {})
    if answer_kind == "choice":
        if isinstance(selected, dict) and isinstance(selected.get("transition"), dict):
            return _normalize_transition(selected["transition"])
        transition = dict(question.get("next_on_choice") or question.get("next_on_answer") or {})
        per_choice = transition.get("choices") if isinstance(transition.get("choices"), dict) else {}
        value_key = str(selected.get("value", "")) if isinstance(selected, dict) else ""
        if value_key and value_key in per_choice:
            return _normalize_transition(per_choice[value_key])
        return transition
    if answer_kind == "manual":
        return dict(question.get("next_on_manual") or question.get("next_on_answer") or {})
    return {}


def _patch_for_answer(
    state: dict[str, Any],
    question: dict[str, Any],
    answer_kind: str,
    selected: Any,
    transition: dict[str, Any],
) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    _deep_merge_into(patch, question.get("state_patch_on_valid") or {})
    _deep_merge_into(patch, transition.get("state_patch") or {})
    if isinstance(selected, dict) and isinstance(selected.get("state_patch"), dict):
        _deep_merge_into(patch, selected["state_patch"])
    if answer_kind == "yes" and str(question.get("id") or "") in {"quick_assumed_smoke_confirm", "confirm_assumed_smoke"}:
        patch["assumed_for_smoke"] = True
        patch["target_mode"] = "fake-node"
        patch["allowed_next_actions"] = ["tool:run_quick_assumed_fake_node_smoke"]
    question_id = str(question.get("id") or "")
    question_field = str(question.get("field") or "")
    selected_value = str(selected.get("value") if isinstance(selected, dict) else selected).strip()
    if answer_kind == "yes" and (
        question_id in {"default_workload_confirm", "workload_confirm", "confirm_default"}
        or question_field in {"confirm_default", "confirmations", "rpc_workload_confirmed", "rpc_workload_confirmation"}
    ):
        _apply_default_workload_patch(patch, state)
    if answer_kind == "choice" and question_id == "workload_customization_choice" and selected_value == "use_defaults":
        _apply_default_workload_patch(patch, state)

    value = selected.get("value") if isinstance(selected, dict) else selected
    fields = list(question.get("fields") or [])
    field = str(question.get("field") or "").strip()
    if field:
        fields.insert(0, field)
    for field_name in fields:
        _write_answer_field(patch, field_name, value)

    if transition.get("workflow_step"):
        patch["workflow_step"] = transition["workflow_step"]

    next_actions = []
    if transition.get("tool"):
        next_actions.append(f"tool:{transition['tool']}")
    for item in list(transition.get("tool_order") or []):
        next_actions.append(f"tool:{item}")
    if transition.get("validator"):
        next_actions.append(f"validator:{transition['validator']}")
    if transition.get("next_question_id"):
        next_actions.append(f"ask:{transition['next_question_id']}")
    if next_actions:
        patch["allowed_next_actions"] = next_actions

    if transition.get("blocker_on_fail"):
        patch["blockers"] = [str(transition["blocker_on_fail"])]

    if "pending_question" not in patch and transition.get("clear_pending_question", question.get("clear_on_answer", True)):
        patch["pending_question"] = {}
    return patch


def _write_answer_field(patch: dict[str, Any], field_name: str, value: Any) -> None:
    if not field_name:
        return
    value = _normalize_answer_value_for_field(field_name, value)
    root_fields = {"target_mode", "chain", "rpc_mode", "workflow_step", "active_intent", "active_workflow"}
    if field_name in root_fields:
        patch[field_name] = _normalize_scalar_field(field_name, value)
    elif field_name == "rpc_methods":
        if isinstance(value, list):
            patch["rpc_methods"] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            patch["rpc_methods"] = [value.strip()]
    elif field_name == "mixed_weights":
        weights = _parse_rpc_weight_pairs(value)
        patch["mixed_weights"] = weights if weights else (value if isinstance(value, dict) else {"value": value})
        if weights:
            patch["rpc_methods"] = list(weights)
    elif field_name in {"benchmark_profile", "observability", "endpoint_validation", "fixture_status", "approval"}:
        patch[field_name] = value if isinstance(value, dict) else {"value": value}
    else:
        patch.setdefault("confirmed_config", {})[field_name] = value


def _normalize_answer_value_for_field(field_name: str, value: Any) -> Any:
    value = _normalize_user_answer_for_field_value(field_name, value)
    if field_name in {"LEDGER_DEVICE", "ACCOUNTS_DEVICE", "ledger_device", "accounts_device"}:
        text = str(value or "").strip()
        if text.startswith("/dev/"):
            return text.removeprefix("/dev/")
        return text
    return value


def _parse_rpc_weight_pairs(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        parsed: dict[str, int] = {}
        for key, raw_weight in value.items():
            method = str(key or "").strip()
            if not method:
                continue
            try:
                parsed[method] = int(raw_weight)
            except (TypeError, ValueError):
                return {}
        return parsed
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = {}
    for part in re.split(r"[,，;；]\s*", text):
        if not part.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z0-9_.:/-]+)\s*=\s*([0-9]{1,3})\s*%?\s*", part)
        if not match:
            return {}
        parsed[match.group(1)] = int(match.group(2))
    return parsed


def _normalize_user_answer_for_question(answer: Any, question: dict[str, Any]) -> str:
    raw = str(answer or "").strip()
    kind = str(question.get("kind") or question.get("expected_answer") or "").strip()
    if kind in {"free_text", "rpc_weight", "evidence_apply"}:
        return raw
    return _strip_scalar_wrappers(raw)


def _normalize_user_answer_for_field_value(field_name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if field_name in {"custom_rpc_params_sample", "TARGET_PARAMS", "rpc_params"}:
        return value.strip()
    return _strip_scalar_wrappers(value)


def _strip_scalar_wrappers(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"', "“", "‘"} and text[-1] in {"`", "'", '"', "”", "’"}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；"}:
        text = text[:-1].strip()
    return text


def _deep_merge_into(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in dict(incoming or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_into(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _apply_default_workload_patch(patch: dict[str, Any], state: dict[str, Any]) -> None:
    chain = str(patch.get("chain") or state.get("chain") or "").strip()
    rpc_mode = str(patch.get("rpc_mode") or state.get("rpc_mode") or "").strip().lower()
    if not chain or rpc_mode not in {"single", "mixed"}:
        return
    workload = default_workload(chain)
    if not workload.get("exists"):
        return
    if rpc_mode == "single":
        method = str(workload.get("single") or "").strip()
        if method:
            patch["rpc_methods"] = [method]
        return
    weights = {}
    for item in list(workload.get("mixed_weighted") or []):
        method = str(item.get("method") or "").strip()
        if method:
            weights[method] = int(item.get("weight") or 0)
    if weights:
        patch["mixed_weights"] = weights
        patch["rpc_methods"] = list(weights)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _push_history(state: dict[str, Any]) -> None:
    snapshot = deepcopy(state)
    snapshot.pop("history", None)
    history = list(state.get("history") or [])
    history.append(snapshot)
    state["history"] = history[-20:]
