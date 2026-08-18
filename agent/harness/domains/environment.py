"""Environment domain: cloud metadata, disks, and network configuration."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import partial
from typing import Any

from ..contracts import (
    ActionProposal,
    FailureDescriptor,
    HandlerResult,
    ResponseFragment,
    StateDelta,
    text_ref_to_dict,
)
from ..questions import (
    choice_question as _choice_question,
    manual_question as _manual_question,
    normalize_scalar,
    question_text,
)
from ..state import AgentGraphState
from ..transitions import (
    field_confirmation_revision,
    mark_field_confirmed,
    record_group_invalidations,
)

from agent.utils.redaction import redact
from agent.knowledge.entry_contract import (
    REAL_NODE_ENDPOINT_FIELDS,
    SYNC_OBSERVE_FIELDS,
)
from agent.workflows.group_registry import (
    FIELD_OWNER,
    group_for_field,
    invalidation_targets,
)

choice_question = partial(_choice_question, owner="environment")
manual_question = partial(_manual_question, owner="environment")

ENVIRONMENT_GROUPS = {"provider_deployment", "ledger_disk", "accounts_disk", "network"}

CONFIRMABLE_CONFIG_FIELDS = {
    "CLOUD_REGION",
    "CLOUD_ZONE",
    "MACHINE_TYPE",
    "LEDGER_DEVICE",
    "DATA_VOL_TYPE",
    "DATA_VOL_SIZE",
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
    "ACCOUNTS_DEVICE",
    "ACCOUNTS_VOL_TYPE",
    "ACCOUNTS_VOL_SIZE",
    "ACCOUNTS_VOL_MAX_IOPS",
    "ACCOUNTS_VOL_MAX_THROUGHPUT",
    "NETWORK_INTERFACE",
    "NETWORK_MAX_BANDWIDTH_GBPS",
    "BLOCKCHAIN_PROCESS_NAMES",
    "CHAIN_REST_URL",
    "CHAIN_INDEXER_URL",
    "CHAIN_SIDECAR_URL",
    "CHAIN_EVM_RPC_URL",
    "CHAIN_JSON_RPC_URL",
    "CHAIN_MIRROR_URL",
    "RPC_API_KEY",
}
PROPOSED_ENDPOINT_FIELDS = {
    field.env
    for field in (*REAL_NODE_ENDPOINT_FIELDS, *SYNC_OBSERVE_FIELDS)
    if field.env and field.key != "node_prometheus_metrics_url"
}
CONFIRMABLE_CONFIG_FIELDS.update(
    field.env
    for field in SYNC_OBSERVE_FIELDS
    if field.env and field.key == "node_prometheus_metrics_url"
)
SPECIAL_CONFIG_FIELDS = {
    "HAS_ACCOUNTS_DEVICE",
    "SYNC_OBSERVE_STOP_CONDITION",
    "SYNC_OBSERVE_DURATION_SECONDS",
}
CONFIG_PROPOSAL_FIELDS = (
    CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
)
WORKFLOW_DIMENSION_FIELDS = {
    "CHAIN",
    "BLOCKCHAIN_NODE",
    "TARGET_MODE",
    "WORKFLOW_MODE",
    "USE_FAKE_NODE",
    "RPC_MODE",
    "QPS_MODE",
    "BENCHMARK_MODE",
    "OBSERVABILITY",
}
POSITIVE_NUMBER_FIELDS = {
    "DATA_VOL_SIZE",
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
    "ACCOUNTS_VOL_SIZE",
    "ACCOUNTS_VOL_MAX_IOPS",
    "ACCOUNTS_VOL_MAX_THROUGHPUT",
    "NETWORK_MAX_BANDWIDTH_GBPS",
}

_CONFIG_ALIASES = {
    "cloud.region": "CLOUD_REGION",
    "region": "CLOUD_REGION",
    "cloud_region": "CLOUD_REGION",
    "cloud.zone": "CLOUD_ZONE",
    "zone": "CLOUD_ZONE",
    "cloud_zone": "CLOUD_ZONE",
    "cloud.machine_type": "MACHINE_TYPE",
    "machine": "MACHINE_TYPE",
    "machine_type": "MACHINE_TYPE",
    "instance_type": "MACHINE_TYPE",
    "disk.ledger": "LEDGER_DEVICE",
    "disk.ledger_device": "LEDGER_DEVICE",
    "storage.ledger": "LEDGER_DEVICE",
    "storage.ledger_device": "LEDGER_DEVICE",
    "ledger_device": "LEDGER_DEVICE",
    "disk.data_vol_type": "DATA_VOL_TYPE",
    "storage.data_vol_type": "DATA_VOL_TYPE",
    "data_vol_type": "DATA_VOL_TYPE",
    "disk.data_vol_size": "DATA_VOL_SIZE",
    "storage.data_vol_size": "DATA_VOL_SIZE",
    "data_vol_size": "DATA_VOL_SIZE",
    "disk.size_gib": "DATA_VOL_SIZE",
    "storage.size_gib": "DATA_VOL_SIZE",
    "disk.data_vol_max_iops": "DATA_VOL_MAX_IOPS",
    "storage.data_vol_max_iops": "DATA_VOL_MAX_IOPS",
    "data_vol_max_iops": "DATA_VOL_MAX_IOPS",
    "disk.iops": "DATA_VOL_MAX_IOPS",
    "storage.iops": "DATA_VOL_MAX_IOPS",
    "iops": "DATA_VOL_MAX_IOPS",
    "disk.data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
    "storage.data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
    "data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
    "disk.throughput": "DATA_VOL_MAX_THROUGHPUT",
    "storage.throughput": "DATA_VOL_MAX_THROUGHPUT",
    "throughput": "DATA_VOL_MAX_THROUGHPUT",
    "network.interface": "NETWORK_INTERFACE",
    "network_interface": "NETWORK_INTERFACE",
    "interface": "NETWORK_INTERFACE",
    "network.bandwidth_gbps": "NETWORK_MAX_BANDWIDTH_GBPS",
    "network_max_bandwidth_gbps": "NETWORK_MAX_BANDWIDTH_GBPS",
    "network.bandwidth": "NETWORK_MAX_BANDWIDTH_GBPS",
    "bandwidth": "NETWORK_MAX_BANDWIDTH_GBPS",
    "accounts.has_accounts_device": "HAS_ACCOUNTS_DEVICE",
    "has_accounts_device": "HAS_ACCOUNTS_DEVICE",
    "local_rpc_url": "LOCAL_RPC_URL",
    "mainnet_rpc_url": "MAINNET_RPC_URL",
    "blockchain_process_names": "BLOCKCHAIN_PROCESS_NAMES",
}
_CONFIG_ALIASES.update({
    alias: field.env
    for field in (*REAL_NODE_ENDPOINT_FIELDS, *SYNC_OBSERVE_FIELDS)
    if field.env
    for alias in (field.key, field.env.lower())
})


def _is_workflow_dimension_key(key: Any) -> bool:
    """Return whether a structured key belongs to another workflow owner."""

    normalized = str(key or "").strip().replace("-", "_").upper()
    leaf = normalized.rsplit(".", 1)[-1]
    return leaf in WORKFLOW_DIMENSION_FIELDS


def _mapped_config_field(key: Any) -> str:
    """Resolve a known config field from a full structured path or its leaf."""

    text = str(key or "").strip()
    if not text:
        return ""
    allowed = CONFIG_PROPOSAL_FIELDS
    normalized = text.upper()
    alias = _CONFIG_ALIASES.get(text.lower().replace("-", "_"))
    if normalized in allowed:
        return normalized
    if alias:
        return alias
    leaf = text.rsplit(".", 1)[-1]
    normalized_leaf = leaf.upper()
    leaf_alias = _CONFIG_ALIASES.get(leaf.lower().replace("-", "_"))
    if normalized_leaf in allowed:
        return normalized_leaf
    return leaf_alias or ""


def _is_structured_config_key(key: Any) -> bool:
    """Return whether a key has JSON/YAML/env field structure.

    This is an input-shape contract, not an intent heuristic. In particular,
    prose prefixes and URL schemes cannot become inferred configuration keys.
    """

    return bool(
        re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*",
            str(key or "").strip(),
        )
    )


def build_config_proposal(action: dict[str, Any]) -> dict[str, Any]:
    """Normalize a resolver-proposed mixed-format configuration payload."""

    raw_values = action.get("config_values") if isinstance(action.get("config_values"), dict) else {}
    config_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    allowed = CONFIG_PROPOSAL_FIELDS
    for key, value in raw_values.items():
        normalized_key = str(key or "").strip().upper()
        mapped_key, mapped_value = normalize_proposed_config_value(normalized_key, value)
        if mapped_key in allowed and mapped_value not in {"", None}:
            config_values[mapped_key] = mapped_value
        elif (
            normalized_key
            and normalized_key not in allowed
            and _is_structured_config_key(key)
            and not _is_workflow_dimension_key(normalized_key)
        ):
            unmapped[normalized_key] = value
    raw_unmapped = action.get("unmapped_values")
    if isinstance(raw_unmapped, dict):
        for key, value in raw_unmapped.items():
            text_key = str(key or "").strip()
            mapped_key = _mapped_config_field(text_key)
            if mapped_key:
                final_key, final_value = normalize_proposed_config_value(mapped_key, value)
                if final_key and final_value not in {"", None}:
                    config_values.setdefault(final_key, final_value)
                continue
            if (
                text_key
                and _is_structured_config_key(text_key)
                and not _is_workflow_dimension_key(text_key)
            ):
                unmapped[text_key] = value
    proposal = {
        "config_values": config_values,
        "unmapped_values": unmapped,
        "source_format": _strip_scalar(str(action.get("source_format") or "mixed")),
        "reason": _strip_scalar(str(action.get("reason") or "")),
    }
    conflicts = _normalize_conflicts(action.get("conflicts"))
    if conflicts:
        proposal["conflicts"] = conflicts
    return proposal


def apply_environment_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply environment-owned model actions through typed state contracts."""

    if action.action_type == "set_accounts_presence":
        if "has_accounts_device" not in action.arguments:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.environment.failure.accounts_presence_missing",
                    source=__name__,
                )
            )
        next_state: AgentGraphState = deepcopy(state)
        confirmed = next_state.setdefault("confirmed_config", {})
        has_accounts = bool(action.arguments.get("has_accounts_device"))
        confirmed["has_accounts_device"] = has_accounts
        mark_field_confirmed(next_state, "has_accounts_device")
        if not has_accounts:
            for key in (
                "ACCOUNTS_DEVICE",
                "ACCOUNTS_VOL_TYPE",
                "ACCOUNTS_VOL_SIZE",
                "ACCOUNTS_VOL_MAX_IOPS",
                "ACCOUNTS_VOL_MAX_THROUGHPUT",
            ):
                confirmed.pop(key, None)
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            invalidated_groups=invalidation_targets("accounts_disk"),
            reconfigured_groups=("accounts_disk",),
            clear_pending=True,
            next_group="accounts_disk",
            completion="completed",
        )
    if action.action_type == "propose_config_values":
        proposal = build_config_proposal(dict(action.arguments))
        proposal = _without_already_confirmed_config_values(state, proposal)
        # Unknown keys are reviewable only alongside at least one recognized
        # configuration value. Otherwise log lines, HTTP headers, and protocol
        # examples such as ``RuntimeError: ...`` would become configuration
        # transactions and compete with their actual analysis owner.
        if not proposal.get("config_values"):
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                completion="completed",
            )
        pending = state.get("pending_question") or {}
        current = (state.get("inferred_config") or {}).get("pending_review")
        if (
            str(pending.get("id") or "") == "inferred_config_review"
            and isinstance(current, dict)
        ):
            proposal = merge_config_proposals(current, proposal)
        return propose_config_assignments_for_review(
            state,
            proposal,
            consumed_action_ids=(action.action_id,),
        )
    return HandlerResult(
        blocker=FailureDescriptor(
            code="harness.environment.failure.unsupported_action",
            arguments={"action_type": action.action_type},
            source=__name__,
        )
    )


def _without_already_confirmed_config_values(
    state: AgentGraphState,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Remove proposal fields already satisfied by authoritative state.

    Deferred structured proposals can outlive an intervening typed question.
    The environment owner consumes equal values idempotently while preserving
    conflicting values and unrelated fields for explicit review.
    """

    confirmed = state.get("confirmed_config") or {}
    values = dict(proposal.get("config_values") or {})
    retained: dict[str, Any] = {}
    for key, proposed in values.items():
        current_key = "has_accounts_device" if key == "HAS_ACCOUNTS_DEVICE" else key
        if current_key not in confirmed:
            retained[key] = proposed
            continue
        _, normalized_current = normalize_proposed_config_value(
            key,
            confirmed[current_key],
        )
        _, normalized_proposed = normalize_proposed_config_value(key, proposed)
        if normalized_current != normalized_proposed:
            retained[key] = proposed
    normalized = dict(proposal)
    normalized["config_values"] = retained
    return normalized


def propose_config_assignments_for_review(
    state: AgentGraphState,
    proposal: dict[str, Any],
    *,
    consumed_action_ids: tuple[str, ...] = (),
) -> HandlerResult:
    """Open the single review transaction for recognized config candidates."""

    normalized = build_config_proposal(proposal)
    if not normalized.get("config_values"):
        return HandlerResult(
            consumed_action_ids=consumed_action_ids,
            completion="completed",
        )
    next_state: AgentGraphState = deepcopy(state)
    next_state.setdefault("inferred_config", {})["pending_review"] = normalized
    review_group = str(next_state.get("active_group") or "opening")
    question = config_proposal_review_question(
        review_group,
        normalized,
        language=str(next_state.get("language") or "en"),
    )
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        consumed_action_ids=consumed_action_ids,
        pending_question=question,
        completion="completed",
    )


def extract_structured_config_proposal(text: str) -> dict[str, Any] | None:
    """Extract known config facts from explicit JSON, YAML, or env text.

    Natural-language and table extraction remains model supplied through
    :func:`build_config_proposal`; this deterministic fallback is not an
    intent router and never applies values without a review.
    """

    raw = str(text or "")
    if "\n" not in raw and "{" not in raw and ":" not in raw:
        return None
    config_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    config_values.update(parse_known_config_assignments(raw))

    json_values, json_unmapped = extract_json_config_values(raw)
    config_values.update(json_values)
    unmapped.update(json_unmapped)

    yaml_values, yaml_unmapped = extract_yaml_like_config_values(raw)
    config_values.update(yaml_values)
    unmapped.update(yaml_unmapped)

    if not config_values and not unmapped:
        return None
    action: dict[str, Any] = {
        "type": "propose_config_values",
        "source_format": "mixed",
        "config_values": config_values,
        "unmapped_values": unmapped,
        "confidence": "high",
        "reason": "structured config paste",
    }
    return action


def extract_structured_input_candidates(text: str) -> dict[str, Any] | None:
    """Return syntax-derived facts for the semantic planner.

    These facts are not executable actions. They keep parsing deterministic
    while leaving intent, workflow ownership, and transactional admission to
    the single LLM plan. Unknown keys stay visible instead of disappearing.
    """

    flattened: dict[str, Any] = {}
    semantic_fields: dict[str, Any] = {}
    detected_formats: set[str] = set()
    json_values = _extract_json_flat_values(text)
    source_evidence: dict[str, str] = {}
    if json_values:
        detected_formats.add("json")
        flattened.update(json_values)
        semantic_fields.update(_extract_json_semantic_values(text))
        source_evidence.update(_extract_json_semantic_source_evidence(text))
    else:
        yaml_values = _extract_yaml_flat_values(text)
        if yaml_values:
            detected_formats.add("yaml")
            flattened.update(yaml_values)
            semantic_fields.update(yaml_values)
    env_values: dict[str, Any] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        for key, value in iter_config_assignments(line):
            env_values[key] = value
    if env_values:
        detected_formats.add("env")
        flattened.update(env_values)
        semantic_fields.update(env_values)
    if not flattened:
        return None

    config_values, workflow_values, unmapped_values = classify_flat_input_values(flattened)
    if not config_values and not workflow_values and not unmapped_values:
        return None
    field_candidates: list[dict[str, Any]] = []
    for raw_path, raw_value in semantic_fields.items():
        path = str(raw_path or "").strip()
        if not path:
            continue
        field_config, field_workflow, field_unmapped = classify_flat_input_values(
            {path: raw_value}
        )
        if field_config:
            kind = "config"
            canonical = next(iter(field_config))
        elif field_workflow:
            kind = "workflow"
            canonical = next(iter(field_workflow))
        elif field_unmapped:
            kind = "unmapped"
            canonical = path
        else:
            continue
        field_candidates.append({
            "source_path": path,
            "raw_value": raw_value,
            "candidate_kind": kind,
            "canonical_key": canonical,
            **(
                {"source_evidence": source_evidence[path]}
                if source_evidence.get(path)
                else {}
            ),
        })
    return {
        "config_values": config_values,
        "workflow_values": workflow_values,
        "unmapped_values": unmapped_values,
        "field_candidates": field_candidates,
        "source_format": next(iter(detected_formats)) if len(detected_formats) == 1 else "mixed",
    }


def extract_config_proposal_fragment(text: str) -> dict[str, Any] | None:
    """Extract one structured or terminal-style fragment for a pending review."""

    proposal_action = extract_structured_config_proposal(text)
    if proposal_action is not None:
        proposal = build_config_proposal(proposal_action)
        # A structured payload that contains no recognized configuration key
        # belongs to another semantic owner (for example JSON-RPC evidence).
        # It must not be captured merely because a config review is pending.
        if proposal.get("config_values"):
            return proposal
        return None
    direct = parse_known_config_assignments(text)
    if not direct:
        return None
    return build_config_proposal({
        "type": "propose_config_values",
        "source_format": "mixed",
        "config_values": direct,
        "unmapped_values": {},
        "confidence": "high",
        "reason": "additional config line",
    })


def merge_config_proposals(
    current: dict[str, Any] | None,
    incoming: dict[str, Any],
    *,
    reason: str = "additional config fragments",
) -> dict[str, Any]:
    """Merge review fragments, preserving last-value-wins and conflict evidence."""

    current = current if isinstance(current, dict) else {}
    current_values = dict(current.get("config_values") or {})
    current_unmapped = dict(current.get("unmapped_values") or {})
    conflicts = _normalize_conflicts(current.get("conflicts"))
    current_values.update(dict(incoming.get("config_values") or {}))
    current_unmapped.update(dict(incoming.get("unmapped_values") or {}))
    conflicts.extend(item for item in _normalize_conflicts(incoming.get("conflicts")) if item not in conflicts)
    merged: dict[str, Any] = {
        "config_values": current_values,
        "unmapped_values": current_unmapped,
        "source_format": "mixed",
        "reason": reason,
    }
    if conflicts:
        merged["conflicts"] = conflicts
    return merged


def merge_config_proposal_from_text(
    current: dict[str, Any] | None,
    text: str,
    *,
    reason: str = "additional config fragments",
) -> dict[str, Any] | None:
    """Return a merged proposal without mutating graph or coordinator state."""

    incoming = extract_config_proposal_fragment(text)
    if incoming is None or not incoming.get("config_values") and not incoming.get("unmapped_values"):
        return None
    return merge_config_proposals(current, incoming, reason=reason)


def config_proposal_review_question(group: str, proposal: dict[str, Any], *, language: str = "en") -> dict[str, Any]:
    """Build the typed Y/N review contract; rendering remains coordinator-owned."""

    question = choice_question(
        group,
        "inferred_config_review",
        question_text(
            "question.environment.inferred_config_review.prompt",
            **_config_proposal_prompt_arguments(proposal),
        ),
        field="inferred_config_review",
        kind="yes_no",
        manual_input_allowed=False,
        options=[
            {
                "label": question_text("question.common.option.yes"),
                "value": True,
                "expected_patch": {"inferred_config.pending_review": {}},
            },
            {
                "label": question_text("question.common.option.no"),
                "value": False,
                "expected_patch": {"inferred_config.pending_review": {}},
            },
        ],
        accepted_action_types=("propose_config_values",),
        queue_barrier=True,
        owner="environment",
    )
    question["supersedes_action_types"] = ["propose_config_values"]
    question["barrier_policy"] = "explicit_detour_only"
    return question


def reconstruct_environment_question(
    state: AgentGraphState,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Rebuild an environment-owned question from current authoritative state."""

    group = str(identity.get("group") or "").strip()
    question_id = str(
        identity.get("question_id") or identity.get("id") or ""
    ).strip()
    field = str(identity.get("field") or "").strip()
    if question_id == "inferred_config_review":
        proposal = (state.get("inferred_config") or {}).get("pending_review")
        if not isinstance(proposal, dict) or not proposal.get("config_values"):
            return None
        return config_proposal_review_question(
            group,
            proposal,
            language=str(state.get("language") or "en"),
        )
    reconfiguration_target = str(
        identity.get("reconfiguration_target_field") or ""
    ).strip()
    if field and reconfiguration_target == field:
        baseline_revision = int(
            identity.get("reconfiguration_baseline_revision") or 0
        )
        current_revision = field_confirmation_revision(
            state,
            field,
            group=group,
        )
        if current_revision > baseline_revision:
            return None
        question = question_for_environment_field(state, group, field)
        if question and str(question.get("id") or "") == question_id:
            return question
    question = question_for_environment(state, group)
    if question and str(question.get("id") or "") == question_id:
        return question
    return None


def _config_proposal_prompt_arguments(proposal: dict[str, Any]) -> dict[str, str]:
    """Build language-neutral arguments for the central review template."""

    config_values = proposal.get("config_values") if isinstance(proposal.get("config_values"), dict) else {}
    unmapped = proposal.get("unmapped_values") if isinstance(proposal.get("unmapped_values"), dict) else {}
    conflicts = _normalize_conflicts(proposal.get("conflicts"))
    display_config_values = redact(config_values)
    display_unmapped = redact(unmapped)
    display_conflicts = redact(conflicts)
    mapped_text = "\n".join(
        f"- {key}: `{display_config_values[key]}`"
        for key in sorted(display_config_values)
    ) or "<none>"
    unmapped_text = "\n".join(
        f"- {key}: `{display_unmapped[key]}`"
        for key in sorted(display_unmapped)
    ) or "<none>"
    conflicts_text = "\n".join(f"- {item}" for item in display_conflicts) or "<none>"
    return {
        "mapped_values": mapped_text,
        "unmapped_values": unmapped_text,
        "conflicts": conflicts_text,
        "reason": str(proposal.get("reason") or "<none>"),
    }


def apply_direct_config_assignments(state: AgentGraphState, config_values: dict[str, Any]) -> HandlerResult:
    """Apply explicit ``KEY=value`` facts and return structured outcome evidence."""

    next_state: AgentGraphState = deepcopy(state)
    local_values, delegated = _partition_reviewed_config_values(config_values)
    outcome = _apply_config_values(next_state, local_values)
    previous_invalidated = set(state.get("invalidated_groups") or [])
    next_invalidated = set(next_state.get("invalidated_groups") or [])
    followups: tuple[dict[str, Any], ...] = ()
    followups = _reviewed_config_followups(delegated, outcome["sync_options"])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(next_invalidated - previous_invalidated)),
        reconfigured_groups=tuple(sorted(previous_invalidated - next_invalidated)),
        clear_pending=True,
        evidence=({"type": "direct_config_application", **outcome},),
        followup_actions=followups,
        completion="in_progress",
    )


def apply_inferred_config_review(
    state: AgentGraphState,
    accepted: bool,
    proposal: dict[str, Any] | None = None,
) -> HandlerResult:
    """Resolve a reviewed proposal without touching queues or rendering output."""

    next_state: AgentGraphState = deepcopy(state)
    inferred = next_state.setdefault("inferred_config", {})
    pending = inferred.pop("pending_review", {}) if isinstance(inferred.get("pending_review"), dict) else {}
    reviewed = proposal if isinstance(proposal, dict) else pending
    if not accepted:
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            clear_pending=True,
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.environment.inferred_config_discarded",
                    source=__name__,
                ),
            ),
            evidence=({
                "type": "inferred_config_review",
                "accepted": False,
                "unmapped_values": dict(reviewed.get("unmapped_values") or {}),
                "conflicts": _normalize_conflicts(reviewed.get("conflicts")),
            },),
            completion="unchanged",
        )

    local_values, delegated = _partition_reviewed_config_values(
        dict(reviewed.get("config_values") or {})
    )
    outcome = _apply_config_values(next_state, local_values)
    accepted_review = {
        "applied": outcome["applied"],
        "delegated": delegated,
        "endpoint_proposals": outcome["endpoint_proposals"],
        "unmapped_values": dict(reviewed.get("unmapped_values") or {}),
        "source_format": reviewed.get("source_format") or "",
    }
    conflicts = _normalize_conflicts(reviewed.get("conflicts"))
    if conflicts:
        accepted_review["conflicts"] = conflicts
    inferred.setdefault("accepted_reviews", []).append(accepted_review)
    response_fragments: list[ResponseFragment] = []
    if outcome["applied"]:
        details = ", ".join(f"{key}={value}" for key, value in sorted(outcome["applied"].items()))
        response_fragments.append(
            ResponseFragment(
                kind="message",
                message_id="harness.environment.inferred_config_applied",
                arguments={"details": details},
                source=__name__,
            )
        )
    if outcome["endpoint_proposals"]:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(outcome["endpoint_proposals"].items())
        )
        response_fragments.append(
            ResponseFragment(
                kind="message",
                message_id="harness.environment.endpoint_candidates_saved",
                arguments={"details": details},
                source=__name__,
            )
        )
    if outcome["invalid"]:
        details = ", ".join(f"{key}={value}" for key, value in sorted(outcome["invalid"].items()))
        response_fragments.append(
            ResponseFragment(
                kind="warning",
                message_id="harness.environment.invalid_candidates_ignored",
                arguments={"details": details},
                source=__name__,
            )
        )
    if accepted_review["unmapped_values"]:
        response_fragments.append(
            ResponseFragment(
                kind="evidence",
                message_id="harness.environment.unmapped_values_retained",
                source=__name__,
            )
        )
    followups: tuple[dict[str, Any], ...] = ()
    followups = _reviewed_config_followups(delegated, outcome["sync_options"])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(set(next_state.get("invalidated_groups") or []) - set(state.get("invalidated_groups") or []))),
        reconfigured_groups=tuple(sorted(set(state.get("invalidated_groups") or []) - set(next_state.get("invalidated_groups") or []))),
        clear_pending=True,
        response_fragments=tuple(response_fragments),
        evidence=({
            "type": "inferred_config_review",
            "accepted": True,
            **outcome,
            "unmapped_values": accepted_review["unmapped_values"],
            "conflicts": conflicts,
        },),
        followup_actions=followups,
        completion="in_progress",
    )


def _partition_reviewed_config_values(
    config_values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split reviewed fields by their registry owner before state mutation."""

    local: dict[str, Any] = {}
    delegated: dict[str, dict[str, Any]] = {}
    for raw_key, value in config_values.items():
        key, _normalized = normalize_proposed_config_value(
            str(raw_key or "").strip().upper(),
            value,
        )
        owner = FIELD_OWNER.get(key, "") if key in CONFIRMABLE_CONFIG_FIELDS else ""
        if owner and owner != "environment":
            delegated.setdefault(owner, {})[key] = value
        else:
            local[key or str(raw_key)] = value
    return local, delegated


def _reviewed_config_followups(
    delegated: Mapping[str, Mapping[str, Any]],
    sync_options: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Lower one accepted review into owner-specific internal commits."""

    actions: list[dict[str, Any]] = []
    action_types = {
        "chain_rpc": "apply_reviewed_chain_rpc_config",
        "sync_observe": "apply_reviewed_sync_observe_config",
    }
    for owner in ("chain_rpc", "sync_observe"):
        values = dict(delegated.get(owner) or {})
        if values:
            actions.append({
                "type": action_types[owner],
                "config_values": values,
            })
    if sync_options:
        actions.append({"type": "set_sync_observe_options", **dict(sync_options)})
    return tuple(actions)


def apply_reviewed_owned_config_values(
    state: AgentGraphState,
    action: ActionProposal,
    *,
    owner: str,
) -> HandlerResult:
    """Commit one reviewed field subset through its authoritative owner."""

    values = dict(action.arguments.get("config_values") or {})
    if not values or any(FIELD_OWNER.get(str(key).upper(), "") != owner for key in values):
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.environment.failure.reviewed_config_owner_mismatch",
                arguments={"owner": owner},
                source=__name__,
            )
        )
    next_state: AgentGraphState = deepcopy(state)
    outcome = _apply_config_values(next_state, values)
    if outcome["endpoint_proposals"] or outcome["sync_options"]:
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.environment.failure.reviewed_config_commit_invalid",
                arguments={"owner": owner},
                source=__name__,
            )
        )
    response_fragments: tuple[ResponseFragment, ...] = ()
    if outcome["applied"]:
        details = ", ".join(
            f"{key}={value}"
            for key, value in sorted(outcome["applied"].items())
        )
        response_fragments = (
            ResponseFragment(
                kind="message",
                message_id="harness.environment.inferred_config_applied",
                arguments={"details": details},
                source=__name__,
            ),
        )
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        consumed_action_ids=(action.action_id,),
        invalidated_groups=tuple(
            sorted(
                set(next_state.get("invalidated_groups") or [])
                - set(state.get("invalidated_groups") or [])
            )
        ),
        response_fragments=response_fragments,
        evidence=({
            "type": "reviewed_config_owner_commit",
            "owner": owner,
            **outcome,
        },),
        completion="in_progress",
    )


def extract_json_config_values(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return map_flat_config_values(_extract_json_flat_values(text))


def _extract_json_flat_values(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return flatten_mapping(parsed)


def _extract_json_semantic_values(text: str) -> dict[str, Any]:
    """Preserve JSON composite fields without reparsing JSON as YAML-like text."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return flatten_mapping(parsed)
    return _flatten_semantic_mapping(parsed)


def _extract_json_semantic_source_evidence(text: str) -> dict[str, str]:
    """Return exact JSON value slices keyed by their semantic source path."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    source = text[start : end + 1]
    decoder = json.JSONDecoder()
    output: dict[str, str] = {}

    def skip_space(index: int) -> int:
        while index < len(source) and source[index].isspace():
            index += 1
        return index

    def visit_object(index: int, prefix: str = "") -> int:
        index = skip_space(index)
        if index >= len(source) or source[index] != "{":
            raise json.JSONDecodeError("expected object", source, index)
        index = skip_space(index + 1)
        if index < len(source) and source[index] == "}":
            return index + 1
        while index < len(source):
            key, key_end = decoder.raw_decode(source, index)
            if not isinstance(key, str):
                raise json.JSONDecodeError("expected object key", source, index)
            index = skip_space(key_end)
            if index >= len(source) or source[index] != ":":
                raise json.JSONDecodeError("expected colon", source, index)
            value_start = skip_space(index + 1)
            value, value_end = decoder.raw_decode(source, value_start)
            path = f"{prefix}.{key}" if prefix else key
            output[path] = source[value_start:value_end]
            if isinstance(value, dict):
                visit_object(value_start, path)
            index = skip_space(value_end)
            if index < len(source) and source[index] == ",":
                index = skip_space(index + 1)
                continue
            if index < len(source) and source[index] == "}":
                return index + 1
            raise json.JSONDecodeError("expected comma or object end", source, index)
        raise json.JSONDecodeError("unterminated object", source, index)

    try:
        visit_object(0)
    except (json.JSONDecodeError, TypeError):
        return {}
    return output


def _flatten_semantic_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        output[next_key] = item
        if isinstance(item, dict):
            output.update(_flatten_semantic_mapping(item, next_key))
    return output


def extract_yaml_like_config_values(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return map_flat_config_values(_extract_yaml_flat_values(text))


def _extract_yaml_flat_values(text: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith(("#", "- ")):
            continue
        if ":" not in raw_line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.strip().split(":", 1)
        key = key.strip().strip("\"'")
        value = value.strip().strip(",").strip()
        if not key or not _is_structured_config_key(key):
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not value:
            stack.append((indent, key))
            continue
        path = ".".join([item[1] for item in stack] + [key])
        flattened[path] = value.strip("\"'")
    return flattened


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_mapping(item, next_key))
    else:
        output[prefix] = value
    return output


def map_flat_config_values(flattened: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_values, _, unmapped = classify_flat_input_values(flattened)
    return config_values, unmapped


def classify_flat_input_values(
    flattened: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Classify parsed fields without claiming workflow intent."""

    config_values: dict[str, Any] = {}
    workflow_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    for raw_key, value in flattened.items():
        key = str(raw_key or "").strip()
        normalized = key.upper()
        mapped_key = _mapped_config_field(key)
        scalar = _strip_scalar(str(value)).strip().strip("\"'")
        if not scalar:
            continue
        if mapped_key:
            final_key, final_value = normalize_proposed_config_value(mapped_key, scalar)
            if final_key and final_value not in {"", None}:
                config_values[final_key] = final_value
        elif _is_workflow_dimension_key(key):
            workflow_values[normalized.rsplit(".", 1)[-1]] = scalar
        elif _is_structured_config_key(key):
            unmapped[key] = scalar
    return config_values, workflow_values, unmapped


def parse_known_config_assignments(text: str) -> dict[str, Any]:
    """Parse only explicit known AnyChain assignments from terminal input."""

    values: dict[str, Any] = {}
    allowed = CONFIG_PROPOSAL_FIELDS
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        for key, raw_value in iter_config_assignments(line):
            if key not in allowed:
                continue
            final_key, value = normalize_proposed_config_value(key, raw_value)
            if final_key and value not in {"", None}:
                values[final_key] = value
    return values


def is_assignment_only_config_text(text: str) -> bool:
    """Return whether a single turn contains only explicit config assignments.

    The coordinator may apply this shape without an LLM round trip.  Mixed
    natural language plus assignments must stay on the full action-planning
    path so chain, mode, workload, navigation, and consultation demands are not
    discarded by the optimization.
    """

    raw = str(text or "").strip()
    if not raw or "\n" in raw:
        return False
    key = r"[A-Za-z_][A-Za-z0-9_]*"
    value = r'(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\s,，;；]+)'
    assignment = rf"(?:export\s+)?{key}\s*=\s*{value}"
    return bool(re.fullmatch(rf"\s*{assignment}(?:\s*[,，;；]\s*{assignment})*\s*", raw))


def iter_config_assignments(line: str) -> list[tuple[str, str]]:
    """Return comma-, semicolon-, or full-width-separated ``KEY=value`` pairs."""

    pattern = re.compile(
        r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<value>.*?)(?=\s*[,，;；]\s*[A-Za-z_][A-Za-z0-9_]*\s*=|$)"
    )
    pairs: list[tuple[str, str]] = []
    for match in pattern.finditer(line):
        key = match.group("key").strip().upper()
        value = _strip_scalar(match.group("value"))
        if key and value:
            pairs.append((key, value))
    return pairs


def normalize_proposed_config_value(key: str, value: Any) -> tuple[str, Any]:
    """Normalize inferred, structured, and direct values to one contract."""

    normalized_key = str(key or "").strip().upper()
    if normalized_key == "HAS_ACCOUNTS_DEVICE" and isinstance(value, bool):
        return normalized_key, value
    scalar = _strip_scalar(str(value)).strip().strip("\"'")
    if not normalized_key or not scalar:
        return normalized_key, ""
    key_aliases = {
        "SYNC_OBSERVE_DURATION": "SYNC_OBSERVE_DURATION_SECONDS",
        "SYNC_OBSERVE_DURATION_SEC": "SYNC_OBSERVE_DURATION_SECONDS",
        "SYNC_OBSERVE_STOP_CONDITIONS": "SYNC_OBSERVE_STOP_CONDITION",
    }
    normalized_key = key_aliases.get(normalized_key, normalized_key)
    if normalized_key == "HAS_ACCOUNTS_DEVICE":
        if scalar.lower() in {"true", "1"}:
            return normalized_key, True
        if scalar.lower() in {"false", "0"}:
            return normalized_key, False
        return normalized_key, scalar
    if normalized_key in {"DATA_VOL_SIZE", "ACCOUNTS_VOL_SIZE"}:
        size = _size_to_gib(scalar)
        return normalized_key, size or _signed_number_text(scalar)
    if normalized_key in {
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
    }:
        return normalized_key, _signed_number_text(scalar)
    if normalized_key in {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE"}:
        return normalized_key, scalar.lower()
    if normalized_key == "SYNC_OBSERVE_DURATION_SECONDS":
        return normalized_key, scalar
    if normalized_key == "SYNC_OBSERVE_STOP_CONDITION":
        lowered = scalar.lower()
        aliases = {
            "duration": "duration", "fixed": "duration", "固定时长": "duration", "固定": "duration",
            "until_stopped": "until_stopped", "stopped": "until_stopped", "手动停止": "until_stopped", "一直运行": "until_stopped",
            "until_synced": "until_synced", "synced": "until_synced", "同步完成": "until_synced",
        }
        return normalized_key, aliases.get(lowered, lowered)
    return normalized_key, scalar


def _apply_config_values(
    state: AgentGraphState,
    config_values: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    confirmed = state.setdefault("confirmed_config", {})
    applied: dict[str, Any] = {}
    endpoint_saved: dict[str, Any] = {}
    invalid: dict[str, Any] = {}
    sync_options: dict[str, Any] = {}
    for key, value in config_values.items():
        key, scalar = normalize_proposed_config_value(str(key or "").strip().upper(), value)
        if scalar in {"", None}:
            continue
        if key in CONFIRMABLE_CONFIG_FIELDS:
            if key in POSITIVE_NUMBER_FIELDS and _invalid_positive_number(scalar):
                invalid[key] = scalar
            else:
                confirmed[key] = scalar
                applied[key] = scalar
                mark_field_confirmed(state, key)
        elif key in PROPOSED_ENDPOINT_FIELDS:
            endpoint_proposals = state.setdefault("endpoint_evidence", {}).setdefault(
                "proposed_values", {}
            )
            endpoint_proposals[key] = scalar
            endpoint_saved[key] = scalar
        elif key == "HAS_ACCOUNTS_DEVICE":
            has_accounts = bool(scalar)
            confirmed["has_accounts_device"] = has_accounts
            mark_field_confirmed(state, "has_accounts_device")
            if not has_accounts:
                for accounts_key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
                    confirmed.pop(accounts_key, None)
            applied["has_accounts_device"] = has_accounts
        elif key == "SYNC_OBSERVE_STOP_CONDITION":
            sync_options["sync_observe_stop_condition"] = str(scalar).lower()
        elif key == "SYNC_OBSERVE_DURATION_SECONDS":
            try:
                duration_seconds = int(scalar)
                if duration_seconds <= 0:
                    invalid[key] = scalar
                else:
                    sync_options["sync_observe_duration_seconds"] = duration_seconds
            except (TypeError, ValueError):
                invalid[key] = scalar
    for changed_group in sorted({group_for_field(key) for key in applied} - {""}):
        record_group_invalidations(state, changed_group)
    return {
        "applied": applied,
        "endpoint_proposals": endpoint_saved,
        "invalid": invalid,
        "sync_options": sync_options,
    }


def _normalize_conflicts(value: Any) -> list[str]:
    if isinstance(value, dict):
        items = [f"{key}: {item}" for key, item in value.items()]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif value in {None, ""}:
        items = []
    else:
        items = [value]
    output: list[str] = []
    for item in items:
        text = _strip_scalar(str(item))
        if text and text not in output:
            output.append(text)
    return output


def _invalid_positive_number(value: Any) -> bool:
    text = str(value or "").strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return True
    return float(text) <= 0


def _size_to_gib(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:I?B)?", text)
    if not match:
        return ""
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "T":
        number *= 1024
    elif unit == "M":
        number /= 1024
    elif unit == "K":
        number /= 1024 * 1024
    return str(int(number)) if number >= 1 else str(round(number, 3))


def _first_number_text(value: Any) -> str:
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value or ""))
    if not match:
        return ""
    number = float(match.group(0))
    return str(int(number)) if number.is_integer() else str(number)


def _signed_number_text(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if match:
        return match.group(0)
    return _first_number_text(text)


def _strip_scalar(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"'} and text[-1] in {"`", "'", '"'}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；", "、"}:
        text = text[:-1].strip()
    return text


def question_for_environment(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    if group not in ENVIRONMENT_GROUPS:
        return None
    confirmed = state.get("confirmed_config") or {}
    discovery = state.get("discovery") or {}
    if group == "provider_deployment":
        cloud = discovery.get("cloud") or {}
        inferred = state.get("inferred_config") or {}
        for env_key, prompt_key, detected_key in (
            ("CLOUD_REGION", "cloud_region", "region"),
            ("CLOUD_ZONE", "cloud_zone", "zone"),
            ("MACHINE_TYPE", "machine_type", "machine_type"),
        ):
            if confirmed.get(env_key):
                continue
            detected = normalize_scalar(str(cloud.get(detected_key) or ""))
            if detected and not inferred.get(f"{env_key}_manual_required"):
                return choice_question(
                    group,
                    env_key,
                    question_text(
                        "question.environment.detected_value.prompt",
                        field=env_key,
                        value=detected,
                    ),
                    field=env_key,
                    kind="yes_no",
                    manual_input_allowed=True,
                    structured_config_key=env_key,
                    validation={"value_type": "scalar_token"},
                    options=[
                        {
                            "label": question_text("question.common.option.yes"),
                            "value": detected,
                        },
                        {
                            "label": question_text("question.common.option.no"),
                            "value": "__manual__",
                            "manual_entry": True,
                            "expected_patch": {f"inferred_config.{env_key}_manual_required": True},
                        },
                    ],
                )
            return manual_question(
                group,
                env_key,
                question_text(f"question.environment.{prompt_key}.prompt"),
                field=env_key,
                structured_config_key=env_key,
                validation={"value_type": "scalar_token"},
            )
        return None
    if group == "ledger_disk":
        return _disk_question(state, prefix="DATA", device_key="LEDGER_DEVICE", group=group)
    if group == "accounts_disk":
        if "has_accounts_device" not in confirmed:
            return choice_question(
                group,
                "has_accounts_device",
                question_text("question.environment.has_accounts_device.prompt"),
                field="has_accounts_device",
                kind="yes_no",
                options=[
                    {"label": question_text("question.common.option.yes"), "value": True},
                    {"label": question_text("question.common.option.no"), "value": False},
                ],
            )
        if confirmed.get("has_accounts_device"):
            return _disk_question(state, prefix="ACCOUNTS", device_key="ACCOUNTS_DEVICE", group=group)
        return None
    if group == "network":
        if not confirmed.get("NETWORK_INTERFACE"):
            network = discovery.get("network") or {}
            default = normalize_scalar(str(network.get("default_interface") or ""))
            interfaces = _usable_interfaces(list(network.get("interfaces") or []), default)
            if interfaces:
                return choice_question(
                    group,
                    "network_interface",
                    question_text("question.environment.network_interface.prompt"),
                    field="NETWORK_INTERFACE",
                    kind="device",
                    manual_input_allowed=True,
                    structured_config_key="NETWORK_INTERFACE",
                    validation={"value_type": "scalar_token"},
                    options=[
                        {
                            "label": question_text(
                                (
                                    "question.environment.network_interface.option_default"
                                    if item == default
                                    else "question.environment.network_interface.option"
                                ),
                                interface=item,
                            ),
                            "value": item,
                        }
                        for item in interfaces
                    ],
                )
            return manual_question(
                group,
                "network_interface",
                question_text("question.environment.network_interface.prompt"),
                field="NETWORK_INTERFACE",
                kind="device",
                structured_config_key="NETWORK_INTERFACE",
                validation={"value_type": "scalar_token"},
            )
        if not confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS"):
            return manual_question(
                group,
                "NETWORK_MAX_BANDWIDTH_GBPS",
                question_text(
                    "question.environment.network_max_bandwidth_gbps.prompt"
                ),
                field="NETWORK_MAX_BANDWIDTH_GBPS",
                structured_config_key="NETWORK_MAX_BANDWIDTH_GBPS",
                validation={"value_type": "positive_number"},
            )
    return None


def question_for_environment_field(
    state: AgentGraphState,
    group: str,
    field: str,
) -> dict[str, Any] | None:
    """Build the exact registered environment question for one edited field."""

    from agent.workflows.group_registry import reconfiguration_question_for_field

    question_id = reconfiguration_question_for_field(field)
    if not question_id or group_for_field(field) != group:
        return None
    projected = deepcopy(state)
    confirmed = dict(projected.get("confirmed_config") or {})
    confirmed.pop(field, None)
    if (
        group == "accounts_disk"
        and field != "has_accounts_device"
        and confirmed.get("has_accounts_device") is not True
    ):
        confirmed.pop("has_accounts_device", None)
    projected.pop(field, None)
    projected["confirmed_config"] = confirmed
    projected.setdefault("inferred_config", {})[f"{field}_manual_required"] = True
    question = question_for_environment(projected, group)
    if question:
        question = dict(question)
        question["reconfiguration_target_field"] = field
        if str(question.get("field") or "") == field:
            question["prompt_ref"] = text_ref_to_dict(
                question_text(
                    "question.environment.reconfiguration_value.prompt",
                    field=field,
                )
            )
        return question
    return None


def apply_environment_answer(state: AgentGraphState, question: dict[str, Any], value: Any) -> HandlerResult:
    if str(question.get("id") or "") == "inferred_config_review":
        return apply_inferred_config_review(state, bool(value))
    next_state: AgentGraphState = deepcopy(state)
    field = str(question.get("field") or "")
    group = str(question.get("group") or "")
    confirmed = next_state.setdefault("confirmed_config", {})
    if value == "__manual__":
        next_state.setdefault("inferred_config", {})[f"{field}_manual_required"] = True
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            clear_pending=True,
            next_group=group,
            completion="in_progress",
        )
    if field in {
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_SIZE",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
    }:
        normalized = normalize_scalar(str(value))
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", normalized) or float(normalized) <= 0:
            return HandlerResult(
                response_fragments=(
                    ResponseFragment(
                        kind="warning",
                        message_id="harness.environment.positive_number_required",
                        source=__name__,
                    ),
                ),
                pending_question=question,
                completion="blocked",
            )
        value = normalized
    if field in {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE"}:
        value = normalize_scalar(str(value)).lower()
    if field == "has_accounts_device":
        confirmed[field] = bool(value)
        if not value:
            for key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
                confirmed.pop(key, None)
    elif field:
        confirmed[field] = value
    if field:
        mark_field_confirmed(next_state, field, group=group)
    if field in {"LEDGER_DEVICE", "ACCOUNTS_DEVICE"}:
        prefix = "DATA" if field == "LEDGER_DEVICE" else "ACCOUNTS"
        for key in (f"{prefix}_VOL_TYPE", f"{prefix}_VOL_SIZE", f"{prefix}_VOL_MAX_IOPS", f"{prefix}_VOL_MAX_THROUGHPUT"):
            confirmed.pop(key, None)
        next_state.setdefault("inferred_config", {}).pop(f"{prefix}_VOL_SIZE_manual_required", None)
    invalidated = set(next_state.get("invalidated_groups") or [])
    invalidated.discard(group)
    next_state["invalidated_groups"] = sorted(invalidated)
    record_group_invalidations(next_state, group)
    invalidated = set(next_state.get("invalidated_groups") or [])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(invalidated - set(state.get("invalidated_groups") or []))),
        reconfigured_groups=(group,),
        clear_pending=True,
        next_group=group,
        completion="in_progress",
    )


def _disk_question(state: AgentGraphState, *, prefix: str, device_key: str, group: str) -> dict[str, Any] | None:
    confirmed = state.get("confirmed_config") or {}
    if not confirmed.get(device_key):
        candidates = _disk_candidates(state)
        if candidates:
            return choice_question(
                group,
                device_key,
                question_text(
                    "question.environment.disk_device.prompt",
                    device_key=device_key,
                ),
                field=device_key,
                kind="device",
                manual_input_allowed=True,
                structured_config_key=device_key,
                validation={"value_type": "scalar_token"},
                options=[
                    {
                        "label": question_text(
                            "question.environment.disk_device.option",
                            name=item["name"],
                            size=str(item.get("size") or "?"),
                            device_type=str(item.get("type") or "?"),
                        ),
                        "value": item["name"],
                    }
                    for item in candidates
                ],
            )
        return manual_question(
            group,
            device_key,
            question_text(
                "question.environment.disk_device_manual.prompt",
                device_key=device_key,
            ),
            field=device_key,
            kind="device",
            structured_config_key=device_key,
            validation={"value_type": "scalar_token"},
        )
    fields = (
        (f"{prefix}_VOL_TYPE", "data_vol_type" if prefix == "DATA" else "accounts_vol_type"),
        (f"{prefix}_VOL_SIZE", "data_vol_size" if prefix == "DATA" else "accounts_vol_size"),
        (f"{prefix}_VOL_MAX_IOPS", "data_vol_max_iops" if prefix == "DATA" else "accounts_vol_max_iops"),
        (f"{prefix}_VOL_MAX_THROUGHPUT", "data_vol_max_throughput" if prefix == "DATA" else "accounts_vol_max_throughput"),
    )
    for env_key, prompt_key in fields:
        if confirmed.get(env_key):
            continue
        if env_key.endswith("_VOL_SIZE"):
            inferred = _disk_size_gib(state, str(confirmed.get(device_key) or ""))
            manual = (state.get("inferred_config") or {}).get(f"{env_key}_manual_required")
            if inferred and not manual:
                return choice_question(
                    group,
                    env_key,
                    question_text(
                        "question.environment.detected_disk_size.prompt",
                        field=env_key,
                        value=inferred,
                    ),
                    field=env_key,
                    kind="yes_no",
                    manual_input_allowed=True,
                    structured_config_key=env_key,
                    validation={"value_type": "positive_number"},
                    options=[
                        {
                            "label": question_text("question.common.option.yes"),
                            "value": inferred,
                        },
                        {
                            "label": question_text("question.common.option.no"),
                            "value": "__manual__",
                            "manual_entry": True,
                            "expected_patch": {f"inferred_config.{env_key}_manual_required": True},
                        },
                    ],
                )
        value_type = "positive_number" if env_key.endswith(("_VOL_SIZE", "_VOL_MAX_IOPS", "_VOL_MAX_THROUGHPUT")) else "scalar_token"
        return manual_question(
            group,
            env_key,
            question_text(f"question.environment.{prompt_key}.prompt"),
            field=env_key,
            structured_config_key=env_key,
            validation={"value_type": value_type},
        )
    return None


def _disk_candidates(state: AgentGraphState) -> list[dict[str, Any]]:
    disks = (state.get("discovery") or {}).get("disks") or {}
    candidates = disks.get("candidates") if isinstance(disks, dict) else []
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict) and str(item.get("name") or "").strip() and str(item.get("size") or "") not in {"0", "0B"}]


def _disk_size_gib(state: AgentGraphState, device: str) -> str:
    for item in _disk_candidates(state):
        if str(item.get("name") or "") != device:
            continue
        match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGTP]?)B?\s*", str(item.get("size") or ""), re.IGNORECASE)
        if not match:
            return ""
        number = float(match.group(1))
        unit = match.group(2).upper()
        multiplier = {"": 1 / (1024**3), "K": 1 / (1024**2), "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024**2}[unit]
        return str(max(1, int(number * multiplier)))
    return ""


def _usable_interfaces(interfaces: list[str], default: str) -> list[str]:
    virtual = {"lo", "bonding_masters", "sit0", "tunl0", "gre0", "gretap0", "erspan0", "ip_vti0", "ip6_vti0", "ip6gre0", "ip6tnl0"}
    values = [normalize_scalar(item) for item in interfaces]
    output = [item for item in values if item and (item == default or item not in virtual)]
    if default and default not in output:
        output.insert(0, default)
    return list(dict.fromkeys(output))
