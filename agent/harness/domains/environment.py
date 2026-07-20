"""Environment domain: cloud metadata, disks, and network configuration."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..localization import localized
from ..questions import choice_question, manual_question, normalize_scalar
from ..state import AgentGraphState
from ..transitions import record_group_invalidations

from agent.planners import question_prompts
from agent.workflows.group_registry import group_for_field, invalidation_targets
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
PROPOSED_ENDPOINT_FIELDS = {"LOCAL_RPC_URL", "MAINNET_RPC_URL"}
SPECIAL_CONFIG_FIELDS = {
    "HAS_ACCOUNTS_DEVICE",
    "SYNC_OBSERVE_STOP_CONDITION",
    "SYNC_OBSERVE_DURATION_SECONDS",
}
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
    allowed = CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
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
    allowed = CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
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
    if "HAS_ACCOUNTS_DEVICE" not in config_values and _text_mentions_no_accounts_disk(
        action.get("source_text") or action.get("_origin_text") or ""
    ):
        config_values["HAS_ACCOUNTS_DEVICE"] = False
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
            return HandlerResult(blocker="set_accounts_presence requires has_accounts_device")
        next_state: AgentGraphState = deepcopy(state)
        confirmed = next_state.setdefault("confirmed_config", {})
        has_accounts = bool(action.arguments.get("has_accounts_device"))
        confirmed["has_accounts_device"] = has_accounts
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
        # Unknown keys are reviewable only alongside at least one recognized
        # configuration value. Otherwise log lines, HTTP headers, and protocol
        # examples such as ``RuntimeError: ...`` would become configuration
        # transactions and compete with their actual analysis owner.
        if not proposal.get("config_values"):
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                completion="completed",
            )
        return propose_config_assignments_for_review(
            state,
            proposal,
            consumed_action_ids=(action.action_id,),
        )
    return HandlerResult(blocker=f"unsupported environment action: {action.action_type}")


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
    flattened.update(_extract_json_flat_values(text))
    flattened.update(_extract_yaml_flat_values(text))
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        for key, value in iter_config_assignments(line):
            flattened[key] = value
    if not flattened:
        return None

    config_values, workflow_values, unmapped_values = classify_flat_input_values(flattened)
    if not config_values and not workflow_values and not unmapped_values:
        return None
    return {
        "config_values": config_values,
        "workflow_values": workflow_values,
        "unmapped_values": unmapped_values,
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

    return choice_question(
        group,
        "inferred_config_review",
        format_config_proposal_prompt(proposal, language=language),
        field="inferred_config_review",
        kind="yes_no",
        manual_input_allowed=False,
        options=[
            {"label": "Y", "value": True, "expected_patch": {"inferred_config.pending_review": {}}},
            {"label": "N", "value": False, "expected_patch": {"inferred_config.pending_review": {}}},
        ],
        queue_barrier=True,
    )


def format_config_proposal_prompt(proposal: dict[str, Any], *, language: str = "en") -> str:
    """Format review content without performing final terminal rendering."""

    config_values = proposal.get("config_values") if isinstance(proposal.get("config_values"), dict) else {}
    unmapped = proposal.get("unmapped_values") if isinstance(proposal.get("unmapped_values"), dict) else {}
    conflicts = _normalize_conflicts(proposal.get("conflicts"))
    lines = []
    if config_values:
        lines.append(localized(language, "我从你粘贴的内容中推断出这些配置候选值：", "I inferred these candidate config values from your pasted content:"))
        for key in sorted(config_values):
            lines.append(f"- {key}: `{config_values[key]}`")
    if unmapped:
        lines.append(localized(language, "以下内容没有自动映射到已知配置项，不会自动写入：", "These extracted values did not map to known config fields and will not be applied automatically:"))
        for key in sorted(unmapped):
            lines.append(f"- {key}: `{unmapped[key]}`")
    if conflicts:
        lines.append(localized(language, "以下输入存在冲突，请在确认前核对：", "The following inputs conflict; review them before confirming:"))
        lines.extend(f"- {item}" for item in conflicts)
    if proposal.get("reason"):
        lines.append(localized(language, f"推断依据：{proposal.get('reason')}", f"Reason: {proposal.get('reason')}"))
    lines.append(localized(language, "是否确认应用这些已映射的候选值？", "Apply the mapped candidate values?"))
    lines.append(localized(language, "endpoint 类值只会保存为待验证候选，后续仍必须通过 endpoint probe。", "Endpoint values are saved only as candidates; they must still pass endpoint probing later."))
    return "\n".join(lines)


def apply_direct_config_assignments(state: AgentGraphState, config_values: dict[str, Any]) -> HandlerResult:
    """Apply explicit ``KEY=value`` facts and return structured outcome evidence."""

    next_state: AgentGraphState = deepcopy(state)
    outcome = _apply_config_values(next_state, config_values)
    previous_invalidated = set(state.get("invalidated_groups") or [])
    next_invalidated = set(next_state.get("invalidated_groups") or [])
    followups: tuple[dict[str, Any], ...] = ()
    if outcome["sync_options"]:
        followups = ({"type": "set_sync_observe_options", **outcome["sync_options"]},)
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
            visible_result=localized(
                next_state.get("language", "en"),
                "已丢弃这次推断的候选配置。你可以重新粘贴更完整的信息，或直接说明要修改哪个配置组。",
                "Discarded these inferred candidate values. Paste corrected information, or name the configuration group to change.",
            ),
            evidence=({
                "type": "inferred_config_review",
                "accepted": False,
                "unmapped_values": dict(reviewed.get("unmapped_values") or {}),
                "conflicts": _normalize_conflicts(reviewed.get("conflicts")),
            },),
            completion="unchanged",
        )

    outcome = _apply_config_values(next_state, dict(reviewed.get("config_values") or {}))
    accepted_review = {
        "applied": outcome["applied"],
        "endpoint_proposals": outcome["endpoint_proposals"],
        "unmapped_values": dict(reviewed.get("unmapped_values") or {}),
        "source_format": reviewed.get("source_format") or "",
    }
    conflicts = _normalize_conflicts(reviewed.get("conflicts"))
    if conflicts:
        accepted_review["conflicts"] = conflicts
    inferred.setdefault("accepted_reviews", []).append(accepted_review)
    messages: list[str] = []
    if outcome["applied"]:
        details = ", ".join(f"{key}={value}" for key, value in sorted(outcome["applied"].items()))
        messages.append(localized(
            next_state.get("language", "en"),
            f"已应用确认的配置候选值：{details}",
            f"Applied confirmed candidate values: {details}",
        ))
    if outcome["endpoint_proposals"]:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(outcome["endpoint_proposals"].items())
        )
        messages.append(localized(
            next_state.get("language", "en"),
            f"已保存 endpoint 候选值，后续仍会进行真实性验证：{details}",
            f"Saved endpoint candidate values for mandatory validation later: {details}",
        ))
    if outcome["invalid"]:
        details = ", ".join(f"{key}={value}" for key, value in sorted(outcome["invalid"].items()))
        messages.append(localized(
            next_state.get("language", "en"),
            f"以下候选值无效，未写入：{details}",
            f"These candidate values were invalid and were not applied: {details}",
        ))
    if accepted_review["unmapped_values"]:
        messages.append(localized(
            next_state.get("language", "en"),
            "未映射字段已作为证据保留，不会自动写入配置。",
            "Unmapped values were retained as evidence and were not applied automatically.",
        ))
    followups: tuple[dict[str, Any], ...] = ()
    if outcome["sync_options"]:
        followups = ({"type": "set_sync_observe_options", **outcome["sync_options"]},)
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(set(next_state.get("invalidated_groups") or []) - set(state.get("invalidated_groups") or []))),
        reconfigured_groups=tuple(sorted(set(state.get("invalidated_groups") or []) - set(next_state.get("invalidated_groups") or []))),
        clear_pending=True,
        visible_result="\n".join(messages),
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
    allowed = CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
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
    scalar = _strip_scalar(str(value)).strip().strip("\"'")
    if not normalized_key or not scalar:
        return normalized_key, ""
    key_aliases = {
        "SYNC_OBSERVE_DURATION": "SYNC_OBSERVE_DURATION_SECONDS",
        "SYNC_OBSERVE_DURATION_SEC": "SYNC_OBSERVE_DURATION_SECONDS",
        "SYNC_OBSERVE_STOP_CONDITIONS": "SYNC_OBSERVE_STOP_CONDITION",
    }
    normalized_key = key_aliases.get(normalized_key, normalized_key)
    if normalized_key == "ACCOUNTS_DEVICE" and _is_absence_value(scalar):
        return "HAS_ACCOUNTS_DEVICE", False
    if normalized_key == "HAS_ACCOUNTS_DEVICE":
        if _is_absence_value(scalar):
            return normalized_key, False
        if scalar.lower() in {"y", "yes", "true", "1", "有", "是"}:
            return normalized_key, True
        if scalar.lower() in {"n", "no", "false", "0", "没有", "无"}:
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
    endpoint_proposals = state.setdefault("endpoint_evidence", {}).setdefault("proposed_values", {})
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
        elif key in PROPOSED_ENDPOINT_FIELDS:
            endpoint_proposals[key] = scalar
            endpoint_saved[key] = scalar
        elif key == "HAS_ACCOUNTS_DEVICE":
            has_accounts = bool(scalar)
            confirmed["has_accounts_device"] = has_accounts
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


def _is_absence_value(value: Any) -> bool:
    text = _strip_scalar(str(value or "")).strip().lower()
    return text in {
        "none", "<none>", "null", "nil", "n/a", "na", "false", "0", "no", "n",
        "none detected", "not detected", "no separate disk", "no accounts", "no accounts disk",
        "without accounts", "没有", "无", "没有 accounts", "没有 accounts 盘", "没有独立 accounts 盘", "不需要",
    }


def _text_mentions_no_accounts_disk(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or "accounts" not in text and "account" not in text:
        return False
    return any(marker in text for marker in ("没有", "无", "no ", "without", "not have", "don't have", "does not have"))


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
    language = str(state.get("language") or "en")
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
                    localized(language, f"检测到 {env_key} 为 `{detected}`，是否使用？", f"Detected {env_key}: `{detected}`. Use it?"),
                    field=env_key,
                    kind="yes_no",
                    manual_input_allowed=True,
                    validation={"value_type": "scalar_token"},
                    options=[
                        {"label": "Y", "value": detected},
                        {"label": "N", "value": "__manual__", "expected_patch": {f"inferred_config.{env_key}_manual_required": True}},
                    ],
                )
            return manual_question(
                group,
                env_key,
                question_prompts.text_for(prompt_key, language=language),
                field=env_key,
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
                localized(language, "这个节点是否有独立的 accounts/state 磁盘？", "Does this node have a separate accounts/state disk?"),
                field="has_accounts_device",
                kind="yes_no",
                options=[{"label": "Y", "value": True}, {"label": "N", "value": False}],
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
                    question_prompts.text_for("network_interface", language=language),
                    field="NETWORK_INTERFACE",
                    kind="device",
                    manual_input_allowed=True,
                    validation={"value_type": "scalar_token"},
                    options=[{"label": f"{item}{' (default)' if item == default else ''}", "value": item} for item in interfaces],
                )
            return manual_question(
                group,
                "network_interface",
                question_prompts.text_for("network_interface", language=language),
                field="NETWORK_INTERFACE",
                kind="device",
                validation={"value_type": "scalar_token"},
            )
        if not confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS"):
            return manual_question(
                group,
                "NETWORK_MAX_BANDWIDTH_GBPS",
                question_prompts.text_for("network_max_bandwidth_gbps", language=language),
                field="NETWORK_MAX_BANDWIDTH_GBPS",
                validation={"value_type": "positive_number"},
            )
    return None


def apply_environment_answer(state: AgentGraphState, question: dict[str, Any], value: Any) -> HandlerResult:
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
                visible_result=localized(state.get("language", "en"), "输入无效：请输入大于 0 的数值。", "Invalid input: enter a number greater than zero."),
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
    language = str(state.get("language") or "en")
    confirmed = state.get("confirmed_config") or {}
    if not confirmed.get(device_key):
        candidates = _disk_candidates(state)
        if candidates:
            return choice_question(
                group,
                device_key,
                localized(language, f"请选择 {device_key}，或直接输入设备名。", f"Choose {device_key}, or type the device name."),
                field=device_key,
                kind="device",
                manual_input_allowed=True,
                validation={"value_type": "scalar_token"},
                options=[{"label": f"{item['name']} ({item.get('size') or '?'}, {item.get('type') or '?'})", "value": item["name"]} for item in candidates],
            )
        return manual_question(
            group,
            device_key,
            localized(language, f"请输入 {device_key}。", f"Enter {device_key}."),
            field=device_key,
            kind="device",
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
                    localized(language, f"检测到 {env_key} 为 `{inferred}` GiB，是否使用？", f"Detected {env_key}: `{inferred}` GiB. Use it?"),
                    field=env_key,
                    kind="yes_no",
                    manual_input_allowed=True,
                    validation={"value_type": "positive_number"},
                    options=[
                        {"label": "Y", "value": inferred},
                        {"label": "N", "value": "__manual__", "expected_patch": {f"inferred_config.{env_key}_manual_required": True}},
                    ],
                )
        value_type = "positive_number" if env_key.endswith(("_VOL_SIZE", "_VOL_MAX_IOPS", "_VOL_MAX_THROUGHPUT")) else "scalar_token"
        return manual_question(
            group,
            env_key,
            question_prompts.text_for(prompt_key, language=language),
            field=env_key,
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
