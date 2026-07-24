"""Pure durable action ordering for the AnyChain Harness."""

from __future__ import annotations

from .action_registry import (
    ACTION_BY_TYPE,
    action_execution_phase,
    action_is_turn_local,
)
from .input_values import normalize_target_mode
from .state import AgentGraphState


def order_action_queue(
    state: AgentGraphState,
    actions: list[dict],
) -> list[dict]:
    """Topologically order typed actions while preserving semantic order."""

    requirements = [action_requirements(state, action) for action in actions]
    providers = [action_provisions(action) for action in actions]
    incoming: list[set[int]] = [set() for _ in actions]
    outgoing: list[set[int]] = [set() for _ in actions]
    for consumer_index, required in enumerate(requirements):
        missing = {item for item in required if not state_has_capability(state, item)}
        if not missing:
            continue
        for provider_index, provided in enumerate(providers):
            if provider_index == consumer_index or not (missing & provided):
                continue
            incoming[consumer_index].add(provider_index)
            outgoing[provider_index].add(consumer_index)

    proposal_indexes = [
        index
        for index, action in enumerate(actions)
        if str(action.get("type") or "") == "propose_config_values"
    ]
    proposal_blocked_types = {
        "rpc_catalog_command",
        "rpc_workload_command",
        "set_rpc_mode",
        "use_default_workload",
        "configure_workload_weights",
        "set_qps_mode",
        "request_qps_customization",
        "set_qps_override",
        "set_observability",
        "set_sync_observe_source",
        "clear_sync_observe_source",
        "set_sync_observe_options",
        "approve_preflight_smoke",
        "approve_final_benchmark",
    }
    for proposal_index in proposal_indexes:
        for consumer_index, action in enumerate(actions):
            if consumer_index == proposal_index:
                continue
            proposal_scope = str(actions[proposal_index].get("_plan_scope") or "")
            consumer_scope = str(action.get("_plan_scope") or "")
            if not proposal_scope or proposal_scope != consumer_scope:
                continue
            if str(action.get("type") or "") not in proposal_blocked_types:
                continue
            incoming[consumer_index].add(proposal_index)
            outgoing[proposal_index].add(consumer_index)

    ready = [index for index, dependencies in enumerate(incoming) if not dependencies]
    ordered: list[dict] = []
    while ready:
        ready.sort(key=lambda index: action_plan_order(actions[index], index))
        current = ready.pop(0)
        ordered.append(actions[current])
        for consumer in sorted(outgoing[current]):
            incoming[consumer].discard(current)
            if not incoming[consumer] and consumer not in ready:
                ready.append(consumer)
    if len(ordered) != len(actions):
        raise RuntimeError("Harness action dependency cycle")
    return ordered


def action_plan_order(
    action: dict,
    queue_index: int,
) -> tuple[int, int, int, int]:
    submitted_turn = int(action.get("_submitted_turn_index") or 0)
    plan_index = action.get("_plan_index")
    if isinstance(plan_index, int) and plan_index >= 0:
        return (-submitted_turn, 0, plan_index, queue_index)
    return (
        -submitted_turn,
        1,
        action_execution_phase(action),
        queue_index,
    )


def action_requirements(state: AgentGraphState, action: dict) -> set[str]:
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    required = set(spec.requires_capabilities if spec else ())
    if str(action.get("type") or "") == "answer_pending":
        required.update(
            str(item).strip()
            for item in (state.get("pending_question") or {}).get("requires_capabilities") or []
            if str(item).strip()
        )
    return required


def action_provisions(action: dict) -> set[str]:
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return set(spec.provides_capabilities if spec else ())


def state_has_capability(state: AgentGraphState, capability: str) -> bool:
    if capability == "chain_identity":
        identity = state.get("chain_identity") or {}
        return bool(str(identity.get("canonical") or identity.get("raw") or "").strip())
    if capability == "target_mode":
        return bool(normalize_target_mode(state.get("target_mode")))
    return False


def action_can_run_while_pending(
    state: AgentGraphState,
    action: dict,
) -> bool:
    """Apply the registry-backed pending barrier policy during selection."""

    if action_is_turn_local(action):
        return True
    pending = dict(state.get("pending_question") or {})
    action_type = str(action.get("type") or "").strip()
    accepted = {
        str(item).strip()
        for item in pending.get("accepted_action_types") or []
        if str(item).strip()
    }
    if action_type == "answer_pending" or action_type in accepted:
        return True
    spec = ACTION_BY_TYPE.get(action_type)
    source_evidence = str(action.get("source_evidence") or "").strip()
    origin_text = str(action.get("_origin_text") or "").strip()
    source_is_grounded = bool(
        source_evidence
        and origin_text
        and source_evidence.casefold() in origin_text.casefold()
    )
    user_grounded_detour = bool(
        spec is not None
        and spec.preserve_pending
        and source_is_grounded
    )
    declared_intake_detour = bool(
        spec is not None
        and spec.crosses_pending_barrier
        and (spec.incomplete_mutation_intake or spec.entry_intake)
        and source_is_grounded
    )
    administrative_detour = bool(
        declared_intake_detour
        or (spec is not None and spec.internal_only)
        or action_type in {"reset_session", "go_back"}
    )
    explicit_administrative_detour = action_type in {
        "reset_session",
        "go_back",
    }
    explicit_navigation = bool(
        spec is not None
        and spec.effect == "workflow_navigation"
        and spec.crosses_pending_barrier
        and action.get("navigation_explicit") is True
        and source_is_grounded
    )
    if pending.get("queue_barrier"):
        barrier_policy = str(pending.get("barrier_policy") or "")
        pending_created_this_turn = bool(
            int(pending.get("created_turn_index") or 0)
            == int(state.get("turn_index") or 0)
        )
        if barrier_policy == "exclusive_owner":
            if pending_created_this_turn:
                return action_is_turn_local(action)
            return bool(
                action_is_turn_local(action)
                or explicit_navigation
                or administrative_detour
            )
        if barrier_policy == "explicit_detour_only":
            return bool(
                action_is_turn_local(action)
                or explicit_navigation
                or explicit_administrative_detour
            )
        if pending_created_this_turn:
            return bool(
                action_is_turn_local(action)
                or explicit_navigation
                or user_grounded_detour
            )
        return (
            action_is_turn_local(action)
            or administrative_detour
            or explicit_navigation
        )
    pending_created_this_turn = bool(
        pending
        and int(pending.get("created_turn_index") or 0)
        == int(state.get("turn_index") or 0)
    )
    if pending_created_this_turn:
        return bool(
            pending.get("same_turn_navigation_allowed") is True
            and (user_grounded_detour or administrative_detour)
        )
    return (
        action_is_turn_local(action)
        or user_grounded_detour
        or administrative_detour
    )
