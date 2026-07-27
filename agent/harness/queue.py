"""Pure durable action ordering for the AnyChain Harness."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent.workflows.group_registry import GROUP_SPEC_BY_NAME

from .action_registry import (
    ACTION_BY_TYPE,
    action_merge_key,
    action_execution_phase,
    action_is_turn_local,
    pending_barrier_semantics,
    state_has_capability,
)
from .state import AgentGraphState


class ActionQueueConflict(ValueError):
    """Raised when one transaction contains mutually exclusive typed mutations."""


def order_action_queue(
    state: AgentGraphState,
    actions: list[dict],
) -> list[dict]:
    """Topologically order typed actions while preserving semantic order."""

    _reject_conflicting_mutations(actions)
    requirements = [action_requirements(state, action) for action in actions]
    providers = [action_provisions(action) for action in actions]
    target_groups = [action_target_groups(action) for action in actions]
    incoming: list[set[int]] = [set() for _ in actions]
    outgoing: list[set[int]] = [set() for _ in actions]
    for consumer_index, required in enumerate(requirements):
        missing = {item for item in required if not state_has_capability(state, item)}
        for provider_index, provided in enumerate(providers):
            if (
                provider_index == consumer_index
                or not (
                    missing & provided
                    or (
                        required & provided
                        and _same_action_transaction(
                            actions[provider_index],
                            actions[consumer_index],
                        )
                    )
                )
            ):
                continue
            _add_dependency(
                provider_index,
                consumer_index,
                incoming=incoming,
                outgoing=outgoing,
            )

    for mutator_index, mutated_groups in enumerate(target_groups):
        if not mutated_groups or not _action_mutates_workflow(actions[mutator_index]):
            continue
        invalidated_groups = {
            invalidated
            for group in mutated_groups
            for invalidated in _group_invalidations(group)
        }
        for consumer_index, consumer_groups in enumerate(target_groups):
            if (
                consumer_index == mutator_index
                or not consumer_groups
                or not _same_action_transaction(
                    actions[mutator_index],
                    actions[consumer_index],
                )
            ):
                continue
            prerequisite_groups = {
                prerequisite
                for group in consumer_groups
                for prerequisite in _transitive_group_dependencies(group)
            }
            if not (
                mutated_groups & prerequisite_groups
                or invalidated_groups & consumer_groups
            ):
                continue
            _add_dependency(
                mutator_index,
                consumer_index,
                incoming=incoming,
                outgoing=outgoing,
            )

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
            _add_dependency(
                proposal_index,
                consumer_index,
                incoming=incoming,
                outgoing=outgoing,
            )

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


def action_target_groups(action: Mapping[str, Any]) -> set[str]:
    """Return registry-owned groups whose state an action can establish or mutate."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    if spec is None:
        return set()
    groups = {
        str(group).strip()
        for group in (
            spec.target_group,
            *spec.compiler_groups,
            *spec.provides_capabilities,
        )
        if str(group).strip() in GROUP_SPEC_BY_NAME
    }
    if spec.target_field_argument:
        from agent.workflows.group_registry import group_for_field

        dynamic_group = group_for_field(
            str(action.get(spec.target_field_argument) or "").strip()
        )
        if dynamic_group:
            groups.add(dynamic_group)
    return groups


def _action_mutates_workflow(action: Mapping[str, Any]) -> bool:
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return bool(
        spec is not None
        and spec.effect in {
            "configuration_mutation",
            "workflow_state_mutation",
        }
    )


def _group_invalidations(group: str) -> tuple[str, ...]:
    spec = GROUP_SPEC_BY_NAME.get(group)
    return spec.invalidates if spec else ()


def _transitive_group_dependencies(group: str) -> set[str]:
    discovered: set[str] = set()
    group_spec = GROUP_SPEC_BY_NAME.get(group)
    pending = list(group_spec.depends_on if group_spec is not None else ())
    while pending:
        dependency = pending.pop()
        if dependency in discovered:
            continue
        discovered.add(dependency)
        dependency_spec = GROUP_SPEC_BY_NAME.get(dependency)
        if dependency_spec is not None:
            pending.extend(dependency_spec.depends_on)
    return discovered


def _same_action_transaction(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_scope = str(left.get("_plan_scope") or "").strip()
    right_scope = str(right.get("_plan_scope") or "").strip()
    if left_scope and right_scope:
        return left_scope == right_scope
    left_turn = left.get("_submitted_turn_index")
    right_turn = right.get("_submitted_turn_index")
    if left_turn is not None and right_turn is not None:
        return int(left_turn) == int(right_turn)
    return True


def _add_dependency(
    provider_index: int,
    consumer_index: int,
    *,
    incoming: list[set[int]],
    outgoing: list[set[int]],
) -> None:
    incoming[consumer_index].add(provider_index)
    outgoing[provider_index].add(consumer_index)


def _reject_conflicting_mutations(actions: list[dict]) -> None:
    decisions: dict[tuple[str, str, str], tuple[str, int]] = {}
    for index, action in enumerate(actions):
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.mutation_dimension:
            continue
        semantic_family = str(action_merge_key(action)[0])
        transaction = _action_transaction_identity(action)
        key = (transaction, spec.mutation_dimension, semantic_family)
        identity = _mutation_decision_identity(action, spec.allowed_arguments)
        prior = decisions.get(key)
        if prior is None:
            decisions[key] = (identity, index)
            continue
        prior_identity, prior_index = prior
        if identity != prior_identity:
            raise ActionQueueConflict(
                "conflicting same-turn mutation requires explicit clarification "
                f"before queue admission: {spec.mutation_dimension} "
                f"(actions {prior_index} and {index})"
            )


def _action_transaction_identity(action: Mapping[str, Any]) -> str:
    scope = str(action.get("_plan_scope") or "").strip()
    if scope:
        return f"scope:{scope}"
    turn = action.get("_submitted_turn_index")
    if turn is not None:
        return f"turn:{int(turn)}"
    return "implicit"


def _mutation_decision_identity(
    action: Mapping[str, Any],
    allowed_arguments: tuple[str, ...],
) -> str:
    excluded = {
        "source_evidence",
        "target_mode_explicit",
        "mutation_explicit",
    }
    payload = {
        key: action.get(key)
        for key in allowed_arguments
        if key not in excluded and key in action
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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
    trusted_runtime_control = bool(
        spec is not None
        and spec.internal_only
    )
    administrative_detour = bool(
        declared_intake_detour
        or trusted_runtime_control
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
    barrier = pending_barrier_semantics(pending)
    if barrier["queue_barrier"]:
        barrier_policy = str(barrier["policy"])
        pending_created_this_turn = bool(
            int(pending.get("created_turn_index") or 0)
            == int(state.get("turn_index") or 0)
        )
        if barrier_policy == "exclusive_owner":
            if pending_created_this_turn:
                return bool(
                    action_is_turn_local(action)
                    or trusted_runtime_control
                )
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
                or trusted_runtime_control
            )
        if pending_created_this_turn:
            return bool(
                action_is_turn_local(action)
                or explicit_navigation
                or user_grounded_detour
                or trusted_runtime_control
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
