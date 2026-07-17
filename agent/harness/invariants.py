"""Global state invariants enforced after every Harness transition."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.workflows.group_registry import GROUPS
from .contracts import DOMAIN_CONTROL_ROOTS, StateDelta, StatePath
from .domains.registry import GROUP_OWNER
from .state import AgentGraphState
from .input_values import normalize_target_mode


class StateInvariantError(RuntimeError):
    pass


def _registry_owner_paths() -> dict[str, set[StatePath]]:
    paths: dict[str, set[StatePath]] = {}
    top_level = set(AgentGraphState.__optional_keys__) | set(AgentGraphState.__required_keys__)
    for spec in GROUPS:
        owned = paths.setdefault(spec.owner, set())
        for field in spec.fields:
            owned.add(("confirmed_config", field))
            if field in top_level:
                owned.add((field,))
    return paths


_REGISTRY_OWNER_PATHS = _registry_owner_paths()
_INVALIDATION_PATHS: set[StatePath] = {
    ("preflight",),
    ("plan",),
    ("plan_file",),
    ("smoke",),
    ("final_benchmark",),
    ("job",),
}


OWNER_PATH_POLICY: dict[str, frozenset[StatePath]] = {
    "orientation": frozenset(
        _REGISTRY_OWNER_PATHS.get("orientation", set())
        | {
            ("language",),
            ("checkpoint_recovery",),
            ("resume_context",),
        }
    ),
    "environment": frozenset(
        _REGISTRY_OWNER_PATHS.get("environment", set())
        | {
            ("inferred_config",),
            ("endpoint_evidence", "proposed_values"),
            # Structured review is one typed environment action. These exact
            # fields are shared proposal inputs, not workflow-control grants.
            ("confirmed_config", "CHAIN_REST_URL"),
            ("confirmed_config", "CHAIN_INDEXER_URL"),
            ("confirmed_config", "CHAIN_SIDECAR_URL"),
            ("confirmed_config", "CHAIN_EVM_RPC_URL"),
            ("confirmed_config", "CHAIN_JSON_RPC_URL"),
            ("confirmed_config", "CHAIN_MIRROR_URL"),
            ("confirmed_config", "RPC_API_KEY"),
        }
    ),
    "chain_rpc": frozenset(
        _REGISTRY_OWNER_PATHS.get("chain_rpc", set())
        | {("target_mode_change_candidate",)}
    ),
    "performance": frozenset(
        _REGISTRY_OWNER_PATHS.get("performance", set())
    ),
    "sync_observe": frozenset(
        _REGISTRY_OWNER_PATHS.get("sync_observe", set())
    ),
    "execution": frozenset(
        _REGISTRY_OWNER_PATHS.get("execution", set())
        | _INVALIDATION_PATHS
    ),
    "recovery": frozenset(
        _REGISTRY_OWNER_PATHS.get("recovery", set())
        | {
            ("failure_recovery",),
        }
    ),
    "analysis": frozenset(
        _REGISTRY_OWNER_PATHS.get("analysis", set())
    ),
    "coordinator": frozenset(),
}


def validate_delta_owner(delta: StateDelta, owner: str) -> None:
    """Reject every delta path not assigned to the registry-supplied owner."""

    if owner not in OWNER_PATH_POLICY:
        raise StateInvariantError(f"unknown delta owner: {owner or '<missing>'}")
    allowed = OWNER_PATH_POLICY[owner]
    for path in [*(write.path for write in delta.writes), *delta.deletes]:
        if not path:
            raise StateInvariantError("state delta contains an empty path")
        if path[0] in DOMAIN_CONTROL_ROOTS or path[0].startswith("_"):
            raise StateInvariantError(
                f"domain {owner} cannot mutate coordinator path: {'.'.join(path)}"
            )
        if not any(path[: len(prefix)] == prefix for prefix in allowed):
            raise StateInvariantError(
                f"domain {owner} does not own state path: {'.'.join(path)}"
            )


def apply_state_delta(
    state: AgentGraphState,
    delta: StateDelta,
    *,
    owner: str,
) -> AgentGraphState:
    """Apply one owner-validated delta to an isolated candidate state."""

    validate_delta_owner(delta, owner)
    candidate: AgentGraphState = deepcopy(state)
    for write in delta.writes:
        _write_path(candidate, write.path, deepcopy(write.value))
    for path in delta.deletes:
        _delete_path(candidate, path)
    return candidate


def _write_path(state: dict[str, Any], path: StatePath, value: Any) -> None:
    current: dict[str, Any] = state
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _delete_path(state: dict[str, Any], path: StatePath) -> None:
    current: Any = state
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def validate_state(state: AgentGraphState) -> None:
    active_group = str(state.get("active_group") or "opening")
    if active_group not in GROUP_OWNER:
        raise StateInvariantError(f"unknown active group: {active_group}")

    pending = state.get("pending_question") or {}
    if pending:
        pending_group = str(pending.get("group") or "")
        if pending_group not in GROUP_OWNER:
            raise StateInvariantError(f"pending question has unknown group: {pending_group}")
        if pending_group != active_group:
            raise StateInvariantError(
                f"pending question owner {pending_group} differs from active group {active_group}"
            )
        invalidated = set(state.get("invalidated_groups") or [])
        group_state = (state.get("group_states") or {}).get(pending_group) or {}
        if pending_group in invalidated and str(group_state.get("status") or "").lower() not in {
            "in_progress",
            "reconfiguring",
        }:
            raise StateInvariantError(f"pending question belongs to invalidated group: {pending_group}")
        if str(group_state.get("status") or "").lower() in {"complete", "completed"}:
            raise StateInvariantError(f"pending question belongs to completed group: {pending_group}")
        for capability in pending.get("requires_capabilities") or []:
            if str(capability) == "chain_identity":
                identity = state.get("chain_identity") or {}
                if not str(identity.get("canonical") or identity.get("raw") or "").strip():
                    raise StateInvariantError("pending question requires chain identity")

    queue = state.get("action_queue") or []
    action_ids = [str(item.get("action_id") or "") for item in queue if isinstance(item, dict)]
    non_empty_ids = [item for item in action_ids if item]
    if len(non_empty_ids) != len(set(non_empty_ids)):
        raise StateInvariantError("action queue contains duplicate action ids")

    history = state.get("group_history") or []
    unknown_history = [group for group in history if group not in GROUP_OWNER]
    if unknown_history:
        raise StateInvariantError(f"group history contains unknown groups: {unknown_history}")

    workflow_goals = state.get("workflow_goals") or []
    if not isinstance(workflow_goals, list):
        raise StateInvariantError("workflow goals must be a list")
    goal_keys: set[tuple[str, str]] = set()
    for item in workflow_goals:
        if not isinstance(item, dict):
            raise StateInvariantError("workflow goal must be an object")
        target_mode = normalize_target_mode(item.get("target_mode"))
        goal = str(item.get("goal") or "").strip()
        source = str(item.get("source_evidence") or "").strip()
        if not target_mode or not goal or not source:
            raise StateInvariantError("workflow goal requires canonical target_mode, goal, and source_evidence")
        key = (target_mode, goal)
        if key in goal_keys:
            raise StateInvariantError("workflow goals contain a duplicate target and goal")
        goal_keys.add(key)

    if str(state.get("workflow_mode") or "") == "sync_observe" and any(
        (state.get("rpc_mode"), state.get("workload"), state.get("custom_rpc"), state.get("fixture_evidence"), state.get("qps_profile"))
    ):
        raise StateInvariantError("sync-observe state contains RPC benchmark workload or QPS configuration")


def verify_expected_patch(state: AgentGraphState, expected: dict[str, Any]) -> None:
    """Fail an option execution whose declared postcondition did not occur."""

    for dotted_key, expected_value in expected.items():
        current: Any = state
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                raise StateInvariantError(f"missing option postcondition: {dotted_key}")
            current = current[part]
        if current != expected_value:
            raise StateInvariantError(
                f"option postcondition mismatch for {dotted_key}: expected {expected_value!r}, got {current!r}"
            )
