"""Version-scoped checkpoint adapters outside the current-turn runtime."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_v12_action_envelope(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the retired v12 arguments envelope for checkpoint migration."""

    action = dict(raw)
    nested = action.pop("arguments", None)
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if key in {"type", "intent", "action_id", "confidence", "reason"}:
                continue
            normalized_key = str(key)
            if normalized_key in action and action[normalized_key] != value:
                raise ValueError(
                    f"conflicting flat and v12 arguments values for {normalized_key}"
                )
            action.setdefault(normalized_key, value)
    if "type" not in action and "intent" in action:
        action["type"] = action["intent"]
    action.pop("intent", None)
    return action


def compile_v12_custom_rpc_action(
    raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile the retired v12 custom-RPC action into current commands."""

    action = normalize_v12_action_envelope(raw)
    if str(action.get("type") or "") != "start_custom_rpc":
        return [action]
    metadata = {
        key: value
        for key, value in action.items()
        if key in {"confidence", "reason"}
    }
    catalog_metadata = dict(metadata)
    if str(action.get("source_evidence") or "").strip():
        catalog_metadata["source_evidence"] = action["source_evidence"]
    commands: list[dict[str, Any]] = []
    if action.get("rpc_endpoint"):
        commands.append({
            "type": "rpc_catalog_command",
            "catalog_command": "set_endpoint",
            "rpc_endpoint": action["rpc_endpoint"],
            **catalog_metadata,
        })
    if action.get("rpc_method"):
        commands.append({
            "type": "rpc_catalog_command",
            "catalog_command": "set_method",
            "rpc_method": action["rpc_method"],
            **catalog_metadata,
        })
    if action.get("rpc_schema_evidence"):
        commands.append({
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "rpc_schema_evidence": action["rpc_schema_evidence"],
            **catalog_metadata,
        })
    if action.get("workload_scope"):
        workload = {
            "type": "rpc_workload_command",
            "workload_scope": action["workload_scope"],
            **metadata,
        }
        if isinstance(action.get("rpc_weights"), Mapping):
            workload["rpc_weights"] = dict(action["rpc_weights"])
        if "finish_methods" in action:
            workload["finish_methods"] = bool(action.get("finish_methods"))
        commands.append(workload)
    if not commands:
        commands.append({
            "type": "rpc_catalog_command",
            "catalog_command": "enter",
            **catalog_metadata,
        })
    return commands
