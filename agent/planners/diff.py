"""Structured plan diff helpers."""

from __future__ import annotations

from typing import Any


INTERESTING_PATHS = [
    ("chain",),
    ("strategy",),
    ("rpc_mode",),
    ("use_fake_node",),
    ("dependency_mode",),
    ("execution", "command"),
    ("execution", "environment"),
    ("advanced_defaults", "qps"),
    ("advanced_defaults", "observability"),
    ("workload", "methods"),
    ("approval_checkpoints",),
    ("requires_confirmation",),
]


def diff_plans(old: dict[str, Any], new: dict[str, Any]) -> dict[str, list[str]]:
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    for path in INTERESTING_PATHS:
        label = ".".join(path)
        old_exists, old_value = _get(old, path)
        new_exists, new_value = _get(new, path)
        if old_exists and not new_exists:
            removed.append(label)
        elif new_exists and not old_exists:
            added.append(label)
        elif old_value != new_value:
            changed.append(f"{label}: {_short(old_value)} -> {_short(new_value)}")

    return {"changed": changed, "added": added, "removed": removed}


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _short(value: Any, limit: int = 140) -> str:
    text = repr(value)
    return text[:limit] + ("..." if len(text) > limit else "")
