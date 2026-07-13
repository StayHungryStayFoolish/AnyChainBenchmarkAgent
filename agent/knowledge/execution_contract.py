"""Static benchmark execution contract description, shared by the CLI tool-call surface."""

from __future__ import annotations

from typing import Any

try:
    from .entry_contract import (
        ENTRYPOINT_PHASES,
        OPTIONAL_ACCOUNTS_FIELDS,
        REAL_NODE_ENDPOINT_FIELDS,
        RUNTIME_BASELINE_FIELDS,
        dependency_names,
        required_keys_for_target,
    )
except ImportError:  # script execution with agent/ on sys.path
    from knowledge.entry_contract import (
        ENTRYPOINT_PHASES,
        OPTIONAL_ACCOUNTS_FIELDS,
        REAL_NODE_ENDPOINT_FIELDS,
        RUNTIME_BASELINE_FIELDS,
        dependency_names,
        required_keys_for_target,
    )


def load_execution_contract(use_fake_node: bool | None = None) -> dict[str, Any]:
    """Describe the benchmark execution contract: phases, required variables, gates.

    Use this before planning or explaining a benchmark workflow. It describes
    entrypoint phases, required runtime variables, optional accounts-disk
    variables, real-node endpoint requirements, and dependency expectations.
    """
    target_required = {
        "unknown_target": list(required_keys_for_target(None)),
        "fake_node": list(required_keys_for_target(True)),
        "real_node": list(required_keys_for_target(False)),
    }
    if use_fake_node is True:
        selected_required = target_required["fake_node"]
        deps = list(dependency_names(True))
    elif use_fake_node is False:
        selected_required = target_required["real_node"]
        deps = list(dependency_names(False))
    else:
        selected_required = target_required["unknown_target"]
        deps = []
    return {
        "entrypoint": "./blockchain_node_benchmark.sh",
        "phases": list(ENTRYPOINT_PHASES),
        "runtime_baseline_fields": [_field_payload(field) for field in RUNTIME_BASELINE_FIELDS],
        "real_node_endpoint_fields": [_field_payload(field) for field in REAL_NODE_ENDPOINT_FIELDS],
        "optional_accounts_fields": [_field_payload(field) for field in OPTIONAL_ACCOUNTS_FIELDS],
        "required_keys": target_required,
        "selected_required_keys": selected_required,
        "expected_dependencies": deps,
        "mandatory_gates": ["doctor", "configuration_checklist", "preflight", "smoke", "explicit_user_approval"],
    }


def _field_payload(field: Any) -> dict[str, Any]:
    return {
        "key": field.key,
        "env": field.env,
        "label": field.label,
        "reason": field.reason,
        "value_kind": field.value_kind,
        "required": field.required,
        "inferred": field.inferred,
        "optional_when": field.optional_when,
    }
