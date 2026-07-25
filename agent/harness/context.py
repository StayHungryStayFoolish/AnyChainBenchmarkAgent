"""Build model context from typed product registries and current state."""

from __future__ import annotations

from typing import Any

from .action_registry import (
    ACTION_ARGUMENT_SCHEMAS,
    ACTION_SPECS,
    CONSULTATION_TOPIC_PURPOSES,
    CONSULTATION_TOPICS,
    ActionLifetime,
    ActionSpec,
    project_action_specs,
)
from .state import AgentGraphState

from agent.workflows.group_registry import GROUPS, USER_NAVIGABLE_GROUPS


def action_schema(
    *,
    owners: frozenset[str] | None = None,
    groups: frozenset[str] | None = None,
    lifetimes: frozenset[ActionLifetime] | None = None,
    action_types: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    specs = project_action_specs(
        owners=owners,
        groups=groups,
        lifetimes=lifetimes,
        action_types=action_types,
    )
    output = [
        _action_spec_schema(spec)
        for spec in specs
    ]
    for item in output:
        if item["type"] == "answer_opening_question":
            item["allowed_topics"] = list(CONSULTATION_TOPICS)
            item["topic_purposes"] = dict(CONSULTATION_TOPIC_PURPOSES)
        if item["type"] == "change_group":
            item["allowed_groups"] = list(USER_NAVIGABLE_GROUPS)
        if item["target_field_argument"]:
            item["allowed_config_fields"] = sorted({
                field
                for group in GROUPS
                for field, _question in group.reconfiguration_questions
            })
    return output


def _action_spec_schema(spec: ActionSpec) -> dict[str, Any]:
    return {
        "type": spec.action_type,
        "owner": spec.owner,
        "purpose": spec.purpose,
        "allowed_arguments": list(spec.allowed_arguments),
        "required_arguments": list(spec.required_arguments),
        "argument_schemas": {
            name: dict(ACTION_ARGUMENT_SCHEMAS[name])
            for name in spec.allowed_arguments
        },
        "constraints": list(spec.constraints),
        "suppressed_by": list(spec.suppressed_by),
        "execution_phase": spec.execution_phase,
        "effect": spec.effect,
        "target_group": spec.target_group,
        "target_field_argument": spec.target_field_argument,
        "compiler_groups": list(spec.compiler_groups),
        "semantic_operations": list(spec.semantic_operations),
        "requires_specific_change": spec.requires_specific_change,
        "incomplete_mutation_intake": spec.incomplete_mutation_intake,
        "entry_intake": spec.entry_intake,
        "entry_intake_purpose": spec.entry_intake_purpose,
        "entry_intake_fixed_arguments": dict(
            spec.entry_intake_fixed_arguments
        ),
        "entry_intake_value_arguments": list(
            spec.entry_intake_value_arguments
        ),
        "incompatible_target_modes": list(spec.incompatible_target_modes),
        "required_state_path": list(spec.required_state_path),
        "required_state_values": list(spec.required_state_values),
        "state_transition_path": list(spec.state_transition_path),
        "state_transition_value": spec.state_transition_value,
        "exact_source_value_arguments": list(spec.exact_source_value_arguments),
    }


def group_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": group.name,
            "owner": group.owner,
            "fields": list(group.fields),
            "questions": list(group.questions),
            "reconfiguration_questions": {
                field: question for field, question in group.reconfiguration_questions
            },
            "immutable_fields": list(group.immutable_fields),
            "depends_on": list(group.depends_on),
            "invalidates": list(group.invalidates),
            "category": group.category,
            "workflow_modes": list(group.workflow_modes),
            "target_modes": list(group.target_modes),
            "navigation_entry": group.navigation_entry,
            "generic_navigation": group.generic_navigation,
            "entry_actions": [
                {
                    "type": spec.action_type,
                    "purpose": spec.entry_intake_purpose,
                    "fixed_arguments": dict(
                        spec.entry_intake_fixed_arguments
                    ),
                    "value_arguments": list(
                        spec.entry_intake_value_arguments
                    ),
                }
                for spec in ACTION_SPECS
                if spec.target_group == group.name and spec.entry_intake
            ],
        }
        for group in GROUPS
    ]


def workflow_snapshot(state: AgentGraphState) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    return {
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "opening",
        "pending_question": state.get("pending_question") or {},
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "chain_identity": {
            "raw": identity.get("raw") or "",
            "canonical": identity.get("canonical") or "",
            "status": identity.get("status") or "",
            "adapter_family": identity.get("adapter_family") or "",
        },
        "rpc_mode": state.get("rpc_mode") or "",
        "custom_rpc": state.get("custom_rpc") or {},
        "endpoint_evidence": state.get("endpoint_evidence") or {},
        "secondary_handoff": state.get("secondary_handoff") or {},
        "qps_profile": state.get("qps_profile") or {},
        "observability": state.get("observability") or {},
        "sync_observe": state.get("sync_observe") or {},
        "confirmed_config": state.get("confirmed_config") or {},
        "inferred_config": state.get("inferred_config") or {},
        "evidence_collection": _planner_evidence_collection(state.get("evidence_collection") or {}),
        "invalidated_groups": state.get("invalidated_groups") or [],
        "interruption_stack": state.get("interruption_stack") or [],
        "action_queue": state.get("action_queue") or [],
        "workflow_goals": state.get("workflow_goals") or [],
        "group_states": state.get("group_states") or {},
        "current_job": state.get("job") or {},
        "report_context": state.get("report_context") or {},
        "framework_summary": _planner_framework_summary(state.get("framework_summary") or {}),
        "web_research": _planner_web_research(state.get("web_research") or {}),
    }


_OWNER_STATE_ROOTS: dict[str, tuple[str, ...]] = {
    "analysis": ("evidence_collection", "current_job", "report_context"),
    "chain_rpc": (
        "target_mode",
        "workflow_mode",
        "chain_identity",
        "rpc_mode",
        "custom_rpc",
        "endpoint_evidence",
        "secondary_handoff",
        "framework_summary",
        "web_research",
    ),
    "coordinator": (
        "interruption_stack",
        "action_queue",
        "workflow_goals",
        "group_states",
        "invalidated_groups",
    ),
    "environment": ("confirmed_config", "inferred_config"),
    "execution": ("current_job",),
    "orientation": ("current_job", "framework_summary"),
    "performance": ("qps_profile", "observability"),
    "recovery": ("current_job",),
    "sync_observe": ("sync_observe",),
}


def owner_workflow_snapshot(
    state: AgentGraphState,
    owner: str,
    *,
    groups: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Project only state that one owner may need while compiling commands."""

    snapshot = workflow_snapshot(state)
    selected_groups = groups or frozenset(
        group.name for group in GROUPS if group.owner == owner
    )
    output: dict[str, Any] = {
        "language": snapshot["language"],
        "active_group": snapshot["active_group"],
        "pending_question": snapshot["pending_question"],
        "selected_groups": sorted(selected_groups),
        "selected_group_states": {
            name: value
            for name, value in snapshot["group_states"].items()
            if name in selected_groups
        },
    }
    for key in _OWNER_STATE_ROOTS.get(owner, ()):
        if key in snapshot:
            output[key] = snapshot[key]
    return output


def _planner_evidence_collection(collection: dict[str, Any]) -> dict[str, Any]:
    if not collection:
        return {}
    question = collection.get("question") if isinstance(collection.get("question"), dict) else {}
    lines = [str(item) for item in collection.get("lines") or []]
    return {
        "active": str(collection.get("status") or "active") == "active",
        "status": str(collection.get("status") or "active"),
        "question_id": str(question.get("id") or ""),
        "question_group": str(question.get("group") or ""),
        "line_count": len(lines),
        "collected_text": "\n".join(lines),
    }


def _planner_framework_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Project framework inventory into control-plane classification facts."""

    families = summary.get("families") or {}
    if isinstance(families, dict):
        family_names = sorted(str(name) for name in families if str(name).strip())
    elif isinstance(families, list):
        family_names = sorted(str(name) for name in families if str(name).strip())
    else:
        family_names = []
    return {
        "chain_count": summary.get("chain_count"),
        "family_count": summary.get("family_count"),
        "unique_rpc_method_count": summary.get("unique_rpc_method_count"),
        "adapter_families": family_names,
    }


def _planner_web_research(research: dict[str, Any]) -> dict[str, Any]:
    """Expose availability to routing without leaking search evidence payloads."""

    return {
        "status": research.get("status") or "",
        "provider": research.get("provider") or "",
        "available": bool(research.get("available")),
    }
