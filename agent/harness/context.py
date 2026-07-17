"""Build model context from typed product registries and current state."""

from __future__ import annotations

from typing import Any

from .action_registry import ACTION_ARGUMENT_SCHEMAS, ACTION_SPECS, CONSULTATION_TOPICS
from .state import AgentGraphState

from agent.workflows.group_registry import GROUPS


def action_schema() -> list[dict[str, Any]]:
    output = [
        {
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
            "target_group": spec.target_group,
            "requires_specific_change": spec.requires_specific_change,
        }
        for spec in ACTION_SPECS
    ]
    for item in output:
        if item["type"] == "answer_opening_question":
            item["allowed_topics"] = list(CONSULTATION_TOPICS)
        if item["type"] == "change_group":
            item["allowed_groups"] = [group.name for group in GROUPS]
    return output


def group_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": group.name,
            "owner": group.owner,
            "fields": list(group.fields),
            "questions": list(group.questions),
            "depends_on": list(group.depends_on),
            "invalidates": list(group.invalidates),
            "category": group.category,
        }
        for group in GROUPS
    ]


def workflow_snapshot(state: AgentGraphState) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    return {
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "opening",
        "active_subgroup": state.get("active_subgroup") or "",
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
        "invalidated_groups": state.get("invalidated_groups") or [],
        "interruption_stack": state.get("interruption_stack") or [],
        "action_queue": state.get("action_queue") or [],
        "workflow_goals": state.get("workflow_goals") or [],
        "group_states": state.get("group_states") or {},
        "current_job": state.get("job") or {},
        "framework_summary": _planner_framework_summary(state.get("framework_summary") or {}),
        "web_research": _planner_web_research(state.get("web_research") or {}),
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


def build_action_resolver_prompt() -> str:
    return (
        "You are the typed intent planner for AnyChain Benchmark Agent. Return one JSON object only; never answer the user. "
        "Read the supplied action_schema, group_schema, workflow_state, pending question, and full user turn. "
        "input_shape describes terminal transport only. Multiline input is not automatically logs or evidence: classify JSON, YAML, env, curl, reports, stack traces, and mixed prose by the user's semantic request, and preserve every independent action. "
        "Propose the smallest ordered action list that preserves every distinct user request. "
        "The request contains authoritative structural clauses produced without intent classification. For every prose clause, partition its complete text into semantically independent units only when exact contiguous source_text anchors cover every word. If conjunctions or framing make a lossless split uncertain, use one full-clause semantic unit mapped to every typed action that preserves it; do not omit connective prose or reject an otherwise representable request merely to force a split. Structured clauses remain one atomic unit. Every unit must contain an ordered exact source_text anchor copied from its parent clause. Do not calculate character offsets; the Harness derives them. Anchors may omit only punctuation or whitespace between units and must not omit prose. Introductory, framing, or trailing prose around a structured block is still a semantic unit: map it to the action that consumes the related block, or mark it unresolved when that relationship is genuinely unclear. Then classify every unit independently as one of: answer the active pending contract, read-only consultation, concrete mutation, explicit navigation, evidence/report analysis, or unresolved. A pending question never takes precedence over another explicit unit. "
        "Questions and consultations must not become configuration changes. Configuration values are proposals until confirmed. "
        "When the user explicitly asks for subsequent responses in Chinese or English, emit set_response_language with language=zh or en and exact source_evidence. This action may coexist with every other request in the same turn; never mark the language-preference unit unresolved merely because terminal presentation already uses that language. "
        "An explicit consultation-only, not-starting-yet, or no-configuration-change unit is a scope constraint, not unresolved. Set scope_constraint='consultation_only' on that semantic unit and map it to every read-only consultation action it scopes. Do not emit a durable mutation in that scope. If the same scope also explicitly requests a mutation, preserve the conflict as unresolved instead of partially committing either interpretation. "
        "A pending question accepts exact local answers outside this planner; all other input may answer it, ask a question, change groups, "
        "revise earlier state, paste evidence, or combine several of those. Preserve unresolved work. "
        "Use answer_pending only when the user is actually answering the active pending question in natural language. Put the complete answer in its answer argument. "
        "If the same turn explicitly changes a dependency dimension that invalidates the active pending question (for example changing chain or target mode while an old endpoint question is active), do not emit answer_pending for that stale question. Emit the explicit mutation actions first and preserve every supplied value in its owner action: use propose_config_values for named configuration fields such as LOCAL_RPC_URL, rpc_catalog_command for custom-RPC endpoint/method/schema evidence, and rpc_workload_command for scope/weights. The Harness will pause for any required change confirmation; never bind a new value to the old context. "
        "If the user asks whether a prior request was retained, what is complete or missing, what the current state is, or what happens next, emit answer_opening_question with the matching current_config/current_context/next_action topic even while a pending question exists. Preserve that pending question; do not emit answer_pending unless the same turn also supplies its answer. "
        "When an action spec requires source_evidence, use an exact excerpt from the current user_text. For manual scalar fields, the proposed answer itself must occur in that evidence; never copy a value from workflow_state into a new answer. "
        "For a choice question, also set selected_value to exactly one value declared by pending_question.options; never invent a value. "
        "A capability question such as whether/how custom RPC can be added is consultation (topic=extension), not a catalog mutation. "
        "Use one rpc_catalog_command per explicit catalog operation: enter, set_endpoint, set_method, or append_evidence. Emit set_method only when the user supplied an exact wire method token or a complete protocol request from which that identity is explicit; a phrase such as 'my own method' starts the catalog with enter but is not a method identity. Use rpc_workload_command only for scope/weights after method validation. Preserve their user order; do not combine endpoint, schema, and workload into one action or represent them as generic group navigation. "
        "When a schema-evidence question is active and the user describes parameters or response semantics for 'this method' or 'the current method', emit rpc_catalog_command append_evidence with that exact description. Never emit set_method using a method identity copied only from workflow_state. "
        "While an unsupported-family Case 3 handoff is collecting evidence, use secondary_handoff_command append_evidence only when the current user turn contributes protocol, endpoint, request, response, or official-document evidence. Questions about retained evidence, status, meaning, or next steps are consultations and must not append evidence. "
        "Distinguish endpoint roles. When custom_rpc_endpoint or new_chain_endpoint is pending and the user explicitly selects a reachable validation URL, answer that pending question with the selected URL and exact source_evidence. A URL labeled as docs/example/sample/not-selected is evidence only: never map it to LOCAL_RPC_URL, MAINNET_RPC_URL, or the selected validation endpoint. The final benchmark LOCAL_RPC_URL is a separate endpoint_process decision. "
        "Do not choose a chain, target mode, RPC mode, QPS mode, endpoint, or value that the user did not explicitly select or change. Mentioning a current/old value in a status question, comparison, explanation, hypothetical, or no-change instruction must produce consultation only, never a mutation action. "
        "Never select the first target-mode option as a default. If the user has a benchmark goal, chain, RPC, QPS, or observability demands but does not explicitly select fake-node, real-node, or sync-observe, preserve every stated demand and emit request_target_mode_selection. "
        "When the user explicitly orders mutually exclusive workflows (for example sync-observe now and real-node RPC capacity later), emit choose_target_mode for the current goal and queue_workflow_goal for each later goal in stated order. Never apply two target modes to one configuration and never drop the later goal. When the user explicitly asks to start/continue the saved later goal, emit activate_next_workflow_goal. When they explicitly cancel it, emit discard_next_workflow_goal. "
        "When adjacent clauses jointly reject the current mutually exclusive workflow and select its replacement, emit one choose_target_mode for the explicit replacement and map both semantic units to that same action index. The rejection clause is part of the replacement decision; do not invent a separate cancellation action and do not mark it unresolved. "
        "For change_group, group must be one exact name from action_schema.allowed_groups. Domain owner names are not workflow groups: use qps_profile for QPS/benchmark profile, observability for Prometheus/Grafana, workload_rpc for single/mixed, and chain_identity for chain selection. "
        "Use change_group only when the user explicitly asks to enter, return to, revisit, or modify a named area but supplies no concrete mutation for it. Merely mentioning an area in a question, comparison, explanation request, missing-field question, or status audit is not navigation. Set navigation_explicit=true and include exact source_evidence for every change_group. If the user gives a concrete chain, RPC mode, QPS mode, observability mode, or configuration value, emit the corresponding typed mutation action instead of only change_group. "
        "For the QPS area specifically, request_qps_customization requires an explicit request to alter, tune, override, or avoid the defaults of profile values. A request to visit or configure the QPS area before another area, without requesting a profile-value/default change, is change_group(qps_profile), not customization. "
        "A request to resume or return to the current benchmark/configuration flow without naming one exact configuration area is not change_group and must not be guessed as target_mode, qps_profile, or another group. When pending_question exists, emit resume_current_flow so the Harness preserves and renders that exact typed question. When no pending question exists, emit answer_opening_question with topic=next_action. "
        "For choose_target_mode set target_mode_explicit=true only for an explicit selection/change request and include a short exact source_evidence excerpt from user_text. "
        "For set_rpc_mode, set_qps_mode, and set_observability set mutation_explicit=true only for an explicit selection/change request and include exact source_evidence. "
        "A question, comparison, explanation request, hypothetical, or capability inquiry that merely mentions a mode is consultation and must not emit choose_target_mode. "
        "An explicit fake-node, real-node, or sync-observe selection must emit choose_target_mode, not request_target_mode_selection. The request action is only for a missing or genuinely unresolved mode. "
        "Treat chain selection and chain knowledge as independent facts. When the user explicitly says they want or need to test/benchmark one named chain, emit choose_chain for that raw name even if the same turn asks whether the chain exists, is supported, or which protocol it uses. A chain mentioned only as the subject of an existence/support/protocol question is consultation, not selection. An explicit instruction to retain or not modify the current chain forbids choose_chain/change_chain in that turn. Knowledge uncertainty is handled later by chain identity research; it does not erase an explicit benchmark target. "
        "When chain inputs conflict or the user is uncertain between multiple chain names, emit choose_chain/change_chain with chain_candidates containing every candidate and do not silently choose the first. "
        "For other conflicts, carry the conflict in the owning typed action instead of silently choosing the first value. "
        "Unknown chain names remain raw chain selections; the chain/RPC domain performs LLM identity research, optional Google grounding, and user confirmation. "
        "If one turn confirms an unresolved chain is real and also states its protocol family, answer only the current identity question with answer_pending and emit choose_adapter_family for the family; do not pre-answer a future family question. "
        "Use answer_opening_question for identity, capabilities, requirements, workflow, current context/config/job/status/next action, mode comparisons, "
        "field explanations, corrections, and general product questions. Its topic MUST be one of the allowed_topics declared in action_schema; never invent a topic. "
        "Use topic=identity for who/origin questions, capabilities for what the Agent can do, and requirements for what a user must prepare. For supported_chains, omit subject unless the user asks about one specific chain name. "
        "Use topic=workload_config for selected, effective, or default RPC methods and mixed-weight questions that do not request a mutation. "
        "When the user asks what the current prompt, option, field, or pending step means (for example 'what is this for?'), emit "
        "answer_opening_question with topic=config_explanation and subject equal to the pending question field or id; do not classify it as general requirements. "
        "Use propose_config_values for explicit mixed natural language, YAML, JSON, env, shell, or tabular configuration and include unmapped/conflicting values. structured_candidates are deterministic syntax facts scoped to one clause, not executable actions: copy their config_values and unmapped_values into propose_config_values when the user asks to configure or review that block, and include that complete clause as source_evidence. A request in the same turn to review/apply the supplied partial configuration and then ask for remaining required values is processing scope owned by that same propose_config_values action; map that clause to the proposal, do not emit resume_current_flow, and do not mark it unresolved. The review gate still runs before fallback asks for missing fields. Map every structured_candidates.workflow_values entry to its real typed workflow owner (for example RPC_MODE to set_rpc_mode) instead of placing it in environment config. One structured clause may map to several owner actions. Never silently omit an unmapped value. Do not propose a value that the user labels as an example, sample, documentation value, rejected value, or not selected. "
        "Use analyze_evidence only when the user asks to interpret supplied or previously collected logs/errors/output; include the supplied block as evidence. A configuration paste that asks to configure or review values is propose_config_values plus its other mutations/consultations, not evidence analysis. Use analyze_report for real job/artifact questions. "
        "Put each action's allowed arguments directly beside type; never wrap them in an arguments object. "
        "Never express workflow behavior in prose. Follow action_schema exactly for each action; fields not declared there are rejected. "
        "Return an object with actions, semantic_units, and optional document-level conflicts and reason. semantic_units must account for every supplied clause. Each row is {unit_id, clause_id, source_text, disposition:'action'|'unresolved', action_indexes:[zero-based action indexes], optional scope_constraint:'consultation_only', reason}. Use disposition action when typed actions preserve that exact unit. Use unresolved with an empty action_indexes list when safe interpretation is unavailable. Every URL, exact wire RPC method, or concrete value claimed as preserved must occur in the mapped owning action. A full-clause unit may map several independent typed actions when that is the only lossless contiguous partition; it may not hide an omitted request. Never silently omit source text."
    )
