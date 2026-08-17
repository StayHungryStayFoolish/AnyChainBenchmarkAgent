"""Orientation and consultation behavior.

This domain answers questions from product facts and current state. It never
mutates benchmark configuration.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

from agent.knowledge.entry_contract import ALL_RUNTIME_FIELDS
from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.workflows.group_registry import GROUPS, group_applicable
from agent.validators.rpc_workload import default_workload
from agent.runners.job_manager import get_job, list_jobs, verify_job_receipt
from agent.utils.redaction import redact
from ..action_registry import canonical_consultation_topic
from ..contracts import (
    ActionProposal,
    CheckpointCommand,
    FailureDescriptor,
    HandlerResult,
    ResponseFragment,
    StateDelta,
)
from ..failures import failure_record_response_fragment, unresolved_recovery
from ..oracle import compute_next_action
from ..questions import choice_question as _choice_question, question_text
from ..response_catalog import render_text_ref
from ..secret_refs import secret_references_in_value
from ..state import AgentGraphState
from .chain_identity import research_chain_identity
from .orientation_receipts import build_orientation_response_receipt

choice_question = partial(_choice_question, owner="orientation")


def has_resumable_configuration(state: AgentGraphState) -> bool:
    pending = state.get("pending_question") or {}
    pending_id = str(pending.get("id") or "")
    identity = state.get("chain_identity") or {}
    return any(
        (
            bool(state.get("target_mode")),
            bool(identity.get("canonical") or identity.get("raw")),
            bool(state.get("confirmed_config")),
            bool(state.get("rpc_mode")),
            bool(state.get("qps_profile")),
            bool(state.get("observability")),
            bool(state.get("sync_observe")),
            bool(state.get("workflow_goals")),
            bool(pending_id and pending_id not in {"opening_next_action", "resume_harness_session"}),
            bool((state.get("checkpoint_recovery") or {}).get("status") == "quarantined"),
        )
    )


def resume_question(
    state: AgentGraphState,
    *,
    active_action_id: str = "",
) -> dict[str, Any]:
    recovery = state.get("checkpoint_recovery") or {}
    if recovery.get("status") == "quarantined":
        prompt = question_text("question.orientation.resume_quarantined.prompt")
        values = ("inspect_quarantine", "retain_safe", "reset")
    else:
        identity = state.get("chain_identity") or {}
        confirmed = state.get("confirmed_config") or {}
        qps = state.get("qps_profile") or {}
        observability = state.get("observability") or {}
        prompt = question_text(
            "question.orientation.resume.prompt",
            target_mode=str(state.get("target_mode") or "<not selected>"),
            workflow_mode=str(state.get("workflow_mode") or "<not selected>"),
            chain=str(
                identity.get("canonical")
                or identity.get("raw")
                or "<not selected>"
            ),
            rpc_mode=str(state.get("rpc_mode") or "<not selected>"),
            qps=str(qps.get("mode") or "<not selected>"),
            observability=str(observability.get("mode") or "<not selected>"),
            confirmed_fields=(
                ", ".join(sorted(str(key) for key in confirmed))
                if confirmed
                else "<none>"
            ),
            deferred_request_count=_deferred_action_count(
                state,
                active_action_id=active_action_id,
            ),
            saved_workflow_goals=(
                "; ".join(
                    ": ".join(
                        part
                        for part in (
                            str(item.get("target_mode") or "").strip(),
                            str(item.get("goal") or "").strip(),
                        )
                        if part
                    )
                    or "<unspecified>"
                    for item in state.get("workflow_goals") or []
                    if isinstance(item, dict)
                )
                or "<none>"
            ),
        )
        values = ("continue", "modify", "reset")
    question = choice_question(
        "opening",
        "resume_harness_session",
        prompt,
        field="resume_harness_session",
        options=[
            {
                "id": "1",
                "label": _resume_option_label(values[0]),
                "value": values[0],
                **(
                    {"semantic_action": "continue_current_flow"}
                    if values[0] == "continue"
                    else {}
                ),
                "expected_patch": {"checkpoint_recovery.status": "quarantined"}
                if values[0] == "inspect_quarantine"
                else {"resume_context": {}},
                "return_policy": "stop_after_response" if values[0] == "inspect_quarantine" else "fallback",
            },
            {
                "id": "2",
                "label": _resume_option_label(values[1]),
                "value": values[1],
                "expected_patch": {"resume_context": {}},
            },
            {
                "id": "3",
                "label": _resume_option_label(values[2]),
                "value": values[2],
                "expected_patch": {"confirmed_config": {}},
            },
        ],
        queue_barrier=True,
    )
    if state.get("action_queue"):
        question["resume_action_queue"] = True
    return question


def _deferred_action_count(
    state: AgentGraphState,
    *,
    active_action_id: str = "",
) -> int:
    """Count queued work without treating the executing action as deferred."""

    excluded = str(active_action_id or "")
    return sum(
        1
        for item in state.get("action_queue") or ()
        if isinstance(item, dict)
        and (
            not excluded
            or str(item.get("action_id") or "") != excluded
        )
    )


def _resume_option_label(value: str):
    return question_text(f"question.orientation.resume.option.{value}")


def session_reset_confirmation_question() -> dict[str, Any]:
    """Build the sole user-facing authority for destructive session reset."""

    return choice_question(
        "opening",
        "session_reset_confirm",
        question_text("question.orientation.session_reset_confirm.prompt"),
        field="session_reset_confirm",
        options=[
            {
                "id": "yes",
                "label": question_text("question.common.option.yes"),
                "value": True,
                "expected_patch": {"confirmed_config": {}},
            },
            {
                "id": "no",
                "label": question_text("question.common.option.no"),
                "value": False,
                "expected_patch": {},
                "return_policy": "stop_after_response",
            },
        ],
        queue_barrier=True,
    )


def resume_modify_group_question(state: AgentGraphState) -> dict[str, Any]:
    """Build the typed destination selector for a resumed configuration.

    The workflow registry owns which destinations exist and which modes they
    apply to. Each option delegates the actual transition to the coordinator's
    ``change_group`` action, preserving one navigation authority.
    """

    destinations = [
        spec
        for spec in GROUPS
        if spec.name != "opening"
        and spec.navigation_entry == "question_or_status"
        and spec.resume_selector
        and group_applicable(state, spec)
    ]
    return choice_question(
        "opening",
        "resume_modify_group",
        question_text("question.orientation.resume_modify.prompt"),
        field="resume_modify_group",
        options=[
            {
                "id": str(index),
                "label": question_text(
                    "question.orientation.group.option",
                    group=spec.name,
                ),
                "description": question_text(
                    "question.orientation.group.description",
                    fields=", ".join(
                        field
                        for field in spec.fields
                        if field not in spec.immutable_fields
                    )
                    or "<none>",
                ),
                "value": spec.name,
                "action": {
                    "type": "change_group",
                    "group": spec.name,
                    "navigation_explicit": True,
                },
                "completion_effect": question_text(
                    "question.orientation.group.completion",
                    group=spec.name,
                ),
                "expected_patch": {},
                "return_policy": "fallback",
            }
            for index, spec in enumerate(destinations, start=1)
        ],
        accepted_action_types=("change_group",),
        queue_barrier=True,
    )


def apply_orientation_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply an orientation action without stealing workflow ownership."""

    action_type = action.action_type
    if action_type == "prepare_session_entry":
        current_pending = dict(state.get("pending_question") or {})
        semantic_draft = dict(state.get("semantic_plan_draft") or {})
        if (
            semantic_draft.get("status") == "awaiting_clarification"
            and current_pending.get("semantic_draft_binding")
        ):
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                pending_question=current_pending,
                completion="blocked",
                stop_after_response=True,
            )
        resumable = has_resumable_configuration(state)
        question = (
            resume_question(
                state,
                active_action_id=action.action_id,
            )
            if resumable
            else opening_question(state)
        )
        delta = StateDelta()
        if (
            resumable
            and current_pending
            and str(current_pending.get("id") or "") != "resume_harness_session"
        ):
            delta = StateDelta.set_values({
                "resume_context": {
                    "pending_question": current_pending,
                    "active_group": str(
                        state.get("active_group")
                        or current_pending.get("group")
                        or ""
                    ),
                }
            })
        return HandlerResult(
            delta=delta,
            consumed_action_ids=(action.action_id,),
            pending_question=question,
            next_group="opening",
            completion="blocked",
            stop_after_response=True,
        )
    if action_type == "set_response_language":
        language = str(action.arguments.get("language") or "").strip().lower()
        if language not in {"zh", "en"}:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.orientation.failure.invalid_language",
                    source=__name__,
                )
            )
        return HandlerResult(
            delta=StateDelta.set_values({"language": language}),
            consumed_action_ids=(action.action_id,),
            completion="completed",
        )
    if action_type == "greeting":
        language = str(state.get("language") or "en")
        question = None
        if not state.get("pending_question") and not has_resumable_configuration(state):
            question = opening_question(state)
        fragment = ResponseFragment(
            kind="message",
            message_id="harness.orientation.greeting",
            source=__name__,
        )
        return HandlerResult(
            response_fragments=(fragment,),
            control_receipts=(
                build_orientation_response_receipt(
                    state,
                    action,
                    topic="identity",
                    response_fragments=(fragment,),
                ),
            ),
            pending_question=question,
            next_group="opening" if question else "",
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
        )
    if action_type == "clarify_unresolved":
        clauses = [
            (
                render_text_ref(
                    question_text(
                        "harness.orientation.unresolved_sensitive_item"
                    ),
                    str(state.get("language") or "en"),
                    kind="message",
                )
                if secret_references_in_value(str(item))
                else str(item).strip()
            )
            for item in action.arguments.get("clauses") or []
            if str(item).strip()
        ]
        if not clauses:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.orientation.failure.unresolved_clauses_required",
                    source=__name__,
                )
            )
        details = "\n".join(f"- {item}" for item in clauses)
        return HandlerResult(
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.orientation.clarify_unresolved",
                    arguments={"details": details},
                    source=__name__,
                ),
            ),
            pending_question=dict(state.get("pending_question") or {}) or None,
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
            completion="unchanged",
        )
    if action_type == "request_session_reset":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            pending_question=session_reset_confirmation_question(),
            next_group="opening",
            completion="blocked",
            stop_after_response=True,
        )
    if action_type == "reset_session":
        return HandlerResult(
            checkpoint_command=CheckpointCommand(
                "reset",
                preserve_remaining_actions=True,
            ),
            consumed_action_ids=(action.action_id,),
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.orientation.session_reset",
                    source=__name__,
                ),
            ),
            completion="completed",
            stop_after_response=True,
        )
    if action_type in {"ask_capabilities", "answer_opening_question"}:
        raw = {"topic": "capabilities"} if action_type == "ask_capabilities" else dict(action.arguments)
        requested_topic = canonical_consultation_topic(raw.get("topic"))
        topic = effective_consultation_topic(state, raw.get("topic"))
        raw["topic"] = topic
        job_facts = None
        if topic in {
            "current_context",
            "next_action",
            "current_job",
            "job_status",
            "execution_status",
        }:
            job_facts = _job_status_facts(state)
        pending = dict(state.get("pending_question") or {})
        if topic == "recommendation" and not pending and not state.get("target_mode"):
            pending = recommendation_question(state)
        fragment = consultation_fragment(
            state,
            raw,
            job_facts=job_facts[:3] if job_facts is not None else None,
        )
        source_receipts: tuple[Mapping[str, Any], ...] = ()
        if (
            fragment.message_id
            == "harness.orientation.consultation.job_verified"
            and job_facts is not None
            and job_facts[2]
            and job_facts[3]
        ):
            source_receipts = (job_facts[3],)
        fragments = [fragment]
        if (
            requested_topic in {"requirements", "workflow"}
            and _at_preflight_gate(state)
        ):
            fragments.append(
                consultation_fragment(
                    state,
                    {"topic": "execution_preflight_smoke"},
                )
            )
        active_failure = _active_failure_record(state)
        if topic == "current_config" and active_failure:
            fragments.insert(0, failure_record_response_fragment(active_failure))
        return HandlerResult(
            response_fragments=tuple(fragments),
            control_receipts=(
                build_orientation_response_receipt(
                    state,
                    action,
                    topic=topic or "capabilities",
                    response_fragments=tuple(fragments),
                    source_receipts=source_receipts,
                    projection_override=(
                        {
                            "job_id": str(job_facts[0]),
                            "job_status": str(job_facts[1]),
                        }
                        if source_receipts and job_facts is not None
                        else None
                    ),
                ),
            ),
            pending_question=pending or None,
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
        )
    if action_type == "answer_pending":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            blocker=FailureDescriptor(
                code="harness.orientation.failure.answer_pending_not_resolved",
                source=__name__,
            ),
        )
    if action_type == "unknown":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            blocker=FailureDescriptor(
                code="harness.orientation.failure.no_safe_action",
                source=__name__,
            ),
        )
    return HandlerResult(
        blocker=FailureDescriptor(
            code="harness.orientation.failure.unsupported_action",
            arguments={"action_type": action_type},
            source=__name__,
        )
    )


def effective_consultation_topic(
    state: AgentGraphState,
    requested_topic: Any,
) -> str:
    """Resolve state-dependent consultation semantics at one authority."""

    topic = canonical_consultation_topic(requested_topic)
    if topic != "next_action":
        return topic
    pending = state.get("pending_question") or {}
    if str(pending.get("group") or "") == "preflight_smoke_execution":
        return "execution_preflight_smoke"
    if compute_next_action(state).next_blocking_group == "preflight_smoke_execution":
        return "execution_preflight_smoke"
    return topic


def _at_preflight_gate(state: AgentGraphState) -> bool:
    pending = state.get("pending_question") or {}
    return bool(
        str(pending.get("group") or "") == "preflight_smoke_execution"
        or compute_next_action(state).next_blocking_group
        == "preflight_smoke_execution"
    )


def consultation_fragment(
    state: AgentGraphState,
    action: dict[str, Any],
    *,
    job_facts: tuple[str, str, bool] | None = None,
) -> ResponseFragment:
    """Translate a consultation topic into one registered response contract."""

    topic = effective_consultation_topic(state, action.get("topic"))
    subject = str(action.get("subject") or "").strip()
    facts = state.get("framework_summary") or {}
    static_ids = {
        "identity": "identity",
        "execution_preflight_smoke": "preflight_smoke",
        "recommendation": "recommendation",
        "reset_help": "reset_help",
        "reset": "reset_help",
        "evidence_help": "evidence_help",
        "log_help": "evidence_help",
    }
    if topic in static_ids:
        return ResponseFragment(
            kind="message",
            message_id=f"harness.orientation.consultation.{static_ids[topic]}",
            source=__name__,
        )
    if topic in {"capabilities", "agent_capability"}:
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.capabilities",
            arguments={
                "chain_count": str(facts.get("chain_count", "?")),
                "family_count": str(facts.get("family_count", "?")),
                "method_count": str(facts.get("unique_rpc_method_count", "?")),
            },
            source=__name__,
        )
    if topic == "extension":
        current_chain = str((state.get("chain_identity") or {}).get("canonical") or "")
        chain = (
            canonicalize_chain_scalar(
                subject or current_chain,
                known_chains=set(repo_chain_names()),
            )
            or "<not selected>"
        )
        defaults = ", ".join(default_workload(chain).get("methods") or []) or "<none>"
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.extension",
            arguments={"chain": chain, "defaults": defaults},
            source=__name__,
        )
    if topic == "supported_chains":
        configured = sorted(repo_chain_names())
        chain = canonicalize_chain_scalar(
            subject,
            known_chains=set(configured),
        ) if subject else ""
        if chain:
            defaults = default_workload(chain)
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.supported_chain",
                arguments={
                    "chain": chain,
                    "single": str(defaults.get("single") or "<none>"),
                    "mixed": ", ".join(
                        f"{row.get('method')}={row.get('weight')}"
                        for row in defaults.get("mixed_weighted") or []
                        if isinstance(row, dict) and row.get("method")
                    ) or "<none>",
                    "methods": ", ".join(defaults.get("methods") or []) or "<none>",
                },
                source=__name__,
            )
        if subject:
            resolution = research_chain_identity(state, subject)
            web = resolution.get("web_grounding") or {}
            if not isinstance(web, dict):
                web = {}
            exists = resolution.get("chain_exists")
            return ResponseFragment(
                kind="message",
                message_id=(
                    "harness.orientation.consultation.chain_research_grounded"
                    if web.get("used")
                    else "harness.orientation.consultation.chain_research_ungrounded"
                ),
                arguments={
                    "subject": subject,
                    "exists": (
                        "yes" if exists is True else "no" if exists is False else "unconfirmed"
                    ),
                    "chain": str(
                        resolution.get("canonical_chain_name") or subject
                    ),
                    "protocol": str(
                        resolution.get("protocol_or_api") or "<unconfirmed>"
                    ),
                    "adapter_family": str(
                        resolution.get("adapter_family") or "<unconfirmed>"
                    ),
                    "confidence": str(
                        resolution.get("confidence") or "unknown"
                    ),
                    "evidence": str(
                        resolution.get("evidence_summary") or "<none>"
                    ),
                    **(
                        {"grounding": str(web.get("summary") or "<none>")}
                        if web.get("used")
                        else {}
                    ),
                },
                source=__name__,
            )
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.supported_chains",
            arguments={"chains": ", ".join(configured)},
            source=__name__,
        )
    if topic in {"requirements", "workflow"}:
        return ResponseFragment(
            kind="message",
            message_id=(
                "harness.orientation.consultation.requirements_with_recommendation"
                if not state.get("target_mode")
                else "harness.orientation.consultation.requirements"
            ),
            source=__name__,
        )
    if topic in {"mode_comparison", "performance_benchmark_guidance"}:
        return ResponseFragment(
            kind="message",
            message_id=(
                "harness.orientation.consultation.performance_guidance"
                if topic == "performance_benchmark_guidance"
                else "harness.orientation.consultation.mode_comparison"
            ),
            source=__name__,
        )
    if topic == "sync_observe_metrics":
        current_chain = str(
            (state.get("chain_identity") or {}).get("canonical")
            or (state.get("chain_identity") or {}).get("raw")
            or ""
        )
        configured = set(repo_chain_names())
        requested_chain = subject or current_chain
        chain = canonicalize_chain_scalar(
            requested_chain,
            known_chains=configured,
        )
        profiles = [
            dict(item)
            for item in facts.get("client_metric_profiles") or []
            if isinstance(item, dict)
        ]
        matches = [
            profile
            for profile in profiles
            if chain
            and chain in {
                str(item).strip().lower()
                for item in profile.get("chains") or []
            }
        ]
        if len(matches) == 1:
            profile = matches[0]
            response_language = (
                "zh" if str(state.get("language") or "en") == "zh" else "en"
            )
            missing_endpoint = profile.get("missing_endpoint")
            if not isinstance(missing_endpoint, dict):
                missing_endpoint = {}
            requirement_separator = "，" if response_language == "zh" else ", "
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.sync_observe_metrics_profile",
                arguments={
                    "chain": chain,
                    "profile": str(profile.get("display_name") or profile.get("profile_id") or "<unknown>"),
                    "native_samples": ", ".join(
                        str(item) for item in profile.get("native_samples") or []
                    ) or "<none>",
                    "report_kpis": ", ".join(
                        str(item) for item in profile.get("report_kpis") or []
                    ) or "<none>",
                    "metrics_path": str(profile.get("metrics_path") or "<not specified>"),
                    "requirements": requirement_separator.join(
                        str(item.get(response_language) or "")
                        for item in profile.get("requirements") or []
                        if isinstance(item, dict)
                        and str(item.get(response_language) or "")
                    ) or "<none>",
                    "missing_endpoint_behavior": str(
                        missing_endpoint.get(response_language) or "<not specified>"
                    ),
                },
                source=__name__,
            )
        registered_chains = sorted(
            {
                str(item).strip()
                for profile in profiles
                for item in profile.get("chains") or []
                if str(item).strip()
            }
        )
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.sync_observe_metrics_generic",
            arguments={
                "chain": chain or requested_chain or "<not selected>",
                "registered_chains": ", ".join(registered_chains) or "<none>",
            },
            source=__name__,
        )
    if topic == "sync_observe_behavior":
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.sync_observe_behavior",
            source=__name__,
        )
    if topic in {"correction", "clarification"} and not _active_failure_record(state):
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.clarification",
            source=__name__,
        )
    if topic in {"correction", "clarification"}:
        return failure_record_response_fragment(_active_failure_record(state) or {})
    identity = state.get("chain_identity") or {}
    confirmed = state.get("confirmed_config") or {}
    pending = state.get("pending_question") or {}
    if topic == "current_config":
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.current_config",
            arguments={
                "target_mode": str(state.get("target_mode") or "<not selected>"),
                "workflow_mode": str(state.get("workflow_mode") or "<not selected>"),
                "chain": str(identity.get("canonical") or identity.get("raw") or "<not selected>"),
                "rpc_mode": str(state.get("rpc_mode") or "<not selected>"),
                "qps_mode": str((state.get("qps_profile") or {}).get("mode") or "<not selected>"),
                "observability": str((state.get("observability") or {}).get("mode") or "<not selected>"),
                "confirmed_fields": ", ".join(sorted(confirmed)) or "<none>",
                "pending_question": str(pending.get("id") or "<none>"),
                "pending_field": str(pending.get("field") or "<none>"),
            },
            source=__name__,
        )
    if topic == "workload_config":
        workload = state.get("workload") or {}
        chain = str(identity.get("canonical") or identity.get("raw") or "<not selected>")
        if workload.get("confirmed") and workload.get("methods"):
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.workload_effective",
                arguments={
                    "chain": chain,
                    "rpc_mode": str(state.get("rpc_mode") or "<not selected>"),
                    "methods": ", ".join(str(item) for item in workload.get("methods") or []) or "<none>",
                    "weights": ", ".join(
                        f"{method}={weight}"
                        for method, weight in (workload.get("weights") or {}).items()
                    ) or "<not applicable>",
                },
                source=__name__,
            )
        defaults = default_workload(chain) if chain != "<not selected>" else {}
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.workload_template",
            arguments={
                "chain": chain,
                "single": str(defaults.get("single") or "<none>"),
                "mixed": ", ".join(
                    f"{row.get('method')}={row.get('weight')}"
                    for row in defaults.get("mixed_weighted") or []
                    if isinstance(row, dict) and row.get("method")
                ) or "<none>",
            },
            source=__name__,
        )
    if topic in {"current_context", "next_action"}:
        next_action = compute_next_action(state)
        collection = state.get("evidence_collection") or {}
        evidence_lines = (
            list(collection.get("lines") or [])
            if isinstance(collection, dict)
            else []
        )
        if evidence_lines:
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.current_context_evidence",
                arguments={
                    "evidence_lines": len(evidence_lines),
                    "preview": str(redact(evidence_lines)),
                },
                source=__name__,
            )
        common_arguments = {
            "active_group": str(state.get("active_group") or "<none>"),
            "execution_status": str((state.get("execution_state") or {}).get("status") or "not_requested"),
            "evidence_lines": 0,
        }
        if pending:
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.current_context_pending",
                arguments={
                    **common_arguments,
                    "pending_question": str(pending.get("id") or "<none>"),
                    "pending_field": str(pending.get("field") or "<none>"),
                },
                source=__name__,
            )
        if str((state.get("job") or {}).get("job_id") or "").strip():
            job_id, status, verified = (
                job_facts
                if job_facts is not None
                else _job_status_facts(state)[:3]
            )
            return ResponseFragment(
                kind="message",
                message_id=(
                    "harness.orientation.consultation.job_verified"
                    if verified
                    else "harness.orientation.consultation.job_unverified"
                ),
                arguments={"job_id": job_id, "status": status},
                source=__name__,
            )
        if next_action.next_blocking_group == "preflight_smoke_execution":
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.current_context_preflight",
                source=__name__,
            )
        return ResponseFragment(
            kind="message",
            message_id="harness.orientation.consultation.current_context_next_group",
            arguments={
                **common_arguments,
                "next_group": str(next_action.next_blocking_group or "<none>"),
            },
            source=__name__,
        )
    if topic in {"startup_discovery", "environment_inference", "environment_readiness"}:
        discovery = state.get("discovery") or {}
        cloud = discovery.get("cloud") or {}
        deployment = discovery.get("deployment") or {}
        host = discovery.get("host") or {}
        network = discovery.get("network") or {}
        disks = discovery.get("disks") or {}
        dependencies = discovery.get("dependencies") or {}
        return ResponseFragment(
            kind="message",
            message_id=(
                "harness.orientation.consultation.environment_snapshot_blocked"
                if dependencies.get("missing_required")
                else "harness.orientation.consultation.environment_snapshot_ready"
            ),
            arguments={
                "cloud": str(cloud.get("provider") or "other"),
                "deployment": str(cloud.get("platform") or deployment.get("type") or "<unknown>"),
                "region": str(cloud.get("region") or "<not detected>"),
                "zone": str(cloud.get("zone") or "<not detected>"),
                "machine": str(cloud.get("machine_type") or host.get("machine_type") or host.get("hostname") or "<unknown>"),
                "cpu": str(host.get("cpu_count") or host.get("cpu") or "<unknown>"),
                "memory": str(host.get("memory_gib") or host.get("memory") or "<unknown>"),
                "network": str(network.get("default_interface") or "<none>"),
                "ledger": str(disks.get("proposed_ledger_device") or "<none>"),
                "accounts": str(disks.get("proposed_accounts_device") or "<none detected>"),
                "missing_required": ", ".join(
                    str(item) for item in dependencies.get("missing_required") or []
                )
                or "<none>",
                "missing_optional": ", ".join(
                    str(item) for item in dependencies.get("missing_optional") or []
                )
                or "<none>",
            },
            source=__name__,
        )
    if topic == "config_explanation":
        identifier = subject or str(pending.get("field") or pending.get("id") or "")
        normalized_identifier = (
            identifier.strip().lower().replace("-", "_").replace(" ", "_")
        )
        if normalized_identifier in {
            "single",
            "mixed",
            "rpc_mode",
            "rpc_mode_single",
            "rpc_mode_mixed",
        }:
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.rpc_mode",
                source=__name__,
            )
        field = _runtime_field(identifier)
        if field is not None:
            return ResponseFragment(
                kind="message",
                message_id=(
                    "harness.orientation.consultation.config_field_required"
                    if field.required
                    else "harness.orientation.consultation.config_field_optional"
                ),
                arguments={
                    "field": str(field.env or field.key),
                    "label": str(field.label),
                    "purpose": str(field.description or field.reason),
                    "modes": ", ".join(mode.replace("_", "-") for mode in field.applies_to),
                },
                source=__name__,
            )
        if identifier in {"new_chain_endpoint", "custom_rpc_endpoint"}:
            return ResponseFragment(
                kind="message",
                message_id=f"harness.orientation.consultation.{identifier}",
                source=__name__,
            )
        if identifier == "custom_rpc_fixture_choice":
            workload = state.get("workload") or {}
            return ResponseFragment(
                kind="message",
                message_id="harness.orientation.consultation.custom_rpc_fixture_choice",
                arguments={
                    "methods": ", ".join(
                        str(item) for item in workload.get("methods") or []
                    )
                    or "<none>",
                    "preserved_fields": ", ".join(sorted(confirmed)) or "<none>",
                },
                source=__name__,
            )
    if topic in {"current_job", "job_status", "execution_status"}:
        job_id, status, verified = (
            job_facts
            if job_facts is not None
            else _job_status_facts(state)[:3]
        )
        return ResponseFragment(
            kind="message",
            message_id=(
                "harness.orientation.consultation.job_verified"
                if verified
                else "harness.orientation.consultation.job_unverified"
                if job_id
                else "harness.orientation.consultation.job_none"
            ),
            arguments=(
                {"job_id": job_id, "status": status}
                if job_id
                else {}
            ),
            source=__name__,
        )
    return ResponseFragment(
        kind="message",
        message_id="harness.orientation.consultation.general",
        source=__name__,
    )


def _runtime_field(identifier: str):
    normalized = str(identifier or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "").rstrip("s")
    if not normalized:
        return None
    return next(
        (
            field
            for field in ALL_RUNTIME_FIELDS
            if normalized
            in {
                str(field.env).lower().replace("_", ""),
                str(field.key).lower().replace("_", "").replace("-", ""),
                str(field.label).lower().replace(" ", "").rstrip("s"),
            }
        ),
        None,
    )


def _job_status_facts(
    state: AgentGraphState,
) -> tuple[str, str, bool, dict[str, Any]]:
    job = dict(state.get("job") or {})
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        try:
            jobs = list_jobs(limit=1)
            job = dict(jobs[0]) if jobs else {}
            job_id = str(job.get("job_id") or "").strip()
        except Exception:
            return "", "", False, {}
    if not job_id:
        return "", "", False, {}
    try:
        persisted = get_job(job_id)
    except (FileNotFoundError, OSError, ValueError):
        return job_id, str(job.get("status") or "unknown"), False, {}
    receipts = persisted.get("execution_receipts") or {}
    receipt = dict(receipts.get("last_read") or {}) if isinstance(receipts, dict) else {}
    verified = bool(
        verify_job_receipt(receipt)
        and receipt.get("job_id") == job_id
        and receipt.get("observed_status") == persisted.get("status")
        and receipt.get("observed_status") == receipt.get("persisted_status")
    )
    status = str(
        receipt.get("observed_status")
        if verified
        else job.get("status") or persisted.get("status") or "unknown"
    )
    return job_id, status, verified, receipt if verified else {}


def apply_orientation_answer(
    state: AgentGraphState,
    question: dict[str, Any],
    value: Any,
    user_text: str,
) -> HandlerResult:
    """Apply opening/session answers without performing global routing."""

    question_id = str(question.get("id") or "")
    if question_id == "session_reset_confirm":
        if value is True:
            return HandlerResult(
                checkpoint_command=CheckpointCommand(
                    "reset",
                    preserve_remaining_actions=True,
                ),
                response_fragments=(
                    ResponseFragment(
                        kind="message",
                        message_id="harness.orientation.session_reset",
                        source=__name__,
                    ),
                ),
                clear_pending=True,
                completion="completed",
                stop_after_response=True,
            )
        return HandlerResult(
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.orientation.session_reset_cancelled",
                    source=__name__,
                ),
            ),
            clear_pending=True,
            completion="completed",
            stop_after_response=False,
        )
    if question_id == "resume_harness_session":
        recovery = state.get("checkpoint_recovery") or {}
        if value == "inspect_quarantine":
            safe = recovery.get("safe_confirmed_config") or {}
            return HandlerResult(
                response_fragments=(
                    ResponseFragment(
                        kind="warning",
                        message_id="harness.orientation.quarantine_inspection",
                        arguments={
                            "reason": str(recovery.get("error_type") or "unknown"),
                            "safe_values": str(safe or "<none>"),
                        },
                        source=__name__,
                    ),
                ),
                pending_question=question,
                completion="blocked",
                stop_after_response=True,
            )
        if value == "retain_safe":
            return HandlerResult(
                checkpoint_command=CheckpointCommand(
                    "retain_safe",
                    confirmed_config=dict(recovery.get("safe_confirmed_config") or {}),
                ),
                response_fragments=(
                    ResponseFragment(
                        kind="message",
                        message_id="harness.orientation.safe_values_retained",
                        source=__name__,
                    ),
                ),
                clear_pending=True,
                next_group="opening",
                completion="completed",
                stop_after_response=True,
            )
        if value == "reset":
            return HandlerResult(
                checkpoint_command=CheckpointCommand("reset"),
                response_fragments=(
                    ResponseFragment(
                        kind="message",
                        message_id="harness.orientation.session_reset",
                        source=__name__,
                    ),
                ),
                completion="completed",
                stop_after_response=True,
            )
        if value == "modify":
            modify_question = resume_modify_group_question(state)
            return HandlerResult(
                delta=StateDelta.set_values({"resume_context": {}}),
                pending_question=modify_question,
                next_group="opening",
                completion="blocked",
                stop_after_response=True,
            )
        resume_context = dict(state.get("resume_context") or {})
        restored_pending = dict(resume_context.get("pending_question") or {})
        return HandlerResult(
            delta=StateDelta.set_values({"resume_context": {}}),
            pending_question=restored_pending or None,
            clear_pending=not restored_pending,
            next_group=str(resume_context.get("active_group") or restored_pending.get("group") or "opening"),
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.orientation.session_resumed",
                    source=__name__,
                ),
            ),
            completion="blocked" if restored_pending else "completed",
            # A restored question is already an actionable stopping point.
            # Without one, continue through the normal group fallback so a
            # resumed session cannot end on a status-only response.
            stop_after_response=bool(restored_pending),
        )
    if question_id == "accept_recommendation":
        if not value:
            return HandlerResult(
                response_fragments=(
                    ResponseFragment(
                        kind="message",
                        message_id="harness.orientation.recommendation_declined",
                        source=__name__,
                    ),
                ),
                clear_pending=True,
                next_group="opening",
                completion="completed",
            )
        recommended = question.get("recommended_setup") if isinstance(question.get("recommended_setup"), dict) else {}
        mode = str(recommended.get("target_mode") or "fake-node")
        return HandlerResult(
            followup_actions=(
                {
                    "type": "choose_target_mode",
                    "target_mode": mode,
                    "target_mode_explicit": True,
                    "source_evidence": str(user_text or "").strip(),
                    "selection_contract_verified": True,
                    "confidence": "high",
                },
            ),
            response_fragments=(
                ResponseFragment(
                    kind="message",
                    message_id="harness.orientation.recommendation_accepted",
                    arguments={"mode": mode},
                    source=__name__,
                ),
            ),
            clear_pending=True,
            completion="completed",
        )
    return HandlerResult(
        blocker=FailureDescriptor(
            code="harness.orientation.failure.unsupported_question",
            arguments={"question_id": question_id},
            source=__name__,
        )
    )


def opening_question(state: AgentGraphState) -> dict[str, Any]:
    return choice_question(
        "opening",
        "opening_next_action",
        question_text("question.orientation.opening.prompt"),
        field="target_mode",
        options=[
            {
                "label": question_text("question.orientation.opening.fake_node.label"),
                "description": question_text(
                    "question.orientation.opening.fake_node.description"
                ),
                "value": "fake-node",
                "action": {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "fake-node", "workflow_mode": "rpc_benchmark"},
            },
            {
                "label": question_text("question.orientation.opening.real_node.label"),
                "description": question_text(
                    "question.orientation.opening.real_node.description"
                ),
                "value": "real-node",
                "action": {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "real-node", "workflow_mode": "rpc_benchmark"},
            },
            {
                "label": question_text(
                    "question.orientation.opening.sync_observe.label"
                ),
                "description": question_text(
                    "question.orientation.opening.sync_observe.description"
                ),
                "value": "sync-observe",
                "action": {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "sync-observe", "workflow_mode": "sync_observe"},
            },
            {
                "label": question_text("question.orientation.opening.info.label"),
                "description": question_text(
                    "question.orientation.opening.info.description"
                ),
                "value": "info",
                "action": {"type": "answer_opening_question", "topic": "capabilities"},
                "expected_patch": {},
                "return_policy": "stop_after_response",
            },
        ],
    )


def recommendation_question(state: AgentGraphState) -> dict[str, Any]:
    """Create the executable contract for the low-risk validation recommendation."""

    question = choice_question(
        "opening",
        "accept_recommendation",
        question_text("question.orientation.recommendation.prompt"),
        field="accept_recommendation",
        kind="yes_no",
        options=[
            {
                "label": question_text("question.control.option.yes"),
                "value": True,
                "expected_patch": {"active_group": "opening"},
            },
            {
                "label": question_text("question.control.option.no"),
                "value": False,
                "expected_patch": {"active_group": "opening"},
            },
        ],
        queue_barrier=True,
    )
    question["recommended_setup"] = {"target_mode": "fake-node"}
    return question


def _active_failure_record(state: AgentGraphState) -> dict[str, Any]:
    recovery = state.get("failure_recovery") or {}
    record = recovery.get("record") if unresolved_recovery(recovery) else None
    if not isinstance(record, dict):
        endpoint_record = (state.get("endpoint_evidence") or {}).get("last_failure_record")
        record = endpoint_record if isinstance(endpoint_record, dict) else None
    if not record:
        return {}
    return dict(record)
