"""LangGraph runtime wrapper for AnyChain Agent Harness."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, TypedDict

from langgraph.runtime import Runtime

from .checkpoints import create_sqlite_checkpointer, default_checkpoint_path
from ..llm.config import load_llm_config
from ..llm.types import ensure_turn_active, llm_turn_scope
from .coordinator import (
    admit_turn_step,
    adjudicate_turn_step,
    compose_turn_step,
    compile_owner_turn_step,
    commit_selected_action_step,
    commit_side_effect_receipt_step,
    fallback_turn_step,
    partition_turn_step,
    prepare_turn_step,
    review_plan_turn_step,
    invoke_idempotent_side_effect_step,
    mark_side_effect_invoking_step,
    select_action_step,
    execute_selected_owner_step,
)
from .failures import build_failure_record
from .invariants import StateInvariantError, validate_state
from .domains.recovery import question_for_recovery
from .response import finalize_turn_response, reset_turn_response
from .runtime_identity import repository_revision
from .state import AgentGraphState, RESET_PRESERVED_KEYS, ensure_session_metadata, migrate_state, new_state, project_checkpoint_state


class AnyChainGraphRuntime:
    """Thin wrapper around the LangGraph product workflow graph."""

    def __init__(
        self,
        thread_id: str,
        checkpoint_path: str | Path | None = None,
        checkpointer: Any | None = None,
        session_purpose: str = "user",
    ) -> None:
        self.thread_id = thread_id
        self.session_purpose = session_purpose or "user"
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
        self.checkpointer = checkpointer or create_sqlite_checkpointer(self.checkpoint_path)
        self.graph = build_graph(self.checkpointer)

    def close(self) -> None:
        manager = getattr(self.checkpointer, "_anychain_context_manager", None)
        if manager is not None:
            manager.__exit__(None, None, None)
            delattr(self.checkpointer, "_anychain_context_manager")

    def __enter__(self) -> "AnyChainGraphRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def invoke(self, text: str, language: str = "en", context: dict[str, Any] | None = None) -> AgentGraphState:
        state = self._load_state(language=language)
        before = deepcopy(state)
        state["last_user_input"] = text
        state["language"] = language
        invocation_context: InvocationContext = {
            key: deepcopy((context or {}).get(key) or {})
            for key in _INVOCATION_CONTEXT_KEYS
        }
        # Runtime commands are invocation-scoped capabilities. Explicitly
        # clear the field for user turns so a checkpointer/runtime cannot carry
        # a prior startup command into the next graph invocation.
        invocation_context["runtime_action"] = {}
        state = ensure_session_metadata(state, self.thread_id, self.session_purpose)
        config = {"configurable": {"thread_id": self.thread_id}}
        with llm_turn_scope(load_llm_config().turn_timeout_seconds):
            ensure_turn_active()
            try:
                result = self.graph.invoke(
                    state,
                    config=config,
                    context=invocation_context,
                )
                ensure_turn_active()
                validate_state(result)
            except StateInvariantError as exc:
                candidate = dict(locals().get("result") or state)
                candidate["turn_index"] = max(
                    int(candidate.get("turn_index") or 0),
                    int(before.get("turn_index") or 0) + 1,
                )
                recovered = self._recover_invariant_failure(candidate, exc)
                self._write_turn_observation("turn_recovered", before, recovered)
                return recovered
            persisted = dict(result)
            self._write_turn_observation("turn_committed", before, persisted)
            return persisted

    def _recover_invariant_failure(self, state: AgentGraphState, exc: StateInvariantError) -> AgentGraphState:
        """Re-enter the product graph from its last valid checkpoint."""

        record = build_failure_record(
            "HARNESS_INVARIANT_FAILED",
            source="harness",
            severity="blocking",
            facts=[{"code": "HARNESS_INVARIANT_FAILED", "source": "harness", "detail": str(exc)}],
            confirmed_config=state.get("confirmed_config") or {},
        )
        try:
            return self._invoke_runtime_action(
                {
                    "type": "activate_harness_recovery",
                    "failure_record": record,
                    "confidence": "high",
                },
                language=str(state.get("language") or "en"),
                observation="",
            )
        except Exception:
            # If the graph itself cannot execute its recovery action, retain
            # only a typed quarantine state. This is the sole emergency direct
            # checkpoint path and cannot apply normal user or domain actions.
            recovered: AgentGraphState = dict(state)
            recovered["action_queue"] = []
            recovered["selected_action"] = {}
            recovered["current_action"] = {}
            recovered["pending_domain_result"] = {}
            recovered["side_effect_intent"] = {}
            recovered["side_effect_receipt"] = {}
            recovered["pending_question"] = {}
            recovered["failure_recovery"] = {
                "status": "pending",
                "record": record,
            }
            recovered["active_group"] = "failure_recovery"
            recovered["control"] = {}
            recovered["pending_question"] = (
                question_for_recovery(recovered, "failure_recovery") or {}
            )
            reset_turn_response(recovered)
            recovered = finalize_turn_response(recovered)
            validate_state(recovered)
            return self._persist_state(recovered)

    def snapshot(self) -> AgentGraphState:
        return self._load_state(language="en")

    def prepare_resume_offer(self, language: str) -> AgentGraphState:
        return self._invoke_runtime_action(
            {"type": "prepare_session_entry", "confidence": "high"},
            language=language,
            observation="startup_snapshot",
        )

    def reset(self, language: str = "en") -> AgentGraphState:
        """Reset workflow configuration without checkpointing startup read models.

        Environment discovery, framework facts, web-research availability,
        and historical-job selection are invocation context owned by terminal
        query services. The workflow keeps only its own job receipt and report
        context across this reset.
        """

        return self._invoke_runtime_action(
            {"type": "reset_session", "confidence": "high"},
            language=language,
            observation="workflow_reset",
        )

    def clear_evidence_collection(self) -> AgentGraphState:
        """Cancel terminal evidence capture without exposing arbitrary writes."""

        return self._invoke_runtime_action(
            {
                "type": "cancel_evidence_collection",
                "source_evidence": "terminal evidence collection cancellation",
                "confidence": "high",
            },
            language="en",
            observation="evidence_collection_cancelled",
        )

    def _invoke_runtime_action(
        self,
        action: Mapping[str, Any],
        *,
        language: str,
        observation: str,
    ) -> AgentGraphState:
        """Run one trusted terminal command through the product graph."""

        state = self._load_state(language=language)
        before = deepcopy(state)
        state["last_user_input"] = ""
        state["language"] = language
        state = ensure_session_metadata(
            state,
            self.thread_id,
            self.session_purpose,
        )
        config = {"configurable": {"thread_id": self.thread_id}}
        result = self.graph.invoke(
            state,
            config=config,
            context={
                "discovery": {},
                "framework_summary": {},
                "web_research": {},
                "runtime_action": deepcopy(dict(action)),
            },
        )
        validate_state(result)
        persisted = dict(result)
        if observation:
            self._write_turn_observation(observation, before, persisted)
        return persisted

    def _persist_state(self, patch: dict[str, Any]) -> AgentGraphState:
        """Internal checkpoint write used by typed runtime operations."""

        config = {"configurable": {"thread_id": self.thread_id}}
        patch = ensure_session_metadata(dict(patch), self.thread_id, self.session_purpose)
        patch = project_checkpoint_state(patch)
        self.graph.update_state(config, patch)
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        return dict(values)

    def _load_state(self, language: str) -> AgentGraphState:
        config = {"configurable": {"thread_id": self.thread_id}}
        values: dict[str, Any] = {}
        try:
            snapshot = self.graph.get_state(config)
            values = dict(getattr(snapshot, "values", None) or {})
            if values:
                migrated = migrate_state(
                    dict(values),
                    thread_id=self.thread_id,
                    language=language,
                    session_purpose=self.session_purpose,
                )
                validate_state(migrated)
                if migrated != project_checkpoint_state(values):
                    return self._persist_state(migrated)
                return migrated
        except Exception as exc:
            quarantined = new_state(self.thread_id, language=language, session_purpose=self.session_purpose)
            for key in RESET_PRESERVED_KEYS:
                if key in values:
                    quarantined[key] = values[key]  # type: ignore[literal-required]
            quarantined["checkpoint_recovery"] = {
                "status": "quarantined",
                "error_type": type(exc).__name__,
                "safe_confirmed_config": dict(values.get("confirmed_config") or {}),
            }
            quarantined["audit_events"] = list(values.get("audit_events") or []) + [
                {"event": "checkpoint_quarantined", "error_type": type(exc).__name__}
            ]
            return quarantined
        return new_state(self.thread_id, language=language, session_purpose=self.session_purpose)

    def _write_turn_observation(
        self,
        event_type: str,
        before: AgentGraphState,
        after: AgentGraphState,
    ) -> None:
        """Emit an out-of-band committed-state observation for live Chaos.

        The observer is deliberately write-only from the product process. Live
        runners must not open the active SQLite/WAL database through a second
        Docker bind-mount connection. Only hashes and non-secret control-plane
        identifiers are emitted; configuration values and evidence are absent.
        """

        destination = str(os.environ.get("ANYCHAIN_AGENT_TURN_EVENT_FILE") or "").strip()
        if not destination:
            return
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        admitted_action_types: list[str] = []
        for item in (after.get("turn_context") or {}).get("admitted_actions") or []:
            action_type = str((item or {}).get("type") or "")
            if action_type and action_type not in admitted_action_types:
                admitted_action_types.append(action_type)
        admitted_action_provenance = _admitted_action_provenance(after)
        payload = {
            "schema_version": 3,
            "event_type": event_type,
            "thread_id": self.thread_id,
            "session_purpose": self.session_purpose,
            "before_fingerprint": _state_fingerprint(before),
            "after_fingerprint": _state_fingerprint(after),
            "turn_index": int(after.get("turn_index") or 0),
            "active_group": str(after.get("active_group") or ""),
            "pending_question_id": str((after.get("pending_question") or {}).get("id") or ""),
            "pending_contract": deepcopy(after.get("pending_question") or {}),
            "revision": repository_revision(Path(__file__).resolve().parents[2]),
            "action_queue_types": [
                str((item or {}).get("action_type") or "")
                for item in after.get("action_queue") or []
            ],
            "admitted_action_types": admitted_action_types,
            "admitted_action_targets": [
                {
                    "type": str((item or {}).get("type") or ""),
                    "group": str((item or {}).get("group") or ""),
                }
                for item in (after.get("turn_context") or {}).get("admitted_actions") or []
                if str((item or {}).get("type") or "") == "change_group"
                and str((item or {}).get("group") or "")
            ],
            "admitted_action_provenance": admitted_action_provenance,
            "turn_receipt_summary": _turn_receipt_summary(after),
            "pending_transition": _pending_transition_summary(
                before,
                after,
                admitted_action_provenance,
            ),
            "render_manifest": _render_manifest(after),
            "execution_receipt_summary": _execution_receipt_summary(after),
            "control_receipts": _control_receipts(after),
            "state_diff_hashes": _state_diff_hashes(before, after),
            "material_state_diff_hashes": _material_state_diff_hashes(
                before,
                after,
            ),
            "after_value_hashes": _leaf_value_hashes(after),
            "next_result": _next_result(after),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _admitted_action_provenance(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = {
        "type",
        "action_id",
        "source",
        "owner",
        "effect",
        "group",
        "argument_names",
        "arguments_hash",
        "argument_value_hashes",
        "source_unit_ids",
        "source_hash",
        "admission_receipt_id",
    }
    rows: list[dict[str, Any]] = []
    for raw in (state.get("turn_context") or {}).get("admitted_actions") or ():
        if not isinstance(raw, Mapping):
            continue
        row = {
            str(key): deepcopy(value)
            for key, value in raw.items()
            if str(key) in allowed
        }
        if str(row.get("type") or ""):
            rows.append(row)
    return rows


def _turn_receipt_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(state.get("turn_receipt") or {})
    if not receipt:
        return {}
    semantic_units = []
    for raw in receipt.get("semantic_units") or ():
        if not isinstance(raw, Mapping):
            continue
        source_text = str(raw.get("source_text") or "")
        semantic_units.append({
            "unit_id": str(raw.get("unit_id") or ""),
            "clause_id": str(raw.get("clause_id") or ""),
            "disposition": str(raw.get("disposition") or ""),
            "start": raw.get("start"),
            "end": raw.get("end"),
            "action_indexes": [
                int(index)
                for index in raw.get("action_indexes") or ()
                if isinstance(index, int) and not isinstance(index, bool)
            ],
            "source_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        })
    pending_verdicts = []
    for raw in receipt.get("pending_candidate_verdicts") or ():
        if not isinstance(raw, Mapping):
            continue
        pending_verdicts.append({
            "action_index": raw.get("action_index"),
            "admission_action_id": str(raw.get("admission_action_id") or ""),
            "candidate_value_hash": _canonical_hash(raw.get("candidate_value")),
            "semantic_unit_ids": [
                str(item) for item in raw.get("semantic_unit_ids") or () if str(item)
            ],
            "verdict": str(raw.get("verdict") or ""),
        })
    omission_checks = [
        {
            "unit_id": str(raw.get("unit_id") or ""),
            "disposition": str(raw.get("disposition") or ""),
            "action_indexes": [
                int(index)
                for index in raw.get("action_indexes") or ()
                if isinstance(index, int) and not isinstance(index, bool)
            ],
            "verdict": str(raw.get("verdict") or ""),
        }
        for raw in receipt.get("sibling_omission_checks") or ()
        if isinstance(raw, Mapping)
    ]
    return {
        "turn_id": str(receipt.get("turn_id") or ""),
        "input_hash": str(receipt.get("input_hash") or ""),
        "language": str(receipt.get("language") or ""),
        "input_shape": str(receipt.get("input_shape") or ""),
        "status": str(receipt.get("status") or ""),
        "semantic_units": semantic_units,
        "admitted_action_ids": [
            str(item) for item in receipt.get("admitted_action_ids") or () if str(item)
        ],
        "semantic_order": [
            str(item) for item in receipt.get("semantic_order") or () if str(item)
        ],
        "execution_order": [
            str(item) for item in receipt.get("execution_order") or () if str(item)
        ],
        "owner_bindings": {
            str(key): str(value)
            for key, value in dict(receipt.get("owner_bindings") or {}).items()
        },
        "action_unit_bindings": {
            str(key): [str(item) for item in value if str(item)]
            for key, value in dict(receipt.get("action_unit_bindings") or {}).items()
            if isinstance(value, (list, tuple))
        },
        "unit_action_bindings": {
            str(key): [str(item) for item in value if str(item)]
            for key, value in dict(receipt.get("unit_action_bindings") or {}).items()
            if isinstance(value, (list, tuple))
        },
        "pending_candidate_verdicts": pending_verdicts,
        "sibling_omission_checks": omission_checks,
        "unresolved_unit_ids": [
            str(item) for item in receipt.get("unresolved_units") or () if str(item)
        ],
        "pending_before_hash": _canonical_hash(receipt.get("pending_before") or {}),
        "pending_after_hash": _canonical_hash(receipt.get("pending_after") or {}),
        "response_count": int(receipt.get("response_count") or 0),
    }


def _pending_transition_summary(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    pending_before = dict(before.get("pending_question") or {})
    pending_after = dict(after.get("pending_question") or {})
    before_hash = _canonical_hash(pending_before)
    after_hash = _canonical_hash(pending_after)
    if before_hash == after_hash:
        transition = "preserved"
    elif pending_before and pending_after:
        transition = "replaced"
    elif pending_before:
        transition = "consumed"
    elif pending_after:
        transition = "created"
    else:
        transition = "absent"
    return {
        "transition": transition,
        "before_id": str(pending_before.get("id") or ""),
        "before_group": str(pending_before.get("group") or ""),
        "before_hash": before_hash,
        "after_id": str(pending_after.get("id") or ""),
        "after_group": str(pending_after.get("group") or ""),
        "after_hash": after_hash,
        "consumer_action_ids": [
            str(item.get("action_id") or "")
            for item in actions
            if str(item.get("action_id") or "")
        ],
    }


def _render_manifest(state: Mapping[str, Any]) -> dict[str, Any]:
    fragments = [str(item) for item in state.get("visible_response") or ()]
    return {
        "language": str(state.get("language") or ""),
        "fragment_count": len(fragments),
        "fragment_hashes": [
            hashlib.sha256(item.encode("utf-8")).hexdigest() for item in fragments
        ],
        "pending_contract_hash": _canonical_hash(
            state.get("pending_question") or {}
        ),
        "result": _next_result(state),
    }


def _execution_receipt_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    from agent.runners.job_manager import verify_job_receipt

    intent = dict(state.get("side_effect_intent") or {})
    receipt = dict(state.get("side_effect_receipt") or {})
    job = dict(state.get("job") or {})
    execution_receipts = (
        dict(job.get("execution_receipts") or {})
        if isinstance(job.get("execution_receipts"), Mapping)
        else {}
    )
    base_submission = (
        dict(execution_receipts.get("submission") or {})
        if isinstance(execution_receipts.get("submission"), Mapping)
        else {}
    )
    submission_attempt = (
        dict(execution_receipts.get("submission_attempt") or {})
        if isinstance(execution_receipts.get("submission_attempt"), Mapping)
        else {}
    )
    submission = submission_attempt or base_submission
    last_read = (
        dict(execution_receipts.get("last_read") or {})
        if isinstance(execution_receipts.get("last_read"), Mapping)
        else {}
    )
    job_id = str(job.get("job_id") or "")
    if (
        not verify_job_receipt(base_submission)
        or base_submission.get("job_id") != job_id
    ):
        base_submission = {}
    if (
        not verify_job_receipt(submission)
        or submission.get("job_id") != job_id
    ):
        submission = {}
    if (
        not verify_job_receipt(last_read)
        or last_read.get("job_id") != job_id
        or last_read.get("observed_status") != job.get("status")
        or (
            last_read.get("submission_receipt_id")
            and last_read.get("submission_receipt_id")
            != base_submission.get("receipt_id")
        )
    ):
        last_read = {}
    return {
        "intent_id": str(intent.get("intent_id") or ""),
        "intent_action_type": str(intent.get("operation") or ""),
        "intent_idempotency_key": str(intent.get("idempotency_key") or ""),
        "receipt_id": str(receipt.get("receipt_id") or ""),
        "receipt_status": str(receipt.get("status") or ""),
        "receipt_idempotency_key": str(receipt.get("idempotency_key") or ""),
        "job_id": job_id,
        "job_status": str(job.get("status") or ""),
        "manager_submission_receipt_id": str(
            submission.get("receipt_id") or ""
        ),
        "manager_submission_disposition": str(
            submission.get("disposition") or ""
        ),
        "manager_matching_job_count": (
            int(submission.get("matching_job_count"))
            if isinstance(submission.get("matching_job_count"), int)
            and not isinstance(submission.get("matching_job_count"), bool)
            else 0
        ),
        "manager_execution_key_hash": str(
            submission.get("execution_key_hash") or ""
        ),
        "manager_read_receipt_id": str(last_read.get("receipt_id") or ""),
        "manager_observed_status": str(
            last_read.get("observed_status") or ""
        ),
    }


def _control_receipts(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for raw in (state.get("turn_context") or {}).get("control_receipts") or ():
        if not isinstance(raw, Mapping):
            continue
        receipt = deepcopy(dict(raw))
        if (
            str(receipt.get("receipt_type") or "")
            and len(str(receipt.get("receipt_id") or "")) == 64
        ):
            receipts.append(receipt)
    return receipts


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_fingerprint(state: AgentGraphState) -> str:
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_diff_hashes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    before_hashes = _leaf_value_hashes(before)
    after_hashes = _leaf_value_hashes(after)
    return {
        path: {
            "before": before_hashes.get(path, ""),
            "after": after_hashes.get(path, ""),
        }
        for path in sorted(set(before_hashes) | set(after_hashes))
        if before_hashes.get(path) != after_hashes.get(path)
    }


_MATERIAL_STATE_ROOTS = frozenset({
    "active_group",
    "confirmed_config",
    "inferred_config",
    "group_states",
    "group_history",
    "invalidated_groups",
    "interruption_stack",
    "chain_identity",
    "target_mode",
    "workflow_mode",
    "rpc_mode",
    "workload",
    "custom_rpc",
    "qps_profile",
    "endpoint_evidence",
    "fixture_evidence",
    "sync_observe",
    "observability",
    "advanced_tuning",
    "preflight",
    "smoke",
    "final_benchmark",
    "job",
    "failure_recovery",
    "evidence_collection",
    "evidence_buffer",
    "report_analysis",
})


def _material_state_diff_hashes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    return {
        path: hashes
        for path, hashes in _state_diff_hashes(before, after).items()
        if path.split(".", 1)[0] in _MATERIAL_STATE_ROOTS
    }


def _leaf_value_hashes(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, str]:
    if isinstance(value, Mapping) and value:
        output: dict[str, str] = {}
        for key in sorted(value, key=str):
            output.update(_leaf_value_hashes(value[key], (*prefix, str(key))))
        return output
    path = ".".join(prefix)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {path: hashlib.sha256(encoded).hexdigest()} if path else {}


def _next_result(state: Mapping[str, Any]) -> dict[str, Any]:
    pending = dict(state.get("pending_question") or {})
    responses = list(state.get("visible_response") or [])
    turn_local_result_count = int(
        (state.get("turn_context") or {}).get("turn_local_result_count") or 0
    )
    if pending and turn_local_result_count:
        return {
            "kind": "result",
            "response_count": turn_local_result_count,
            "pending_overlay": True,
            "question_id": str(pending.get("id") or ""),
            "group": str(pending.get("group") or ""),
        }
    if pending:
        return {
            "kind": "question",
            "question_id": str(pending.get("id") or ""),
            "group": str(pending.get("group") or ""),
        }
    job = dict(state.get("job") or {})
    return {
        "kind": "result",
        "response_count": len(responses),
        "job_id": str(job.get("job_id") or ""),
    }


class InvocationContext(TypedDict, total=False):
    discovery: dict[str, Any]
    framework_summary: dict[str, Any]
    web_research: dict[str, Any]
    runtime_action: dict[str, Any]


_INVOCATION_CONTEXT_KEYS = (
    "discovery",
    "framework_summary",
    "web_research",
)


def _contextual_step(
    step: Callable[[AgentGraphState], AgentGraphState],
) -> Callable[[AgentGraphState, Runtime[InvocationContext]], AgentGraphState]:
    """Expose invocation read models to one node without checkpointing them."""

    def run(
        state: AgentGraphState,
        runtime: Runtime[InvocationContext],
    ) -> AgentGraphState:
        contextual: AgentGraphState = deepcopy(state)
        invocation_context = runtime.context or {}
        for key in _INVOCATION_CONTEXT_KEYS:
            contextual[key] = deepcopy(
                invocation_context.get(key)
                or state.get(key)
                or {}
            )  # type: ignore[literal-required]
        result = step(contextual)
        for key in _INVOCATION_CONTEXT_KEYS:
            result[key] = {}  # type: ignore[literal-required]
        return result

    return run


def _prepare_graph_step(
    state: AgentGraphState,
    runtime: Runtime[InvocationContext],
) -> AgentGraphState:
    """Prepare a turn and bind a trusted runtime command without checkpointing it."""

    contextual: AgentGraphState = deepcopy(state)
    invocation_context = runtime.context or {}
    for key in _INVOCATION_CONTEXT_KEYS:
        contextual[key] = deepcopy(invocation_context.get(key) or {})
    result = prepare_turn_step(contextual)
    runtime_action = (
        dict(invocation_context.get("runtime_action") or {})
        if not str((result.get("turn_context") or {}).get("text") or "").strip()
        else {}
    )
    if (
        runtime_action
        and str((result.get("control") or {}).get("phase") or "")
        == "adjudicate"
    ):
        result.setdefault("turn_context", {})["runtime_action"] = runtime_action
    for key in _INVOCATION_CONTEXT_KEYS:
        result[key] = {}  # type: ignore[literal-required]
    return result


def _owner_step(
    owner: str,
) -> Callable[[AgentGraphState, Runtime[InvocationContext]], AgentGraphState]:
    def run(
        state: AgentGraphState,
        runtime: Runtime[InvocationContext],
    ) -> AgentGraphState:
        return _contextual_step(
            lambda contextual: execute_selected_owner_step(
                contextual,
                expected_owner=owner,
            )
        )(state, runtime)

    return run


def build_graph(checkpointer: Any) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentGraphState, context_schema=InvocationContext)
    graph.add_node("prepare", _prepare_graph_step)
    graph.add_node("adjudicate", _contextual_step(adjudicate_turn_step))
    graph.add_node("partition", _contextual_step(partition_turn_step))
    graph.add_node("compile_owner", _contextual_step(compile_owner_turn_step))
    graph.add_node("review_plan", _contextual_step(review_plan_turn_step))
    graph.add_node("admit", _contextual_step(admit_turn_step))
    graph.add_node("select_action", _contextual_step(select_action_step))
    for owner in _OWNER_NODE:
        graph.add_node(f"owner_{owner}", _owner_step(owner))
    graph.add_node("commit_action", _contextual_step(commit_selected_action_step))
    graph.add_node("invoke_effect", _contextual_step(mark_side_effect_invoking_step))
    graph.add_node("perform_effect", _contextual_step(invoke_idempotent_side_effect_step))
    graph.add_node("commit_receipt", _contextual_step(commit_side_effect_receipt_step))
    graph.add_node("fallback", _contextual_step(fallback_turn_step))
    graph.add_node("compose", _contextual_step(compose_turn_step))
    graph.add_node("validate", _validate_graph_state)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("adjudicate", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("partition", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("compile_owner", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("review_plan", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("admit", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges(
        "select_action",
        _selected_owner_node,
        {
            **{node: node for node in _OWNER_NODE.values()},
            "select_action": "select_action",
            "fallback": "fallback",
            "compose": "compose",
        },
    )
    for node in _OWNER_NODE.values():
        graph.add_conditional_edges(node, _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("commit_action", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("invoke_effect", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("perform_effect", _next_phase, _PHASE_NODE)
    graph.add_conditional_edges("commit_receipt", _next_phase, _PHASE_NODE)
    graph.add_edge("fallback", "compose")
    graph.add_edge("compose", "validate")
    graph.add_edge("validate", END)
    return graph.compile(checkpointer=checkpointer)


_PHASE_NODE = {
    "adjudicate": "adjudicate",
    "plan": "partition",
    "compile_owner": "compile_owner",
    "review_plan": "review_plan",
    "admit": "admit",
    "execute": "select_action",
    "commit": "commit_action",
    "invoke_effect": "invoke_effect",
    "perform_effect": "perform_effect",
    "commit_receipt": "commit_receipt",
    "fallback": "fallback",
    "compose": "compose",
}


_OWNER_NODE = {
    owner: f"owner_{owner}"
    for owner in (
        "orientation",
        "environment",
        "chain_rpc",
        "performance",
        "sync_observe",
        "execution",
        "recovery",
        "analysis",
        "coordinator",
    )
}


def _selected_owner_node(state: AgentGraphState) -> str:
    phase = str((state.get("control") or {}).get("phase") or "")
    if phase != "route_owner":
        return _PHASE_NODE.get(phase, "compose")
    owner = str((state.get("control") or {}).get("selected_owner") or "")
    if owner not in _OWNER_NODE:
        raise RuntimeError(f"invalid selected action owner: {owner or '<missing>'}")
    return _OWNER_NODE[owner]


def _next_phase(state: AgentGraphState) -> str:
    phase = str((state.get("control") or {}).get("phase") or "compose")
    if phase not in _PHASE_NODE:
        raise RuntimeError(f"invalid Harness graph phase: {phase}")
    return phase


def _validate_graph_state(state: AgentGraphState) -> AgentGraphState:
    validate_state(state)
    return state
