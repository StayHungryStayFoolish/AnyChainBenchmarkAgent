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
    commit_selected_action_step,
    commit_side_effect_receipt_step,
    fallback_turn_step,
    plan_turn_step,
    prepare_turn_step,
    invoke_idempotent_side_effect_step,
    mark_side_effect_invoking_step,
    select_action_step,
    execute_selected_owner_step,
)
from .failures import build_failure_record
from .invariants import StateInvariantError, validate_state
from .domains.recovery import question_for_recovery
from .questions import render_question
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
            recovered["visible_response"] = [
                render_question(
                    recovered["pending_question"],
                    recovered.get("language", "en"),
                )
            ]
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
        payload = {
            "schema_version": 2,
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
            "state_diff_hashes": _state_diff_hashes(before, after),
            "after_value_hashes": _leaf_value_hashes(after),
            "next_result": _next_result(after),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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
    runtime_action = dict(invocation_context.get("runtime_action") or {})
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
    graph.add_node("plan", _contextual_step(plan_turn_step))
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
    graph.add_conditional_edges("plan", _next_phase, _PHASE_NODE)
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
    "plan": "plan",
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
