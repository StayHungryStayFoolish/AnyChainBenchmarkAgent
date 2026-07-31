"""LangGraph runtime wrapper for AnyChain Agent Harness."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, TypedDict

from langgraph.runtime import Runtime

from .checkpoints import create_sqlite_checkpointer, default_checkpoint_path
from .contracts import TurnReceipt, turn_receipt_to_dict
from .control_receipts import validate_persisted_domain_control_receipt
from ..llm.config import load_llm_config
from ..llm.types import (
    LLMProviderError,
    LLMTurnCancelledError,
    LLMTurnTimeoutError,
    ensure_turn_active,
    llm_turn_scope,
    provider_attempt_evidence,
)
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
from .input_identity import user_input_hash
from .domains.recovery import question_for_recovery
from .response import finalize_turn_response, reset_turn_response
from .questions import (
    answer_fits_pending,
    coerce_pending_answer,
    exact_option_answer,
    typed_pending_value_candidates,
)
from .runtime_identity import repository_revision
from .secret_refs import (
    all_owned_secret_references,
    discard_secret_reference,
    discard_draft_secret_references,
    discard_state_secret_references,
    project_secret_input,
    reconcile_state_secret_bindings,
    release_unowned_input_secret_bindings,
    SecretRegistryTransaction,
    secret_registry_transaction,
)
from .semantic_drafts import mark_semantic_plan_draft_stale
from .terminal_protocol import (
    append_jsonl_record,
    claim_jsonl_authority,
    quarantine_jsonl_file,
    read_jsonl_records,
)
from .state import (
    AgentGraphState,
    RESET_PRESERVED_KEYS,
    ensure_session_metadata,
    has_incomplete_semantic_finalization,
    migrate_state,
    new_state,
    project_checkpoint_state,
    quarantine_inflight_semantic_finalization,
)
from .turn_transactions import (
    ProductAuthorityLease,
    ProductHead,
    TerminalOutcome,
    TurnAttempt,
    TurnTransactionStore,
    product_authority_id,
)


_EXECUTION_APPROVAL_QUESTIONS = frozenset({
    "preflight_smoke_confirm",
    "real_node_smoke_confirm",
    "real_node_final_benchmark_confirm",
})


def project_turn_input(
    state: Mapping[str, Any],
    text: str,
    *,
    scope_id: str,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Project one user turn using the active signed question contract."""

    pending = dict(state.get("pending_question") or {})
    sensitive = bool(
        pending.get("sensitive_input")
        or pending.get("secret_reentry_binding")
    )
    option_matched, _option_value = exact_option_answer(text, pending)
    if not sensitive or option_matched:
        return project_secret_input(
            str(text),
            scope_id=scope_id,
            force_secret=False,
        )
    if pending.get("secret_reentry_binding"):
        return project_secret_input(
            str(text),
            scope_id=scope_id,
            force_secret=True,
        )
    if answer_fits_pending(str(text), pending):
        return project_secret_input(
            str(coerce_pending_answer(str(text), pending)),
            scope_id=scope_id,
            force_secret=True,
        )

    candidates = typed_pending_value_candidates(str(text), pending)
    if candidates:
        projected = str(text)
        bindings: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            reference, candidate_bindings = project_secret_input(
                candidate,
                scope_id=f"{scope_id}:candidate:{index}",
                force_secret=True,
            )
            projected = projected.replace(candidate, reference)
            bindings.extend(dict(item) for item in candidate_bindings)
        return projected, tuple(bindings)

    projected, bindings = project_secret_input(
        str(text),
        scope_id=scope_id,
        force_secret=False,
    )
    if bindings:
        return projected, bindings
    # Compound prose with no deterministic value candidate remains semantic
    # input. Bare scalar answers were already projected above, while explicit
    # credentials in prose/JSON were projected by the lexical secret boundary.
    return projected, ()
_EXECUTION_APPROVAL_ACTIONS = frozenset({
    "approve_preflight_smoke",
    "approve_final_benchmark",
})


class AnyChainGraphRuntime:
    """Thin wrapper around the LangGraph product workflow graph."""

    def __init__(
        self,
        thread_id: str,
        checkpoint_path: str | Path | None = None,
        checkpointer: Any | None = None,
        session_purpose: str = "user",
    ) -> None:
        self._closed = False
        self._owned_secret_references: set[str] = set()
        self.thread_id = thread_id
        self.session_purpose = session_purpose or "user"
        self.transaction_authority_id = product_authority_id(
            self.thread_id,
            self.session_purpose,
        )
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
        self.checkpointer = checkpointer or create_sqlite_checkpointer(self.checkpoint_path)
        self.graph = build_graph(self.checkpointer)
        self.turn_transactions = TurnTransactionStore(self.checkpoint_path)
        self._last_terminal_outcome: TerminalOutcome | None = None
        self.authority_lease = ProductAuthorityLease(
            self.checkpoint_path,
            self.transaction_authority_id,
        )
        try:
            self.authority_lease.acquire()
            self._recover_interrupted_attempts()
            self._recover_missing_runtime_observations()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        references = set(
            getattr(self, "_owned_secret_references", set())
        )
        graph = getattr(self, "graph", None)
        if graph is not None:
            try:
                transaction_store = getattr(self, "turn_transactions", None)
                head = (
                    transaction_store.get_product_head(
                        self.transaction_authority_id
                    )
                    if transaction_store is not None
                    else None
                )
                if head is not None:
                    snapshot = graph.get_state({
                        "configurable": {
                            "thread_id": head.checkpoint_thread_id,
                            "checkpoint_ns": "",
                            "checkpoint_id": head.checkpoint_id,
                        }
                    })
                    references.update(all_owned_secret_references(
                        dict(getattr(snapshot, "values", None) or {})
                    ))
            except BaseException as exc:
                if not references:
                    raise RuntimeError(
                        "secret cleanup could not read Product Head state"
                    ) from exc
        for reference in references:
            discard_secret_reference(reference)
        self._owned_secret_references.clear()
        lease = getattr(self, "authority_lease", None)
        if lease is not None:
            lease.close()
        manager = getattr(self.checkpointer, "_anychain_context_manager", None)
        if manager is not None:
            manager.__exit__(None, None, None)
            delattr(self.checkpointer, "_anychain_context_manager")
        self._closed = True

    def _remember_owned_secret_references(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._owned_secret_references.update(
            all_owned_secret_references(state)
        )

    @property
    def last_terminal_outcome(self) -> TerminalOutcome | None:
        return self._last_terminal_outcome

    def pending_terminal_outcomes(self) -> tuple[TerminalOutcome, ...]:
        return self.turn_transactions.list_undelivered_outcomes(
            self.transaction_authority_id
        )

    def unresolved_reconciliation(self) -> TerminalOutcome | None:
        return self.turn_transactions.get_unresolved_reconciliation(
            self.transaction_authority_id
        )

    def product_head(self) -> ProductHead | None:
        return self.turn_transactions.get_product_head(
            self.transaction_authority_id
        )

    def runtime_event_fence(self) -> tuple[int, str, str, str]:
        return self.turn_transactions.runtime_event_fence(
            self.transaction_authority_id
        )

    def _runtime_event_path(self) -> Path:
        configured = str(
            os.environ.get("ANYCHAIN_AGENT_TURN_EVENT_FILE") or ""
        ).strip()
        if configured:
            return Path(configured)
        authority_digest = hashlib.sha256(
            self.transaction_authority_id.encode("utf-8")
        ).hexdigest()[:16]
        return Path(
            f"{self.checkpoint_path}.runtime-events."
            f"{authority_digest}.jsonl"
        )

    def terminal_messages(self, outcome: TerminalOutcome) -> tuple[str, ...]:
        """Load a committed response from its exact Product Head checkpoint."""

        if outcome.logical_thread_id != self.transaction_authority_id:
            raise RuntimeError("terminal outcome belongs to another Product Head")
        if outcome.outcome != "committed":
            return ()
        snapshot = self.graph.get_state({
            "configurable": {
                "thread_id": outcome.product_checkpoint_thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": outcome.product_checkpoint_id,
            }
        })
        values = dict(getattr(snapshot, "values", None) or {})
        if _state_fingerprint(values) != outcome.product_fingerprint:
            raise RuntimeError("terminal replay checkpoint fingerprint mismatch")
        messages = tuple(str(item) for item in values.get("visible_response") or ())
        if _canonical_hash(list(messages)) != outcome.render_hash:
            raise RuntimeError("terminal replay response hash mismatch")
        return messages

    def mark_terminal_delivered(
        self,
        outcome: TerminalOutcome,
    ) -> TerminalOutcome:
        delivered = self.turn_transactions.mark_terminal_delivered(
            logical_thread_id=self.transaction_authority_id,
            event_id=outcome.event_id,
        )
        if (
            self._last_terminal_outcome is not None
            and self._last_terminal_outcome.event_id == outcome.event_id
        ):
            self._last_terminal_outcome = delivered
        return delivered

    def ensure_runtime_observation(self, outcome: TerminalOutcome) -> None:
        """Repair and verify the committed runtime event before presentation."""

        destination = self._runtime_event_path()
        if outcome.outcome != "committed":
            return
        self._recover_missing_runtime_observations()
        observed: dict[str, Mapping[str, Any]] = {}
        runtime_event_ids: set[str] = set()
        for payload in read_jsonl_records(destination):
            event_id = str(payload.get("terminal_event_id") or "")
            runtime_event_id = str(payload.get("runtime_event_id") or "")
            if not runtime_event_id or runtime_event_id in runtime_event_ids:
                raise RuntimeError(
                    "runtime observation identity is missing or duplicated"
                )
            runtime_event_ids.add(runtime_event_id)
            if event_id:
                if event_id in observed:
                    raise RuntimeError(
                        "runtime observation contains a duplicate terminal event"
                    )
                observed[event_id] = payload
        if outcome.event_id not in observed:
            raise RuntimeError(
                "committed terminal outcome lacks its runtime observation"
            )
        self._validate_runtime_observation_payload(
            observed[outcome.event_id],
            outcome,
        )

    def resolve_reconciliation(
        self,
        *,
        transaction_id: str,
        resolution: str,
        evidence_hash: str,
    ) -> None:
        self.turn_transactions.resolve_reconciliation(
            logical_thread_id=self.transaction_authority_id,
            transaction_id=transaction_id,
            resolution=resolution,
            evidence_hash=evidence_hash,
        )
        current = self._last_terminal_outcome
        if (
            current is not None
            and current.transaction_id == transaction_id
            and current.outcome == "reconciliation_required"
        ):
            self._last_terminal_outcome = None

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
        state = self._load_state_transactionally(language=language)
        with secret_registry_transaction() as transaction:
            return self._invoke_product_turn(
                text,
                language,
                context,
                state=state,
                registry_transaction=transaction,
            )

    def _invoke_product_turn(
        self,
        text: str,
        language: str,
        context: dict[str, Any] | None,
        *,
        state: AgentGraphState,
        registry_transaction: SecretRegistryTransaction,
    ) -> AgentGraphState:
        before = deepcopy(state)
        input_scope = (
            f"turn:{self.transaction_authority_id}:"
            f"{int(state.get('turn_index') or 0) + 1}"
        )
        safe_text, input_secret_bindings = project_turn_input(
            state,
            text,
            scope_id=input_scope,
        )
        state["last_user_input"] = safe_text
        state["language"] = language
        invocation_context: InvocationContext = {
            key: deepcopy((context or {}).get(key) or {})
            for key in _INVOCATION_CONTEXT_KEYS
        }
        approval_question_id = str(
            (state.get("pending_question") or {}).get("id") or ""
        )
        invocation_context["repository_revision"] = (
            repository_revision(Path(__file__).resolve().parents[2])
            if approval_question_id
            in _EXECUTION_APPROVAL_QUESTIONS
            else {}
        )
        # Runtime commands are invocation-scoped capabilities. Explicitly
        # clear the field for user turns so a checkpointer/runtime cannot carry
        # a prior startup command into the next graph invocation.
        invocation_context["runtime_action"] = {}
        invocation_context["input_secret_bindings"] = {
            "items": [dict(item) for item in input_secret_bindings],
        }
        invocation_context["submitted_input_hash"] = user_input_hash(text)
        state = ensure_session_metadata(state, self.thread_id, self.session_purpose)
        attempt = self._begin_turn_attempt(before)
        self._bind_registry_to_attempt(registry_transaction, attempt)
        invocation_context["product_head"] = _attempt_product_head_context(
            self.transaction_authority_id,
            attempt,
        )
        config = {"configurable": {"thread_id": attempt.physical_thread_id}}
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
                if self._attempt_effect_evidence(attempt)[0]:
                    release_unowned_input_secret_bindings(
                        input_secret_bindings,
                        before,
                    )
                    self._finish_failed_turn_attempt(attempt, exc)
                    raise
                try:
                    recovered = self._recover_invariant_failure(
                        before,
                        candidate,
                        exc,
                        attempt,
                        current_input=safe_text,
                        submitted_input_hash=user_input_hash(text),
                        language=language,
                        registry_transaction=registry_transaction,
                        input_secret_bindings=input_secret_bindings,
                    )
                except BaseException as recovery_exc:
                    self._terminalize_failed_attempt_once(
                        attempt,
                        recovery_exc,
                    )
                    raise
                self._remember_owned_secret_references(recovered)
                self._write_turn_observation(
                    "turn_recovered",
                    before,
                    recovered,
                    self._last_terminal_outcome,
                )
                return recovered
            except BaseException as exc:
                release_unowned_input_secret_bindings(
                    input_secret_bindings,
                    before,
                )
                self._finish_failed_turn_attempt(attempt, exc)
                raise
            persisted = dict(result)
            try:
                release_unowned_input_secret_bindings(
                    input_secret_bindings,
                    persisted,
                )
                registry_transaction.prepare()
                registry_transaction.publish_prepared()
                self._last_terminal_outcome = self._commit_turn_attempt(
                    attempt,
                    persisted,
                )
                registry_transaction.mark_product_head_committed()
                registry_transaction.finalize()
            except BaseException as exc:
                self._terminalize_failed_attempt_once(attempt, exc)
                raise
            self._remember_owned_secret_references(persisted)
            self._write_turn_observation(
                "turn_committed",
                before,
                persisted,
                self._last_terminal_outcome,
            )
            return persisted

    def _recover_invariant_failure(
        self,
        base_state: AgentGraphState,
        candidate_state: AgentGraphState,
        exc: StateInvariantError,
        attempt: TurnAttempt,
        *,
        current_input: str,
        submitted_input_hash: str,
        language: str,
        registry_transaction: SecretRegistryTransaction,
        input_secret_bindings: tuple[Mapping[str, str], ...],
    ) -> AgentGraphState:
        """Commit typed recovery on the original physical attempt."""

        record = build_failure_record(
            "HARNESS_INVARIANT_FAILED",
            source="harness",
            severity="blocking",
            facts=[{"code": "HARNESS_INVARIANT_FAILED", "source": "harness", "detail": str(exc)}],
            confirmed_config=candidate_state.get("confirmed_config") or {},
        )
        if has_incomplete_semantic_finalization(base_state):
            discard_state_secret_references(base_state)
            recovered = quarantine_inflight_semantic_finalization(
                base_state,
                thread_id=self.thread_id,
                language=str(base_state.get("language") or "en"),
                session_purpose=self.session_purpose,
                error_type="InvariantFailureDuringSemanticFinalization",
            )
        else:
            recovered = deepcopy(base_state)
        recovered["turn_index"] = max(
            int(candidate_state.get("turn_index") or 0),
            int(base_state.get("turn_index") or 0) + 1,
        )
        recovered["action_queue"] = []
        recovered["selected_action"] = {}
        recovered["current_action"] = {}
        recovered["pending_domain_result"] = {}
        recovered["side_effect_intent"] = {}
        recovered["side_effect_receipt"] = {}
        draft = dict(recovered.get("semantic_plan_draft") or {})
        if draft.get("status") in {
            "awaiting_clarification",
            "ready_for_review",
        }:
            discard_draft_secret_references(draft)
            recovered["semantic_plan_draft"] = mark_semantic_plan_draft_stale(
                draft,
                reasons=("harness_invariant_recovery",),
            )
            recovered.setdefault("audit_events", []).append({
                "event": "semantic_draft_invalidated",
                "draft_id": str(draft.get("draft_id") or ""),
                "revision": int(draft.get("revision") or 0),
                "reasons": ["harness_invariant_recovery"],
            })
        recovered["failure_recovery"] = {"status": "pending", "record": record}
        recovered["active_group"] = "failure_recovery"
        recovered["control"] = {}
        recovered["pending_question"] = (
            question_for_recovery(recovered, "failure_recovery") or {}
        )
        recovered["last_user_input"] = current_input
        input_shape = str(candidate_state.get("input_shape") or "prose")
        recovered["turn_context"] = {
            "id": int(recovered.get("turn_index") or 0),
            "kind": "invariant_recovery",
            "text": current_input,
            "input_shape": input_shape,
            "origin_group": str(base_state.get("active_group") or ""),
            "pending_snapshot": deepcopy(
                base_state.get("pending_question") or {}
            ),
            "admitted_actions": [],
        }
        recovered["turn_receipt"] = turn_receipt_to_dict(TurnReceipt(
            turn_id=(
                f"{recovered.get('thread_id') or self.thread_id}:"
                f"{int(recovered.get('turn_index') or 0)}"
            ),
            input_hash=submitted_input_hash,
            submitted_input_hash=submitted_input_hash,
            language=language,
            input_shape=input_shape,
            pending_before=deepcopy(
                base_state.get("pending_question") or {}
            ),
            pending_after=deepcopy(
                recovered.get("pending_question") or {}
            ),
            status="failed",
        ))
        reconcile_state_secret_bindings(recovered)
        reset_turn_response(recovered)
        recovered = finalize_turn_response(recovered)
        validate_state(recovered)
        config = self.graph.update_state(
            {"configurable": {"thread_id": attempt.physical_thread_id}},
            project_checkpoint_state(recovered),
            as_node="validate",
        )
        snapshot = self.graph.get_state(config)
        persisted = dict(getattr(snapshot, "values", None) or {})
        validate_state(persisted)
        release_unowned_input_secret_bindings(
            input_secret_bindings,
            persisted,
        )
        registry_transaction.prepare()
        registry_transaction.publish_prepared()
        self._last_terminal_outcome = self._commit_turn_attempt(
            attempt,
            persisted,
            snapshot=snapshot,
        )
        registry_transaction.mark_product_head_committed()
        registry_transaction.finalize()
        return persisted

    def snapshot(self) -> AgentGraphState:
        return self._load_state_transactionally(language="en")

    def _load_state_transactionally(self, language: str) -> AgentGraphState:
        """Load and migrate one Product Head in its own registry transaction."""

        with secret_registry_transaction() as transaction:
            state = self._load_state(
                language=language,
                registry_transaction=transaction,
            )
            if not transaction.committed:
                transaction.prepare()
                transaction.commit()
            return state

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

        state = self._load_state_transactionally(language=language)
        with secret_registry_transaction() as transaction:
            return self._invoke_runtime_action_transaction(
                action,
                language=language,
                observation=observation,
                state=state,
                registry_transaction=transaction,
            )

    def _invoke_runtime_action_transaction(
        self,
        action: Mapping[str, Any],
        *,
        language: str,
        observation: str,
        state: AgentGraphState,
        registry_transaction: SecretRegistryTransaction,
    ) -> AgentGraphState:
        before = deepcopy(state)
        state["last_user_input"] = ""
        state["language"] = language
        state = ensure_session_metadata(
            state,
            self.thread_id,
            self.session_purpose,
        )
        attempt = self._begin_turn_attempt(state)
        self._bind_registry_to_attempt(registry_transaction, attempt)
        config = {"configurable": {"thread_id": attempt.physical_thread_id}}
        product_head_context = _attempt_product_head_context(
            self.transaction_authority_id,
            attempt,
        )
        try:
            result = self.graph.invoke(
                state,
                config=config,
                context={
                    "discovery": {},
                    "framework_summary": {},
                    "web_research": {},
                    "runtime_action": deepcopy(dict(action)),
                    "repository_revision": {},
                    "product_head": product_head_context,
                    "submitted_input_hash": user_input_hash(""),
                },
            )
            validate_state(result)
            persisted = dict(result)
            registry_transaction.prepare()
            registry_transaction.publish_prepared()
            self._last_terminal_outcome = self._commit_turn_attempt(
                attempt,
                persisted,
            )
            registry_transaction.mark_product_head_committed()
            registry_transaction.finalize()
        except BaseException as exc:
            self._terminalize_failed_attempt_once(attempt, exc)
            raise
        self._remember_owned_secret_references(persisted)
        if observation:
            self._write_turn_observation(
                observation,
                before,
                persisted,
                self._last_terminal_outcome,
            )
        return persisted

    def _persist_state(
        self,
        patch: dict[str, Any],
        *,
        registry_transaction: SecretRegistryTransaction | None = None,
    ) -> AgentGraphState:
        """Internal checkpoint write used by typed runtime operations."""

        patch = ensure_session_metadata(dict(patch), self.thread_id, self.session_purpose)
        patch = project_checkpoint_state(patch)
        self._ensure_product_head(
            language=str(patch.get("language") or "en"),
        )
        revision = repository_revision(Path(__file__).resolve().parents[2])
        attempt = self.turn_transactions.begin_attempt(
            logical_thread_id=self.transaction_authority_id,
            origin_revision_commit=revision["commit"],
            origin_revision_worktree_hash=revision["worktree_hash"],
        )
        if registry_transaction is not None:
            self._bind_registry_to_attempt(registry_transaction, attempt)
        try:
            config = self.graph.update_state(
                {"configurable": {"thread_id": attempt.physical_thread_id}},
                patch,
                as_node="validate",
            )
            snapshot = self.graph.get_state(config)
            values = dict(getattr(snapshot, "values", None) or {})
            validate_state(values)
            if registry_transaction is not None:
                registry_transaction.prepare()
                registry_transaction.publish_prepared()
            self._last_terminal_outcome = self._commit_turn_attempt(
                attempt,
                values,
                snapshot=snapshot,
            )
            if registry_transaction is not None:
                registry_transaction.mark_product_head_committed()
                registry_transaction.finalize()
            self._remember_owned_secret_references(values)
            return values
        except BaseException as exc:
            self._terminalize_failed_attempt_once(attempt, exc)
            raise

    def _load_state(
        self,
        language: str,
        *,
        registry_transaction: SecretRegistryTransaction | None = None,
    ) -> AgentGraphState:
        values: dict[str, Any] = {}
        head = self._ensure_product_head(language=language)
        try:
            config = {
                "configurable": {
                    "thread_id": head.checkpoint_thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": head.checkpoint_id,
                }
            }
            snapshot = self.graph.get_state(config)
            values = dict(getattr(snapshot, "values", None) or {})
            if values:
                if _state_fingerprint(values) != head.state_fingerprint:
                    raise RuntimeError(
                        "product head fingerprint does not match its checkpoint"
                    )
                migrated = migrate_state(
                    dict(values),
                    thread_id=self.thread_id,
                    language=language,
                    session_purpose=self.session_purpose,
                )
                validate_state(migrated)
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
            return self._persist_state(
                quarantined,
                registry_transaction=registry_transaction,
            )
        if values:
            if migrated != project_checkpoint_state(values):
                # A Product Head write failure is not checkpoint corruption.
                # Keep it outside recovery so the registry transaction rolls
                # back every process-local mutation made during migration.
                return self._persist_state(
                    migrated,
                    registry_transaction=registry_transaction,
                )
            self._remember_owned_secret_references(migrated)
            return migrated
        return new_state(self.thread_id, language=language, session_purpose=self.session_purpose)

    def _ensure_product_head(
        self,
        *,
        language: str,
        preferred_state: AgentGraphState | None = None,
    ) -> ProductHead:
        head = self.turn_transactions.get_product_head(
            self.transaction_authority_id
        )
        if head is not None:
            return head
        snapshot = self._latest_completed_legacy_snapshot()
        values = dict(getattr(snapshot, "values", None) or {}) if snapshot else {}
        if preferred_state is not None:
            values = dict(preferred_state)
            snapshot = None
        if not values:
            values = new_state(
                self.thread_id,
                language=language,
                session_purpose=self.session_purpose,
            )
        values = ensure_session_metadata(
            values,
            self.thread_id,
            self.session_purpose,
            touch=False,
        )
        values = project_checkpoint_state(values)
        if preferred_state is not None or snapshot is None:
            validate_state(values)
        if snapshot is None:
            bootstrap_thread = f"head:{self.transaction_authority_id}"
            bootstrap_config = self.graph.update_state(
                {"configurable": {"thread_id": bootstrap_thread}},
                values,
                as_node="validate",
            )
            snapshot = self.graph.get_state(bootstrap_config)
        checkpoint_id = str(
            (getattr(snapshot, "config", {}) or {})
            .get("configurable", {})
            .get("checkpoint_id", "")
        )
        checkpoint_thread_id = str(
            (getattr(snapshot, "config", {}) or {})
            .get("configurable", {})
            .get("thread_id", "")
        )
        if not checkpoint_id or not checkpoint_thread_id:
            raise RuntimeError("product head bootstrap checkpoint has no identity")
        return self.turn_transactions.bootstrap_product_head(
            logical_thread_id=self.transaction_authority_id,
            checkpoint_thread_id=checkpoint_thread_id,
            checkpoint_id=checkpoint_id,
            state_fingerprint=_state_fingerprint(values),
        )

    def _latest_completed_legacy_snapshot(self) -> Any | None:
        config = {"configurable": {"thread_id": self.thread_id}}
        for snapshot in self.graph.get_state_history(config):
            metadata = dict(getattr(snapshot, "metadata", None) or {})
            if (
                not tuple(getattr(snapshot, "next", ()) or ())
                or str(metadata.get("source") or "") == "update"
            ):
                values = dict(getattr(snapshot, "values", None) or {})
                if values:
                    return snapshot
        return None

    def _begin_turn_attempt(
        self,
        base_state: AgentGraphState,
    ) -> TurnAttempt:
        self._ensure_product_head(
            language=str(base_state.get("language") or "en"),
        )
        revision = repository_revision(Path(__file__).resolve().parents[2])
        attempt = self.turn_transactions.begin_attempt(
            logical_thread_id=self.transaction_authority_id,
            origin_revision_commit=revision["commit"],
            origin_revision_worktree_hash=revision["worktree_hash"],
        )
        try:
            self.graph.update_state(
                {"configurable": {"thread_id": attempt.physical_thread_id}},
                project_checkpoint_state(base_state),
                as_node="validate",
            )
        except BaseException as exc:
            self._terminalize_failed_attempt_once(attempt, exc)
            raise
        return attempt

    def _bind_registry_to_attempt(
        self,
        transaction: SecretRegistryTransaction,
        attempt: TurnAttempt,
    ) -> None:
        transaction.bind_external_commit_probe(
            lambda: (
                (
                    outcome := self.turn_transactions.get_terminal_outcome(
                        attempt.transaction_id
                    )
                )
                is not None
                and outcome.outcome == "committed"
            )
        )

    def _commit_turn_attempt(
        self,
        attempt: TurnAttempt,
        state: AgentGraphState,
        *,
        snapshot: Any | None = None,
    ) -> TerminalOutcome:
        config = {"configurable": {"thread_id": attempt.physical_thread_id}}
        expected_state = project_checkpoint_state(state)
        validate_state(expected_state)
        expected_fingerprint = _state_fingerprint(expected_state)
        if snapshot is None:
            committed_config = self.graph.update_state(
                config,
                expected_state,
                as_node="validate",
            )
            snapshot = self.graph.get_state(committed_config)
        persisted = dict(getattr(snapshot, "values", None) or {})
        if (
            not persisted
            or _state_fingerprint(persisted) != expected_fingerprint
        ):
            raise RuntimeError(
                "turn attempt checkpoint does not match its committed state"
            )
        if tuple(getattr(snapshot, "next", ()) or ()):
            raise RuntimeError("completed turn attempt did not reach graph END")
        checkpoint_id = str(
            (getattr(snapshot, "config", {}) or {})
            .get("configurable", {})
            .get("checkpoint_id", "")
        )
        if not checkpoint_id:
            raise RuntimeError("completed turn attempt has no checkpoint identity")
        return self.turn_transactions.commit_attempt(
            logical_thread_id=self.transaction_authority_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id=checkpoint_id,
            attempt_fingerprint=expected_fingerprint,
            render_hash=_canonical_hash(state.get("visible_response") or []),
        )

    def _terminalize_failed_attempt_once(
        self,
        attempt: TurnAttempt,
        exc: BaseException,
    ) -> TerminalOutcome:
        existing = self.turn_transactions.get_terminal_outcome(
            attempt.transaction_id
        )
        if existing is not None:
            self._last_terminal_outcome = existing
            return existing
        return self._finish_failed_turn_attempt(attempt, exc)

    def _finish_failed_turn_attempt(
        self,
        attempt: TurnAttempt,
        exc: BaseException,
    ) -> TerminalOutcome:
        provider_evidence = [
            dict(item) for item in provider_attempt_evidence()
        ]
        evidence_checkpoint_id = ""
        evidence_checkpoint_hash = ""
        evidence_persistence_error = ""
        if provider_evidence:
            try:
                (
                    evidence_checkpoint_id,
                    evidence_checkpoint_hash,
                ) = (
                    self._persist_failed_provider_attempt_evidence(
                        attempt,
                        provider_evidence,
                    )
                )
            except Exception as persistence_exc:
                evidence_persistence_error = type(persistence_exc).__name__
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "provider attempt evidence could not be persisted"
                    )
        diagnostic_hash = _canonical_hash({
            "type": type(exc).__name__,
            "provider": str(getattr(exc, "provider", "") or ""),
            "model": str(getattr(exc, "model", "") or ""),
            "stage": str(getattr(exc, "stage", "") or ""),
            "category": str(getattr(exc, "category", "") or ""),
            "status_code": int(getattr(exc, "status_code", 0) or 0),
            "attempt_count": int(
                getattr(exc, "attempt_count", 0) or 0
            ),
            "retry_reasons": list(
                getattr(exc, "retry_reasons", ()) or ()
            ),
            "retry_exhausted": bool(
                getattr(exc, "retry_exhausted", False)
            ),
            "last_finish_reason": str(
                getattr(exc, "last_finish_reason", "") or ""
            ),
            "provider_attempt_evidence": provider_evidence,
            "provider_evidence_checkpoint_id": evidence_checkpoint_id,
            "provider_evidence_checkpoint_hash": evidence_checkpoint_hash,
            "provider_evidence_persistence_error": evidence_persistence_error,
        })
        (
            requires_reconciliation,
            checkpoint_id,
            checkpoint_fingerprint,
        ) = self._attempt_effect_evidence(attempt)
        failure_category = (
            "provider_evidence_persistence_failure"
            if evidence_persistence_error
            else _failure_category(exc)
        )

        if requires_reconciliation:
            checkpoint_arguments = (
                {
                    "attempt_checkpoint_id": checkpoint_id,
                    "attempt_fingerprint": checkpoint_fingerprint,
                }
                if checkpoint_id and checkpoint_fingerprint
                else {}
            )
            self._last_terminal_outcome = (
                self.turn_transactions.require_reconciliation(
                    logical_thread_id=self.transaction_authority_id,
                    transaction_id=attempt.transaction_id,
                    physical_thread_id=attempt.physical_thread_id,
                    diagnostic_hash=diagnostic_hash,
                    failure_category=failure_category,
                    **checkpoint_arguments,
                )
            )
        else:
            self._last_terminal_outcome = self.turn_transactions.abort_attempt(
                logical_thread_id=self.transaction_authority_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                diagnostic_hash=diagnostic_hash,
                attempt_checkpoint_id=(
                    evidence_checkpoint_id or None
                ),
                attempt_fingerprint=(
                    evidence_checkpoint_hash or None
                ),
                failure_category=failure_category,
            )
        return self._last_terminal_outcome

    def _persist_failed_provider_attempt_evidence(
        self,
        attempt: TurnAttempt,
        evidence: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Retain failed-turn provider evidence on the physical attempt only."""

        config = {"configurable": {"thread_id": attempt.physical_thread_id}}
        snapshot = self.graph.get_state(config)
        state = dict(getattr(snapshot, "values", {}) or {})
        if not state:
            raise RuntimeError(
                "failed turn has no physical attempt checkpoint"
            )
        state.setdefault("turn_context", {})[
            "provider_attempt_evidence"
        ] = deepcopy(evidence)
        validate_state(state)
        updated_config = self.graph.update_state(
            config,
            project_checkpoint_state(state),
            as_node="provider_audit",
        )
        persisted = self.graph.get_state(config)
        values = dict(getattr(persisted, "values", {}) or {})
        observed = list(
            (values.get("turn_context") or {}).get(
                "provider_attempt_evidence"
            )
            or ()
        )
        if observed != evidence:
            raise RuntimeError(
                "failed-turn provider evidence checkpoint mismatch"
            )
        checkpoint_id = str(
            (updated_config or {}).get("configurable", {}).get(
                "checkpoint_id",
                "",
            )
        )
        if not checkpoint_id:
            raise RuntimeError(
                "failed-turn provider evidence has no checkpoint identity"
            )
        return checkpoint_id, _state_fingerprint(values)

    def _recover_interrupted_attempts(self) -> None:
        """Close attempts left by a process that no longer owns the lease."""

        for attempt in self.turn_transactions.list_attempts(
            self.transaction_authority_id
        ):
            if attempt.status != "active":
                continue
            reconcile, checkpoint_id, checkpoint_fingerprint = (
                self._attempt_effect_evidence(attempt)
            )
            diagnostic_hash = _canonical_hash({
                "type": "InterruptedTurnAttempt",
                "stage": "runtime_restart_recovery",
            })
            if reconcile:
                checkpoint_arguments = (
                    {
                        "attempt_checkpoint_id": checkpoint_id,
                        "attempt_fingerprint": checkpoint_fingerprint,
                    }
                    if checkpoint_id and checkpoint_fingerprint
                    else {}
                )
                self._last_terminal_outcome = (
                    self.turn_transactions.require_reconciliation(
                        logical_thread_id=self.transaction_authority_id,
                        transaction_id=attempt.transaction_id,
                        physical_thread_id=attempt.physical_thread_id,
                        diagnostic_hash=diagnostic_hash,
                        failure_category="runtime_restart_interrupted",
                        **checkpoint_arguments,
                    )
                )
            else:
                diagnostic_checkpoint_arguments = (
                    self._interrupted_provider_evidence_identity(
                        attempt,
                        checkpoint_id,
                        checkpoint_fingerprint,
                    )
                )
                self._last_terminal_outcome = self.turn_transactions.abort_attempt(
                    logical_thread_id=self.transaction_authority_id,
                    transaction_id=attempt.transaction_id,
                    physical_thread_id=attempt.physical_thread_id,
                    diagnostic_hash=diagnostic_hash,
                    **diagnostic_checkpoint_arguments,
                    failure_category="runtime_restart_interrupted",
                )

    def _interrupted_provider_evidence_identity(
        self,
        attempt: TurnAttempt,
        checkpoint_id: str,
        checkpoint_fingerprint: str,
    ) -> dict[str, str]:
        if not checkpoint_id or not checkpoint_fingerprint:
            return {}
        snapshot = self.graph.get_state({
            "configurable": {
                "thread_id": attempt.physical_thread_id,
                "checkpoint_id": checkpoint_id,
            }
        })
        values = dict(getattr(snapshot, "values", {}) or {})
        turn_context = values.get("turn_context") or {}
        if "provider_attempt_evidence" not in turn_context:
            return {}
        validate_state(values)
        evidence = turn_context.get("provider_attempt_evidence")
        if not evidence:
            raise RuntimeError(
                "interrupted provider evidence checkpoint is empty"
            )
        if _state_fingerprint(values) != checkpoint_fingerprint:
            raise RuntimeError(
                "interrupted provider evidence checkpoint fingerprint mismatch"
            )
        return {
            "attempt_checkpoint_id": checkpoint_id,
            "attempt_fingerprint": checkpoint_fingerprint,
        }

    def _recover_missing_runtime_observations(self) -> None:
        """Rebuild committed runtime evidence after a post-commit crash."""

        path = self._runtime_event_path()
        observed_terminal_events: dict[str, Mapping[str, Any]] = {}
        runtime_event_ids: set[str] = set()
        claim_jsonl_authority(
            path,
            product_authority_id=self.transaction_authority_id,
        )
        records = read_jsonl_records(path) if path.exists() else ()
        if any(
            int(payload.get("schema_version") or 0) < 6
            for payload in records
        ):
            self.turn_transactions.requeue_runtime_events_after_legacy_log_quarantine(
                self.transaction_authority_id
            )
            quarantine_jsonl_file(
                path,
                reason="legacy-runtime-schema",
            )
            records = ()
        observed_sequences: list[int] = []
        for payload in records:
            if int(payload.get("schema_version") or 0) > 6:
                raise RuntimeError(
                    "runtime observation schema is newer than this runtime"
                )
            if (
                str(payload.get("product_authority_id") or "")
                != self.transaction_authority_id
            ):
                raise RuntimeError(
                    "runtime observation belongs to another Product Head authority"
                )
            sequence = payload.get("runtime_event_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise RuntimeError(
                    "runtime observation sequence is invalid"
                )
            observed_sequences.append(sequence)
            runtime_event_id = str(payload.get("runtime_event_id") or "")
            if not runtime_event_id or runtime_event_id in runtime_event_ids:
                raise RuntimeError(
                    "runtime observation identity is missing or duplicated"
                )
            runtime_event_ids.add(runtime_event_id)
            terminal_event_id = str(payload.get("terminal_event_id") or "")
            if terminal_event_id:
                if terminal_event_id in observed_terminal_events:
                    raise RuntimeError(
                        "runtime observation contains a duplicate terminal event"
                    )
                observed_terminal_events[terminal_event_id] = payload
        if observed_sequences != sorted(observed_sequences):
            raise RuntimeError(
                "runtime observations are not ordered by Product Head revision"
            )
        committed_outcomes = sorted(
            (
                outcome
                for outcome in self.turn_transactions.list_terminal_outcomes(
                    self.transaction_authority_id
                )
                if outcome.outcome == "committed"
            ),
            key=lambda outcome: outcome.runtime_event_sequence,
        )
        expected_sequences = list(range(1, len(committed_outcomes) + 1))
        if [
            outcome.runtime_event_sequence
            for outcome in committed_outcomes
        ] != expected_sequences:
            raise RuntimeError(
                "committed runtime-event sequence is not contiguous"
            )
        for outcome in committed_outcomes:
            existing = observed_terminal_events.get(outcome.event_id)
            if existing is not None:
                self._validate_runtime_observation_payload(existing, outcome)
                payload_hash = str(
                    existing.get("runtime_event_payload_hash") or ""
                )
                self.turn_transactions.mark_runtime_event_published(
                    logical_thread_id=self.transaction_authority_id,
                    event_id=outcome.event_id,
                    runtime_event_id=str(existing["runtime_event_id"]),
                    payload_hash=payload_hash,
                )
                continue
            before = self._checkpoint_state(
                outcome.base_checkpoint_thread_id,
                outcome.base_checkpoint_id,
                outcome.base_fingerprint,
            )
            after = self._checkpoint_state(
                outcome.product_checkpoint_thread_id,
                outcome.product_checkpoint_id,
                outcome.product_fingerprint,
            )
            self._write_turn_observation(
                "turn_recovered",
                before,
                after,
                outcome,
            )

    def _validate_runtime_observation_payload(
        self,
        payload: Mapping[str, Any],
        outcome: TerminalOutcome,
    ) -> None:
        expected_revision = {
            "commit": outcome.origin_revision_commit,
            "worktree_hash": outcome.origin_revision_worktree_hash,
        }
        if (
            int(payload.get("schema_version") or 0) != 6
            or str(payload.get("event_type") or "") != "turn_committed"
            or not str(payload.get("observation") or "")
            or str(payload.get("terminal_event_id") or "") != outcome.event_id
            or str(payload.get("transaction_id") or "")
            != outcome.transaction_id
            or str(payload.get("terminal_outcome") or "") != outcome.outcome
            or not str(payload.get("runtime_event_id") or "")
            or str(payload.get("before_fingerprint") or "")
            != outcome.base_fingerprint
            or str(payload.get("after_fingerprint") or "")
            != outcome.product_fingerprint
            or not isinstance(payload.get("base_revision"), int)
            or int(payload["base_revision"]) != outcome.base_revision
            or str(payload.get("base_checkpoint_thread_id") or "")
            != outcome.base_checkpoint_thread_id
            or str(payload.get("base_checkpoint_id") or "")
            != outcome.base_checkpoint_id
            or not isinstance(payload.get("product_revision"), int)
            or int(payload["product_revision"]) != outcome.product_revision
            or str(payload.get("product_checkpoint_thread_id") or "")
            != outcome.product_checkpoint_thread_id
            or str(payload.get("product_checkpoint_id") or "")
            != outcome.product_checkpoint_id
            or str(payload.get("render_hash") or "") != outcome.render_hash
            or dict(payload.get("revision") or {}) != expected_revision
            or str(payload.get("thread_id") or "") != self.thread_id
            or str(payload.get("session_purpose") or "")
            != self.session_purpose
            or str(payload.get("product_authority_id") or "")
            != self.transaction_authority_id
            or str(payload.get("physical_thread_id") or "")
            != outcome.physical_thread_id
            or str(payload.get("attempt_checkpoint_id") or "")
            != str(outcome.attempt_checkpoint_id or "")
        ):
            raise RuntimeError(
                "runtime observation does not match committed terminal facts"
            )
        runtime_event_id = str(payload.get("runtime_event_id") or "")
        if outcome.runtime_event_id and runtime_event_id != outcome.runtime_event_id:
            raise RuntimeError(
                "runtime observation does not match its reserved event identity"
            )
        unsigned = dict(payload)
        payload_hash = str(unsigned.pop("runtime_event_payload_hash", "") or "")
        if not payload_hash or _canonical_hash(unsigned) != payload_hash:
            raise RuntimeError("runtime observation payload hash is invalid")
        if (
            outcome.runtime_event_status == "published"
            and outcome.runtime_event_payload_hash != payload_hash
        ):
            raise RuntimeError(
                "runtime observation differs from its published receipt"
            )

    def _checkpoint_state(
        self,
        checkpoint_thread_id: str,
        checkpoint_id: str,
        expected_fingerprint: str,
    ) -> AgentGraphState:
        snapshot = self.graph.get_state({
            "configurable": {
                "thread_id": checkpoint_thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        })
        values = dict(getattr(snapshot, "values", None) or {})
        if _state_fingerprint(values) != expected_fingerprint:
            raise RuntimeError(
                "runtime observation recovery checkpoint fingerprint mismatch"
            )
        return values

    def _attempt_effect_evidence(
        self,
        attempt: TurnAttempt,
    ) -> tuple[bool, str, str]:
        try:
            baseline = self._checkpoint_state(
                attempt.base_checkpoint_thread_id,
                attempt.base_checkpoint_id,
                attempt.base_fingerprint,
            )
            snapshot = self.graph.get_state({
                "configurable": {"thread_id": attempt.physical_thread_id}
            })
            values = dict(getattr(snapshot, "values", None) or {})
            checkpoint_id = str(
                (getattr(snapshot, "config", {}) or {})
                .get("configurable", {})
                .get("checkpoint_id", "")
            )
            if not checkpoint_id or not values:
                raise RuntimeError(
                    "turn attempt has no verifiable checkpoint identity"
                )
            fingerprint = _state_fingerprint(values)
            baseline_intent = dict(
                baseline.get("side_effect_intent") or {}
            )
            baseline_receipt = dict(
                baseline.get("side_effect_receipt") or {}
            )
            current_intent = dict(
                values.get("side_effect_intent") or {}
            )
            current_receipt = dict(
                values.get("side_effect_receipt") or {}
            )
            new_invocation = (
                str(current_intent.get("status") or "") == "invoking"
                and current_intent != baseline_intent
            )
            new_receipt = (
                bool(current_receipt)
                and current_receipt != baseline_receipt
            )
            return (
                new_invocation or new_receipt,
                checkpoint_id,
                fingerprint,
            )
        except Exception:
            return True, "", ""

    def _write_turn_observation(
        self,
        observation: str,
        before: AgentGraphState,
        after: AgentGraphState,
        outcome: TerminalOutcome | None,
    ) -> TerminalOutcome:
        """Emit an out-of-band committed-state observation for live Chaos.

        The observer is deliberately write-only from the product process. Live
        runners must not open the active SQLite/WAL database through a second
        Docker bind-mount connection. Only hashes and non-secret control-plane
        identifiers are emitted; configuration values and evidence are absent.
        """

        destination = self._runtime_event_path()
        if outcome is None or outcome.outcome != "committed":
            raise RuntimeError(
                "runtime observation requires its committed terminal outcome"
            )
        path = destination
        path.parent.mkdir(parents=True, exist_ok=True)
        admitted_action_types: list[str] = []
        for item in (after.get("turn_context") or {}).get("admitted_actions") or []:
            action_type = str((item or {}).get("type") or "")
            if action_type and action_type not in admitted_action_types:
                admitted_action_types.append(action_type)
        admitted_action_provenance = _admitted_action_provenance(after)
        if (
            _state_fingerprint(before) != outcome.base_fingerprint
            or _state_fingerprint(after) != outcome.product_fingerprint
        ):
            raise RuntimeError(
                "runtime observation state does not match terminal outcome identity"
            )
        runtime_event_id = outcome.runtime_event_id or str(uuid.uuid4())
        payload = {
            "schema_version": 6,
            "event_type": "turn_committed",
            "observation": observation,
            "runtime_event_id": runtime_event_id,
            "runtime_event_sequence": outcome.runtime_event_sequence,
            "terminal_event_id": outcome.event_id,
            "transaction_id": outcome.transaction_id,
            "terminal_outcome": outcome.outcome,
            "render_hash": outcome.render_hash,
            "base_revision": outcome.base_revision,
            "base_checkpoint_thread_id": outcome.base_checkpoint_thread_id,
            "base_checkpoint_id": outcome.base_checkpoint_id,
            "product_revision": outcome.product_revision,
            "product_checkpoint_thread_id": (
                outcome.product_checkpoint_thread_id
            ),
            "product_checkpoint_id": outcome.product_checkpoint_id,
            "thread_id": self.thread_id,
            "session_purpose": self.session_purpose,
            "product_authority_id": self.transaction_authority_id,
            "physical_thread_id": outcome.physical_thread_id,
            "attempt_checkpoint_id": str(
                outcome.attempt_checkpoint_id or ""
            ),
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
                _pending_consumer_action_ids(
                    after,
                    admitted_action_provenance,
                ),
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
        payload["runtime_event_payload_hash"] = _canonical_hash(payload)
        append_jsonl_record(path, payload)
        published = self.turn_transactions.mark_runtime_event_published(
            logical_thread_id=self.transaction_authority_id,
            event_id=outcome.event_id,
            runtime_event_id=runtime_event_id,
            payload_hash=str(payload["runtime_event_payload_hash"]),
        )
        if (
            self._last_terminal_outcome is not None
            and self._last_terminal_outcome.event_id == outcome.event_id
        ):
            self._last_terminal_outcome = published
        return published


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
        "submitted_input_hash": str(
            receipt.get("submitted_input_hash") or ""
        ),
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
    consumer_action_ids: list[str],
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
        "consumer_action_ids": list(consumer_action_ids),
    }


def _pending_consumer_action_ids(
    state: Mapping[str, Any],
    admitted_actions: list[dict[str, Any]],
) -> list[str]:
    """Return admitted actions whose domain commit changed pending state."""

    admitted_ids = {
        str(item.get("action_id") or "")
        for item in admitted_actions
        if str(item.get("action_id") or "")
    }
    turn_index = int(state.get("turn_index") or 0)
    consumed: list[str] = []
    for receipt in _control_receipts(state):
        if receipt.get("receipt_type") != "domain_commit":
            continue
        valid, _reason = validate_persisted_domain_control_receipt(
            receipt,
            turn_index=turn_index,
        )
        if (
            not valid
            or receipt.get("pending_before_hash")
            == receipt.get("pending_after_hash")
        ):
            continue
        for action_id in receipt.get("consumed_action_ids") or ():
            normalized = str(action_id or "")
            if (
                normalized
                and normalized in admitted_ids
                and normalized not in consumed
            ):
                consumed.append(normalized)
    return consumed


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
    from agent.harness.control_receipts import (
        execution_intent_projection,
        execution_side_effect_projection,
    )

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
        or base_submission.get("receipt_type") != "job_submission"
        or base_submission.get("job_id") != job_id
    ):
        base_submission = {}
    if (
        not verify_job_receipt(submission)
        or submission.get("receipt_type") != "job_submission"
        or submission.get("job_id") != job_id
    ):
        submission = {}
    if (
        not verify_job_receipt(last_read)
        or last_read.get("receipt_type") != "job_read"
        or last_read.get("job_id") != job_id
        or last_read.get("observed_status")
        != last_read.get("persisted_status")
        or last_read.get("observed_status") != job.get("status")
        or (
            last_read.get("submission_receipt_id")
            and last_read.get("submission_receipt_id")
            != base_submission.get("receipt_id")
        )
    ):
        last_read = {}
    return {
        "side_effect_intent_projection": (
            execution_intent_projection(intent) if intent else {}
        ),
        "side_effect_receipt_projection": (
            execution_side_effect_projection(receipt) if receipt else {}
        ),
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
        "manager_submission_receipt": deepcopy(submission),
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
        "manager_read_receipt": deepcopy(last_read),
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


def _failure_category(exc: BaseException) -> str:
    if isinstance(exc, LLMTurnCancelledError):
        return "cancelled"
    if isinstance(exc, LLMTurnTimeoutError):
        return "timeout"
    if isinstance(exc, LLMProviderError):
        return "provider_failure"
    if isinstance(exc, StateInvariantError):
        return "state_invariant_failure"
    return "unexpected_failure"


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
    repository_revision: dict[str, str]
    product_head: dict[str, Any]
    input_secret_bindings: dict[str, Any]
    submitted_input_hash: str


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
    receipt = dict(result.get("turn_receipt") or {})
    if receipt:
        receipt["submitted_input_hash"] = str(
            invocation_context.get("submitted_input_hash")
            or receipt.get("input_hash")
            or ""
        )
        result["turn_receipt"] = receipt
    revision = dict(invocation_context.get("repository_revision") or {})
    if revision:
        result.setdefault("turn_context", {})["repository_revision"] = revision
    product_head = dict(invocation_context.get("product_head") or {})
    if product_head:
        result.setdefault("turn_context", {})["product_head"] = product_head
    input_secret_bindings = [
        dict(item)
        for item in (
            invocation_context.get("input_secret_bindings") or {}
        ).get("items") or ()
        if isinstance(item, Mapping)
    ]
    if input_secret_bindings:
        result.setdefault("turn_context", {})[
            "input_secret_bindings"
        ] = input_secret_bindings
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


def _attempt_product_head_context(
    product_authority_id: str,
    attempt: TurnAttempt,
) -> dict[str, Any]:
    """Project the immutable Product Head from which one turn branches."""

    return {
        "product_authority_id": product_authority_id,
        "revision": int(attempt.base_revision),
        "checkpoint_thread_id": attempt.base_checkpoint_thread_id,
        "checkpoint_id": attempt.base_checkpoint_id,
        "state_fingerprint": attempt.base_fingerprint,
    }


def _owner_step(
    owner: str,
) -> Callable[[AgentGraphState, Runtime[InvocationContext]], AgentGraphState]:
    def run(
        state: AgentGraphState,
        runtime: Runtime[InvocationContext],
    ) -> AgentGraphState:
        contextual = deepcopy(state)
        action_type = str(
            (contextual.get("selected_action") or {}).get("action_type")
            or (contextual.get("selected_action") or {}).get("type")
            or ""
        )
        if owner == "execution" and action_type in _EXECUTION_APPROVAL_ACTIONS:
            expected = dict(
                (contextual.get("turn_context") or {}).get(
                    "repository_revision"
                )
                or {}
            )
            observed = repository_revision(Path(__file__).resolve().parents[2])
            if not expected or observed != expected:
                raise StateInvariantError(
                    "repository revision changed before execution approval"
                )
        return _contextual_step(
            lambda contextual: execute_selected_owner_step(
                contextual,
                expected_owner=owner,
            )
        )(state, runtime)

    return run


def _commit_action_step(
    state: AgentGraphState,
    runtime: Runtime[InvocationContext],
) -> AgentGraphState:
    """Commit an execution approval only if its source revision stayed stable."""

    selected = dict(state.get("selected_action") or {})
    action_type = str(
        selected.get("action_type") or selected.get("type") or ""
    )
    if action_type in _EXECUTION_APPROVAL_ACTIONS:
        expected = dict(
            (state.get("turn_context") or {}).get("repository_revision")
            or {}
        )
        observed = repository_revision(Path(__file__).resolve().parents[2])
        if not expected or observed != expected:
            raise StateInvariantError(
                "repository revision changed during execution approval"
            )
    return _contextual_step(commit_selected_action_step)(state, runtime)


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
    graph.add_node("commit_action", _commit_action_step)
    graph.add_node("invoke_effect", _contextual_step(mark_side_effect_invoking_step))
    graph.add_node("perform_effect", _contextual_step(invoke_idempotent_side_effect_step))
    graph.add_node("commit_receipt", _contextual_step(commit_side_effect_receipt_step))
    graph.add_node("fallback", _contextual_step(fallback_turn_step))
    graph.add_node("compose", _contextual_step(compose_turn_step))
    graph.add_node("provider_audit", _provider_audit_graph_step)
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
    graph.add_edge("compose", "provider_audit")
    graph.add_edge("provider_audit", "validate")
    graph.add_edge("validate", END)
    return graph.compile(checkpointer=checkpointer)


def _provider_audit_graph_step(
    state: AgentGraphState,
) -> AgentGraphState:
    """Persist secret-free provider attempt evidence for the completed turn."""

    output: AgentGraphState = deepcopy(state)
    evidence = provider_attempt_evidence()
    if evidence:
        output.setdefault("turn_context", {})[
            "provider_attempt_evidence"
        ] = [dict(item) for item in evidence]
    return output


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
