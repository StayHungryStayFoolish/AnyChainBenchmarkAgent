#!/usr/bin/env python3
"""Product terminal for AnyChain Benchmark Agent.

The terminal owns stable input/output, startup diagnostics, dependency
installation consent, exact shell commands, and LangGraph event rendering.
Benchmark workflow orchestration belongs to the LangGraph Harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]

from ..diagnostics.adk_status import adk_status
from ..diagnostics.doctor import run_doctor
from ..harness.graph import AnyChainGraphRuntime
from ..harness.checkpoints import default_checkpoint_path
from ..harness.invariants import StateInvariantError
from ..harness.input_identity import canonical_user_input
from ..harness.runtime_identity import repository_revision
from ..harness.terminal_protocol import (
    StartupFailureCategory,
    append_terminal_detour_projection,
    append_terminal_session_event,
    append_terminal_outcome_projection,
    build_terminal_detour_projection,
    build_terminal_session_event,
    build_terminal_outcome_projection,
)
from ..harness.turn_transactions import (
    ReconciliationRequiredError,
    TerminalDetour,
    TurnTransactionStore,
    TurnTransactionConflictError,
    product_authority_id,
    state_fingerprint,
)
from ..knowledge.framework_capabilities import load_framework_capabilities
from ..knowledge.framework_context import load_framework_context
from ..llm.config import load_llm_config
from ..llm.providers import probe_provider_readiness, provider_runtime_errors
from ..llm.search_grounding import web_research_status
from ..llm.types import (
    LLMProviderError,
    LLMTurnCancelledError,
    LLMTurnTimeoutError,
    llm_turn_scope,
)
from ..runners.job_manager import list_jobs
from ..utils.redaction import redact
from .io import OutputOnlyIO, TerminalIO
from .job_commands import JobCommandHandler
from .language import detect_language, t
from .startup_state import load_startup_state

DEFAULT_AGENT_SESSION_ID = "default"


@dataclass
class TerminalSession:
    """Terminal shell state, not benchmark workflow state."""

    language: str = "en"
    current_question_id: str = ""
    latest_job_id: str = ""
    discovery: dict[str, Any] = field(default_factory=dict)
    framework_summary: dict[str, Any] = field(default_factory=dict)
    pending_missing_dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TerminalSession":
        question_id = str(payload.get("current_question_id") or "")
        if question_id not in {"install_agent_runtime", "install_dependencies"}:
            question_id = ""
        dependencies = payload.get("pending_missing_dependencies")
        return cls(
            language=str(payload.get("language") or "en"),
            current_question_id=question_id,
            pending_missing_dependencies=(
                [str(item) for item in dependencies if str(item)]
                if isinstance(dependencies, list)
                else []
            ),
        )

    def persisted_shell_state(self) -> dict[str, Any]:
        """Return only terminal UI and installation-consent state."""

        question_id = self.current_question_id
        if question_id not in {"install_agent_runtime", "install_dependencies"}:
            question_id = ""
        return {
            "language": self.language,
            "current_question_id": question_id,
            "pending_missing_dependencies": list(self.pending_missing_dependencies),
        }


class TerminalSessionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or ".agent/terminal/session.json")

    def load(self) -> TerminalSession:
        try:
            return TerminalSession.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return TerminalSession()

    def save(self, state: TerminalSession) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state.persisted_shell_state(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = TerminalSessionStore(args.state_file)
    state = store.load()
    if args.language:
        state.language = args.language
    elif args.prompt:
        state.language = detect_language(args.prompt[0], state.language)
    io = OutputOnlyIO() if args.prompt else TerminalIO()
    app = AnyChainTerminal(
        state=state,
        store=store,
        io=io,
        session_id=args.session_id,
        checkpoint_path=args.checkpoint_path,
        fresh_session=args.fresh_session,
        session_purpose=args.session_purpose,
    )
    if args.prompt:
        try:
            app.startup()
            for prompt in args.prompt:
                app.handle_user_text(prompt)
                store.save(state)
                if app._requested_exit_code is not None:
                    return app._requested_exit_code
            return 0
        finally:
            app.close()
    return app.run()


class AnyChainTerminal:
    def __init__(
        self,
        state: TerminalSession | None = None,
        store: TerminalSessionStore | None = None,
        io: TerminalIO | OutputOnlyIO | None = None,
        session_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        fresh_session: bool = False,
        session_purpose: str = "user",
    ) -> None:
        self.state = state or TerminalSession()
        self.store = store or TerminalSessionStore()
        self.io = io or TerminalIO()
        self.session_id = session_id or DEFAULT_AGENT_SESSION_ID
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.fresh_session = bool(fresh_session)
        self.session_purpose = session_purpose or "user"
        self._harness: AnyChainGraphRuntime | None = None
        self._startup_state: dict[str, Any] = {}
        self._llm_config = load_llm_config()
        self._web_research_status: dict[str, Any] = {}
        self._llm_runtime_available = False
        self._llm_unavailable_reason = ""
        self._llm_readiness_error: LLMProviderError | None = None
        self._job_commands = JobCommandHandler(self.state)
        self._turn_active = False
        self._process_instance_id = str(uuid.uuid4())
        self._startup_failure_category = ""
        self._startup_replayed_projection_ids: list[str] = []
        self._detour_store_cache: TurnTransactionStore | None = None
        self._requested_exit_code: int | None = None
        self._active_terminal_detour_kind = ""

    def run(self) -> int:
        try:
            with _terminal_sigint_scope(self):
                self.startup()
                while True:
                    try:
                        text = self.io.input(self.state.language).strip()
                    except KeyboardInterrupt:
                        if self._record_terminal_termination(
                            command_name="ctrl_c_exit",
                            input_text="<CTRL_C>",
                            message_key="ctrl_c_exit",
                            termination_reason="outer_ctrl_c",
                            exit_code=130,
                        ):
                            return 130
                        return 130
                    except EOFError:
                        if self._record_terminal_termination(
                            command_name="eof_exit",
                            input_text="<EOF>",
                            message_key="bye",
                            termination_reason="eof",
                            exit_code=0,
                        ):
                            return 0
                        return 1

                    if not text:
                        continue
                    self.handle_user_text(text)
                    self.store.save(self.state)
                    if self._requested_exit_code is not None:
                        return self._requested_exit_code
        finally:
            self.close()

    def close(self) -> None:
        harness = self._harness
        self._harness = None
        close = getattr(harness, "close", None)
        if callable(close):
            close()

    def startup(self) -> None:
        begin_session = getattr(self.io, "begin_session", None)
        if callable(begin_session):
            begin_session()
        begin_frame = getattr(self.io, "begin_frame", None)
        if callable(begin_frame):
            begin_frame()
        self._startup_failure_category = ""
        self._startup_replayed_projection_ids = []
        self._llm_config = load_llm_config()
        self._startup_state = load_startup_state()
        latest_job = self._startup_state.get("latest_job") or {}
        self.state.latest_job_id = latest_job.get("job_id", "") if latest_job else ""

        self.io.agent(self.state.language, t(self.state.language, "welcome"))
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "mode",
                provider=self._llm_config.provider,
                model=self._llm_config.model,
                auth_mode=self._llm_config.auth_mode,
            ),
        )
        self._web_research_status = web_research_status(self._llm_config).as_dict()
        self.io.agent(
            self.state.language,
            t(self.state.language, "web_research", status=self._web_research_status.get("reason", "unknown")),
        )
        errors = self._llm_config.validate()
        if errors:
            self.io.agent(self.state.language, t(self.state.language, "llm_config_warning", errors="; ".join(errors)))

        status = adk_status().as_dict()
        self.io.agent(self.state.language, t(self.state.language, "adk", status=status.get("reason", "unknown")))

        self._load_framework_context()
        self._startup_doctor()

        if latest_job:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "job_found",
                    job_id=latest_job.get("job_id", ""),
                    status=latest_job.get("status", "unknown"),
                ),
            )
            next_actions = self._startup_state.get("next_actions") or []
            if next_actions:
                self.io.agent(self.state.language, t(self.state.language, "job_next_actions", actions=", ".join(next_actions)))
        else:
            self.io.agent(self.state.language, t(self.state.language, "job_none"))

        runtime_errors = provider_runtime_errors(self._llm_config)
        self._llm_runtime_available = not runtime_errors
        if runtime_errors:
            self._startup_failure_category = (
                StartupFailureCategory.PROVIDER_RUNTIME_UNAVAILABLE.value
            )
            self._llm_readiness_error = None
            self._llm_unavailable_reason = "; ".join(runtime_errors)
            self.state.current_question_id = "install_agent_runtime"
            self.state.pending_missing_dependencies = list(runtime_errors)
            self.io.agent(
                self.state.language,
                t(self.state.language, "agent_runtime_offer", missing="; ".join(runtime_errors)),
            )
        else:
            # `_startup_doctor` may already have set an "install_dependencies"
            # offer (missing vegeta etc.). Do not clobber it here, or the "[Y/n]"
            # prompt becomes dead — the user's "y" would never reach
            # `_install_dependencies` and would fall through to the harness.
            deps_offer_pending = self.state.current_question_id == "install_dependencies"
            if not deps_offer_pending:
                self.state.current_question_id = ""
                self.state.pending_missing_dependencies = []
            if self._establish_provider_readiness():
                try:
                    self._ensure_harness()
                except TurnTransactionConflictError:
                    self._harness = None
                    self._llm_runtime_available = False
                    self._llm_readiness_error = None
                    self._llm_unavailable_reason = "Agent session authority is already owned"
                    self.io.agent(
                        self.state.language,
                        t(self.state.language, "authority_lease_conflict"),
                    )
                    self._publish_startup_event()
                    return
                except Exception as exc:
                    _debug_exception("harness_bootstrap", exc)
                    self._harness = None
                    self._llm_runtime_available = False
                    self._llm_readiness_error = None
                    self._llm_unavailable_reason = "agent workflow runtime unavailable"
                    self.io.agent(self.state.language, t(self.state.language, "harness_runtime_error"))
                    self._startup_failure_category = (
                        StartupFailureCategory.HARNESS_RUNTIME_UNAVAILABLE.value
                    )
                    self._publish_startup_event()
                    return
                self._deliver_pending_terminal_outcomes()
                self._recover_and_deliver_terminal_detours()
                reconciliation = self._ensure_harness().unresolved_reconciliation()
                if reconciliation is not None:
                    self._emit_reconciliation_barrier(reconciliation.transaction_id)
                elif self.fresh_session:
                    self._ensure_harness().reset(language=self.state.language)
                    self._deliver_latest_terminal_outcome()
                elif not deps_offer_pending:
                    self._offer_harness_resume_if_needed()
            else:
                self._startup_failure_category = (
                    StartupFailureCategory.PROVIDER_READINESS_FAILED.value
                )
        self.io.agent(self.state.language, t(self.state.language, "help"))
        self._publish_startup_event()

    def handle_user_text(self, text: str) -> None:
        begin_frame = getattr(self.io, "begin_frame", None)
        if callable(begin_frame):
            begin_frame()
        self.state.language = detect_language(text, self.state.language)
        stripped = text.strip()
        lowered = stripped.lower()
        if _is_shell_command(stripped, lowered, {"exit", "quit", "q"}):
            self._record_terminal_termination(
                command_name="exit",
                input_text=stripped,
                message_key="bye",
                termination_reason="exit",
                exit_code=0,
            )
            return
        if _is_shell_command(stripped, lowered, {"help", "?", "帮助", "？"}):
            self._run_terminal_detour(
                command_name="help",
                effect_class="read_only",
                producer=lambda: (t(self.state.language, "help"),),
                input_text=stripped,
            )
            return
        if _is_shell_command(stripped, lowered, {"doctor", "环境检查"}):
            self._run_terminal_detour(
                command_name="doctor",
                effect_class="observation_refresh",
                producer=self._doctor_messages,
                input_text=stripped,
            )
            return
        job_plan = self._job_commands.plan_command(stripped, lowered)
        if job_plan is not None:
            if job_plan.command_name == "follow":
                self._run_streaming_terminal_detour(
                    job_id=job_plan.job_id,
                    input_text=stripped,
                )
                return
            self._run_terminal_detour(
                command_name=job_plan.command_name,
                effect_class=job_plan.effect_class,
                producer=lambda: self._job_commands.execute(job_plan),
                input_text=stripped,
            )
            return
        if lowered.startswith("reconcile ") or lowered.startswith("对账 "):
            self._handle_reconciliation_command(stripped)
            return
        if self._handle_pending_confirmation(lowered):
            return
        if not self._llm_runtime_available:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "llm_provider_unavailable",
                    reason=self._llm_unavailable_reason
                    or "provider readiness has not been established",
                ),
            )
            return

        try:
            self._turn_active = True
            with _interactive_turn_scope(self._llm_config.turn_timeout_seconds):
                self.io.agent(self.state.language, t(self.state.language, "thinking"))
                graph_state = self._ensure_harness().invoke(
                    text,
                    language=self.state.language,
                    context={
                        "discovery": self.state.discovery,
                        "framework_summary": self.state.framework_summary,
                        "web_research": self._web_research_status,
                    },
                )
        except (LLMTurnCancelledError, KeyboardInterrupt):
            self._deliver_latest_terminal_outcome(
                fallback_key="turn_cancelled",
            )
            return
        except LLMTurnTimeoutError as exc:
            self._deliver_latest_terminal_outcome(
                fallback_message=t(
                    self.state.language,
                    "turn_timeout",
                    timeout=self._llm_config.turn_timeout_seconds,
                    provider=exc.provider or self._llm_config.provider,
                    model=exc.model or self._llm_config.model,
                )
            )
            return
        except ReconciliationRequiredError as exc:
            self._emit_reconciliation_barrier(exc.transaction_id)
            return
        except Exception as exc:
            _debug_exception("harness_turn", exc)
            self._deliver_latest_terminal_outcome(
                fallback_message=_adk_error_message(self.state.language, exc),
            )
            return
        finally:
            self._turn_active = False

        if not self._deliver_latest_terminal_outcome():
            messages = graph_state.get("visible_response") or []
            if not messages:
                self.io.agent(self.state.language, t(self.state.language, "unknown"))
                return
            for message in messages:
                self.io.agent(self.state.language, str(message))

        # A turn may have submitted a new job (fake-node smoke, benchmark). Refresh
        # the latest-job hint from disk so "analyze the latest job" and the injected
        # context point at the just-submitted job, not the startup-detected one.
        try:
            recent = list_jobs(limit=1)
            if recent:
                self.state.latest_job_id = str(recent[0].get("job_id") or "") or self.state.latest_job_id
        except Exception:
            pass

    def _deliver_pending_terminal_outcomes(self) -> None:
        harness = self._ensure_harness()
        for outcome in harness.pending_terminal_outcomes():
            begin_frame = getattr(self.io, "begin_frame", None)
            if callable(begin_frame):
                begin_frame()
            self._deliver_terminal_outcome(
                outcome,
                delivery_phase="startup_replay",
            )

    def _detour_store(self) -> TurnTransactionStore:
        if self._harness is not None:
            return self._harness.turn_transactions
        if self._detour_store_cache is None:
            checkpoint = self.checkpoint_path or default_checkpoint_path()
            self._detour_store_cache = TurnTransactionStore(checkpoint)
        return self._detour_store_cache

    def _detour_authority_id(self) -> str:
        if self._harness is not None:
            return self._harness.transaction_authority_id
        return product_authority_id(self.session_id, self.session_purpose)

    def _shell_state_hash(self) -> str:
        payload = {
            "persisted": self.state.persisted_shell_state(),
            "latest_job_id": self.state.latest_job_id,
            "discovery": self.state.discovery,
            "framework_summary": self.state.framework_summary,
        }
        return state_fingerprint(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def _run_terminal_detour(
        self,
        *,
        command_name: str,
        effect_class: str,
        producer: Any,
        input_text: str = "",
    ) -> TerminalDetour:
        revision = repository_revision(REPO_ROOT)
        store = self._detour_store()
        detour = store.prepare_terminal_detour(
            logical_thread_id=self._detour_authority_id(),
            process_instance_id=self._process_instance_id,
            session_id=self.session_id,
            session_purpose=self.session_purpose,
            command_name=command_name,
            input_hash=state_fingerprint(
                canonical_user_input(input_text or command_name).encode("utf-8")
            ),
            effect_class=effect_class,  # type: ignore[arg-type]
            shell_state_before_hash=self._shell_state_hash(),
            origin_revision_commit=revision["commit"],
            origin_revision_worktree_hash=revision["worktree_hash"],
        )
        self._active_terminal_detour_kind = effect_class
        try:
            raw_messages = tuple(str(item) for item in producer())
            messages = tuple(str(redact(message)) for message in raw_messages)
            completed = store.complete_terminal_detour(
                detour_id=detour.detour_id,
                shell_state_after_hash=self._shell_state_hash(),
                response_messages=messages,
            )
        except BaseException as exc:
            diagnostic_hash = state_fingerprint(
                json.dumps(
                    {
                        "type": type(exc).__name__,
                        "command_name": command_name,
                        "effect_class": effect_class,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            if effect_class == "local_external_effect":
                store.require_terminal_detour_reconciliation(
                    detour_id=detour.detour_id,
                    diagnostic_hash=diagnostic_hash,
                )
            else:
                completed = store.complete_terminal_detour(
                    detour_id=detour.detour_id,
                    shell_state_after_hash=self._shell_state_hash(),
                    response_messages=(
                        t(
                            self.state.language,
                            (
                                "terminal_detour_interrupted"
                                if isinstance(exc, KeyboardInterrupt)
                                else "harness_runtime_error"
                            ),
                        ),
                    ),
                    effect_status="failed",
                    diagnostic_hash=diagnostic_hash,
                )
                self._deliver_terminal_detour(completed)
            if (
                isinstance(exc, KeyboardInterrupt)
                and effect_class != "local_external_effect"
            ):
                return completed
            raise
        finally:
            self._active_terminal_detour_kind = ""
        self._deliver_terminal_detour(completed)
        return completed

    def _run_streaming_terminal_detour(
        self,
        *,
        job_id: str,
        input_text: str,
    ) -> TerminalDetour:
        """Stream one job log under durable, non-workflow detour authority."""

        revision = repository_revision(REPO_ROOT)
        store = self._detour_store()
        detour = store.prepare_terminal_detour(
            logical_thread_id=self._detour_authority_id(),
            process_instance_id=self._process_instance_id,
            session_id=self.session_id,
            session_purpose=self.session_purpose,
            command_name="follow",
            input_hash=state_fingerprint(
                canonical_user_input(input_text).encode("utf-8")
            ),
            effect_class="streaming_observation",
            shell_state_before_hash=self._shell_state_hash(),
            origin_revision_commit=revision["commit"],
            origin_revision_worktree_hash=revision["worktree_hash"],
        )
        messages: list[str] = []

        def publish(message: str) -> None:
            rendered = str(redact(message))
            store.record_terminal_detour_stream_chunk(
                detour_id=detour.detour_id,
                chunk_message=rendered,
            )
            messages.append(rendered)
            self.io.agent(self.state.language, rendered)

        effect_status = "succeeded"
        diagnostic_hash: str | None = None
        stop_reason = "error"
        terminal_message = ""
        self._active_terminal_detour_kind = "streaming_observation"
        try:
            for event in self._job_commands.iter_follow_events(job_id):
                if event.terminal:
                    terminal_message = str(redact(event.message))
                    stop_reason = str(event.stop_reason or "error")
                    if event.failed:
                        effect_status = "failed"
                        diagnostic_hash = state_fingerprint(
                            f"follow-{stop_reason}".encode("utf-8")
                        )
                    break
                publish(event.message)
        except KeyboardInterrupt:
            effect_status = "failed"
            diagnostic_hash = state_fingerprint(b"follow-cancelled")
            stop_reason = "cancelled"
            terminal_message = str(
                redact(
                    t(
                        self.state.language,
                        "follow_stopped",
                        path=job_id or "<unknown>",
                    )
                )
            )
        except Exception as exc:
            effect_status = "failed"
            stop_reason = "error"
            diagnostic_hash = state_fingerprint(
                json.dumps(
                    {
                        "type": type(exc).__name__,
                        "command_name": "follow",
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            terminal_message = str(
                redact(t(self.state.language, "harness_runtime_error"))
            )
        finally:
            self._active_terminal_detour_kind = ""

        if not terminal_message:
            effect_status = "failed"
            stop_reason = "error"
            diagnostic_hash = state_fingerprint(b"follow-missing-terminal-event")
            terminal_message = str(
                redact(t(self.state.language, "harness_runtime_error"))
            )
        messages.append(terminal_message)
        completed = store.complete_terminal_detour(
            detour_id=detour.detour_id,
            shell_state_after_hash=self._shell_state_hash(),
            response_messages=tuple(messages),
            effect_status=effect_status,  # type: ignore[arg-type]
            diagnostic_hash=diagnostic_hash,
            stream_stop_reason=stop_reason,  # type: ignore[arg-type]
        )
        self.io.agent(self.state.language, terminal_message)
        self._deliver_terminal_detour(completed, emit_messages=False)
        return completed

    def _record_terminal_termination(
        self,
        *,
        command_name: str,
        input_text: str,
        message_key: str,
        termination_reason: str,
        exit_code: int,
    ) -> bool:
        """Durably publish one terminal-owned termination before exiting."""

        begin_frame = getattr(self.io, "begin_frame", None)
        if callable(begin_frame):
            begin_frame()
        try:
            revision = repository_revision(REPO_ROOT)
            store = self._detour_store()
            detour = store.prepare_terminal_detour(
                logical_thread_id=self._detour_authority_id(),
                process_instance_id=self._process_instance_id,
                session_id=self.session_id,
                session_purpose=self.session_purpose,
                command_name=command_name,
                input_hash=state_fingerprint(
                    canonical_user_input(input_text).encode("utf-8")
                ),
                result_kind="session_termination",
                termination_reason=termination_reason,  # type: ignore[arg-type]
                exit_code=exit_code,
                effect_class="read_only",
                shell_state_before_hash=self._shell_state_hash(),
                origin_revision_commit=revision["commit"],
                origin_revision_worktree_hash=revision["worktree_hash"],
            )
            completed = store.complete_terminal_detour(
                detour_id=detour.detour_id,
                shell_state_after_hash=self._shell_state_hash(),
                response_messages=(t(self.state.language, message_key),),
            )
            self._deliver_terminal_detour(completed)
        except Exception as exc:
            _debug_exception("terminal_termination", exc)
            return False
        self.store.save(self.state)
        self._requested_exit_code = int(exit_code)
        return True

    def _recover_and_deliver_terminal_detours(self) -> None:
        store = self._detour_store()
        store.recover_prepared_terminal_detours(
            logical_thread_id=self._detour_authority_id(),
            interrupted_message=t(
                self.state.language,
                "terminal_detour_interrupted",
            ),
        )
        for detour in store.list_undelivered_terminal_detours(
            self._detour_authority_id()
        ):
            begin_frame = getattr(self.io, "begin_frame", None)
            if callable(begin_frame):
                begin_frame()
            self._deliver_terminal_detour(
                detour,
                delivery_phase="startup_replay",
            )

    def _deliver_terminal_detour(
        self,
        detour: TerminalDetour,
        *,
        delivery_phase: str = "live",
        emit_messages: bool = True,
    ) -> None:
        if emit_messages:
            for message in detour.response_messages:
                self.io.agent(self.state.language, message)
        rendered_frame = ""
        frame_reader = getattr(self.io, "rendered_frame", None)
        if callable(frame_reader):
            rendered_frame = str(frame_reader())
        if not rendered_frame:
            rendered_frame = "\n".join(
                t(self.state.language, "agent", message=message)
                for message in detour.response_messages
            )
        projection = build_terminal_detour_projection(
            detour,
            rendered_frame=rendered_frame,
            delivery_phase=delivery_phase,  # type: ignore[arg-type]
        )
        append_terminal_detour_projection(
            self._terminal_protocol_path(),
            projection,
        )
        if delivery_phase == "startup_replay":
            self._startup_replayed_projection_ids.append(
                projection.projection_id
            )
        self._detour_store().mark_terminal_detour_delivered(
            logical_thread_id=self._detour_authority_id(),
            detour_id=detour.detour_id,
        )

    def _deliver_latest_terminal_outcome(
        self,
        *,
        fallback_key: str = "",
        fallback_message: str = "",
        emit: bool = True,
    ) -> bool:
        outcome = (
            getattr(self._harness, "last_terminal_outcome", None)
            if self._harness is not None
            else None
        )
        if outcome is None or outcome.delivered_at is not None:
            if emit and fallback_message:
                self.io.agent(self.state.language, fallback_message)
            elif emit and fallback_key:
                self.io.agent(
                    self.state.language,
                    t(self.state.language, fallback_key),
                )
            return False
        self._deliver_terminal_outcome(
            outcome,
            fallback_key=fallback_key,
            fallback_message=fallback_message,
            emit=emit,
        )
        return True

    def _deliver_terminal_outcome(
        self,
        outcome: Any,
        *,
        fallback_key: str = "",
        fallback_message: str = "",
        emit: bool = True,
        delivery_phase: str = "live",
    ) -> None:
        harness = self._ensure_harness()
        harness.ensure_runtime_observation(outcome)
        messages: tuple[str, ...] = ()
        if outcome.outcome == "committed":
            messages = harness.terminal_messages(outcome)
        elif outcome.outcome == "reconciliation_required":
            messages = (t(self.state.language, "turn_reconciliation_required"),)
        elif fallback_message:
            messages = (fallback_message,)
        elif fallback_key:
            messages = (t(self.state.language, fallback_key),)
        elif outcome.failure_category == "cancelled":
            messages = (t(self.state.language, "turn_cancelled"),)
        elif outcome.failure_category == "timeout":
            messages = (t(
                self.state.language,
                "turn_timeout",
                timeout=self._llm_config.turn_timeout_seconds,
                provider=self._llm_config.provider,
                model=self._llm_config.model,
            ),)
        else:
            messages = (t(self.state.language, "harness_runtime_error"),)
        if emit:
            for message in messages:
                self.io.agent(self.state.language, message)
        destination = self._terminal_protocol_path()
        if destination:
            rendered_frame = ""
            frame_reader = getattr(self.io, "rendered_frame", None)
            if callable(frame_reader):
                rendered_frame = str(frame_reader())
            if not rendered_frame:
                rendered_frame = "\n".join(
                    t(self.state.language, "agent", message=message)
                    for message in messages
                )
            projection = build_terminal_outcome_projection(
                outcome,
                rendered_frame=rendered_frame,
                delivery_phase=delivery_phase,  # type: ignore[arg-type]
            )
            append_terminal_outcome_projection(destination, projection)
            if delivery_phase == "startup_replay":
                self._startup_replayed_projection_ids.append(
                    projection.projection_id
                )
        harness.mark_terminal_delivered(outcome)

    def _terminal_protocol_path(self) -> str:
        configured = str(
            os.environ.get("ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE") or ""
        ).strip()
        if configured:
            return configured
        checkpoint = (
            getattr(self._harness, "checkpoint_path", None)
            if self._harness is not None
            else self.checkpoint_path
        )
        checkpoint = checkpoint or Path(".agent/langgraph/checkpoints.sqlite")
        return f"{checkpoint}.terminal-outcomes.jsonl"

    def _publish_startup_event(self) -> None:
        frame_reader = getattr(self.io, "rendered_session", None)
        if not callable(frame_reader):
            frame_reader = getattr(self.io, "rendered_frame", None)
        rendered_frame = str(frame_reader()) if callable(frame_reader) else ""
        dependency_consent_pending = self.state.current_question_id in {
            "install_agent_runtime",
            "install_dependencies",
        }
        ready = bool(
            self._llm_runtime_available
            and self._harness is not None
            and not dependency_consent_pending
        )
        failure_category = "" if ready else (
            StartupFailureCategory.DEPENDENCY_CONSENT_PENDING.value
            if dependency_consent_pending
            else self._startup_failure_category
            or StartupFailureCategory.STARTUP_BLOCKED.value
        )
        product_head = self._harness.product_head() if ready else None
        runtime_event_fence = (
            self._harness.runtime_event_fence()
            if ready
            else (0, "", "", "")
        )
        if ready and product_head is None:
            raise RuntimeError(
                "ready terminal startup lacks its Product Head authority"
            )
        event = build_terminal_session_event(
            process_instance_id=self._process_instance_id,
            session_id=self.session_id,
            session_purpose=self.session_purpose,
            provider=self._llm_config.provider,
            model=self._llm_config.model,
            auth_mode=self._llm_config.auth_mode,
            provider_ready=ready,
            startup_status="ready" if ready else "blocked",
            failure_category=failure_category,
            product_authority_id=(
                self._harness.transaction_authority_id if ready else ""
            ),
            product_revision=(
                product_head.revision if product_head is not None else None
            ),
            product_checkpoint_thread_id=(
                product_head.checkpoint_thread_id
                if product_head is not None
                else ""
            ),
            product_checkpoint_id=(
                product_head.checkpoint_id
                if product_head is not None
                else ""
            ),
            product_fingerprint=(
                product_head.state_fingerprint
                if product_head is not None
                else ""
            ),
            runtime_event_fence_sequence=runtime_event_fence[0],
            runtime_event_fence_terminal_event_id=runtime_event_fence[1],
            runtime_event_fence_id=runtime_event_fence[2],
            runtime_event_fence_hash=runtime_event_fence[3],
            rendered_frame=rendered_frame,
            origin_revision=repository_revision(REPO_ROOT),
            replayed_projection_ids=tuple(
                self._startup_replayed_projection_ids
            ),
        )
        append_terminal_session_event(self._terminal_protocol_path(), event)

    def _emit_reconciliation_barrier(self, transaction_id: str) -> None:
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "reconciliation_barrier",
                transaction_id=transaction_id,
            ),
        )

    def _handle_reconciliation_command(self, text: str) -> None:
        parts = text.split(maxsplit=3)
        if len(parts) != 4:
            self._run_terminal_detour(
                command_name="reconcile_invalid",
                effect_class="read_only",
                producer=lambda: (
                    t(
                        self.state.language,
                        "reconciliation_command_invalid",
                    ),
                ),
            )
            return
        _, transaction_id, raw_resolution, evidence = parts
        resolution = {
            "confirmed": "effect_confirmed",
            "not-observed": "effect_not_observed",
        }.get(raw_resolution.casefold(), "")
        if not resolution or not evidence.strip():
            self._run_terminal_detour(
                command_name="reconcile_invalid",
                effect_class="read_only",
                producer=lambda: (
                    t(
                        self.state.language,
                        "reconciliation_command_invalid",
                    ),
                ),
            )
            return
        response = str(redact(t(
            self.state.language,
            "reconciliation_resolved",
            transaction_id=transaction_id,
            resolution=resolution,
        )))
        revision = repository_revision(REPO_ROOT)
        try:
            detour = self._detour_store().resolve_with_terminal_detour(
                logical_thread_id=self._detour_authority_id(),
                process_instance_id=self._process_instance_id,
                session_id=self.session_id,
                session_purpose=self.session_purpose,
                target_id=transaction_id,
                input_hash=state_fingerprint(text.encode("utf-8")),
                resolution=resolution,
                evidence_hash=hashlib.sha256(
                    evidence.strip().encode("utf-8")
                ).hexdigest(),
                shell_state_hash=self._shell_state_hash(),
                response_messages=(response,),
                origin_revision_commit=revision["commit"],
                origin_revision_worktree_hash=revision["worktree_hash"],
            )
        except Exception:
            self._run_terminal_detour(
                command_name="reconcile_invalid",
                effect_class="read_only",
                producer=lambda: (
                    t(
                        self.state.language,
                        "reconciliation_command_invalid",
                    ),
                ),
            )
            return
        self._deliver_terminal_detour(detour)

    def _handle_pending_confirmation(self, lowered: str) -> bool:
        if self.state.current_question_id == "install_agent_runtime":
            if lowered in {"y", "yes", ""}:
                self._run_terminal_detour(
                    command_name="accept_agent_runtime_install",
                    effect_class="shell_state_mutation",
                    producer=self._install_agent_runtime_messages,
                )
                return True
            if lowered in {"n", "no"}:
                self._run_terminal_detour(
                    command_name="decline_agent_runtime_install",
                    effect_class="shell_state_mutation",
                    producer=lambda: self._decline_install_messages(
                        "agent_runtime_declined"
                    ),
                )
                return True
        if self.state.current_question_id == "install_dependencies":
            if lowered in {"y", "yes", ""}:
                self._run_terminal_detour(
                    command_name="accept_dependency_install",
                    effect_class="shell_state_mutation",
                    producer=self._install_dependencies_messages,
                )
                return True
            if lowered in {"n", "no"}:
                self._run_terminal_detour(
                    command_name="decline_dependency_install",
                    effect_class="shell_state_mutation",
                    producer=lambda: self._decline_install_messages(
                        "dependency_declined"
                    ),
                )
                return True
        return False

    def _decline_install_messages(self, message_key: str) -> tuple[str, ...]:
        self.state.current_question_id = ""
        return (t(self.state.language, message_key),)

    def _ensure_harness(self) -> AnyChainGraphRuntime:
        if self._harness is None:
            self._harness = AnyChainGraphRuntime(
                thread_id=self.session_id,
                checkpoint_path=self.checkpoint_path,
                session_purpose=self.session_purpose,
            )
        return self._harness

    def _offer_harness_resume_if_needed(self) -> None:
        snapshot = self._ensure_harness().snapshot()
        session = snapshot.get("session") or {}
        snapshot_purpose = str(session.get("purpose") or "user")
        if self.session_purpose == "user" and snapshot_purpose != "user":
            return
        if snapshot.get("evidence_collection"):
            self._ensure_harness().clear_evidence_collection()
            self._deliver_latest_terminal_outcome(emit=False)
        self._ensure_harness().prepare_resume_offer(self.state.language)
        self._deliver_latest_terminal_outcome()

    def _startup_doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "startup_doctor_start"))
        self._emit_doctor_summary(run_doctor(), startup=True)

    def _doctor(self) -> None:
        for message in self._doctor_messages():
            self.io.agent(self.state.language, message)

    def _doctor_messages(self) -> tuple[str, ...]:
        return (
            t(self.state.language, "doctor_start"),
            *self._doctor_summary_messages(run_doctor(), startup=False),
        )

    def _emit_doctor_summary(self, report: dict[str, Any], startup: bool) -> None:
        for message in self._doctor_summary_messages(report, startup=startup):
            self.io.agent(self.state.language, message)

    def _doctor_summary_messages(
        self,
        report: dict[str, Any],
        *,
        startup: bool,
    ) -> tuple[str, ...]:
        caps = report.get("capabilities", {})
        env = report.get("environment", {})
        messages: list[str] = []
        self.state.discovery = env
        cloud = env.get("cloud", {})
        deployment = env.get("deployment", {})
        missing = env.get("dependencies", {}).get("missing_required", [])
        if not missing and self.state.current_question_id == "install_dependencies":
            self.state.current_question_id = ""
        self.state.pending_missing_dependencies = list(missing)
        if startup:
            messages.append(
                t(
                    self.state.language,
                    "startup_doctor_summary",
                    status=report.get("status", "unknown"),
                    cloud=cloud.get("provider", "unknown"),
                    deployment=deployment.get("type", "unknown"),
                    missing=", ".join(missing) if missing else "<none>",
                    chains=caps.get("chain_count", "?"),
                    methods=caps.get("unique_rpc_method_count", "?"),
                ),
            )
            summary = _format_environment_inference(env)
            if summary:
                messages.append(
                    t(
                        self.state.language,
                        "environment_inference_summary",
                        summary=summary,
                    )
                )
        else:
            messages.append(
                t(
                    self.state.language,
                    "doctor_summary",
                    status=report.get("status", "unknown"),
                    missing=", ".join(missing) if missing else "<none>",
                    chains=caps.get("chain_count", "?"),
                    methods=caps.get("unique_rpc_method_count", "?"),
                ),
            )
        if missing and self.state.current_question_id != "install_agent_runtime":
            self.state.current_question_id = "install_dependencies"
            messages.append(
                t(
                    self.state.language,
                    "dependency_offer",
                    missing=", ".join(missing),
                )
            )
        return tuple(messages)

    def _install_agent_runtime(self) -> None:
        for message in self._install_agent_runtime_messages():
            self.io.agent(self.state.language, message)

    def _install_agent_runtime_messages(self) -> tuple[str, ...]:
        self.state.current_question_id = ""
        return (
            t(
                self.state.language,
                "agent_runtime_install_external",
                command="bash scripts/install_agent_deps.sh --yes",
            ),
        )

    def _establish_provider_readiness(self) -> bool:
        """Apply the one live-provider readiness transition used by the CLI."""

        readiness_error = probe_provider_readiness(self._llm_config)
        self._llm_readiness_error = readiness_error
        self._llm_runtime_available = readiness_error is None
        self._llm_unavailable_reason = _provider_error_summary(readiness_error)
        if readiness_error is not None:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "llm_provider_unavailable",
                    reason=self._llm_unavailable_reason,
                ),
            )
        return self._llm_runtime_available

    def _install_dependencies(self) -> None:
        for message in self._install_dependencies_messages():
            self.io.agent(self.state.language, message)

    def _install_dependencies_messages(self) -> tuple[str, ...]:
        self.state.current_question_id = ""
        return (
            t(
                self.state.language,
                "dependency_install_external",
                command="bash scripts/install_deps.sh --yes",
            ),
        )

    def _load_framework_context(self) -> None:
        context = load_framework_context(language=self.state.language)
        summary = context.get("capability_summary", {})
        capabilities = load_framework_capabilities()
        self.state.framework_summary = {
            **summary,
            "chain_count": capabilities.get("chain_count", summary.get("chain_count")),
            "family_count": capabilities.get("family_count", summary.get("family_count")),
            "unique_rpc_method_count": capabilities.get("unique_rpc_method_count", summary.get("unique_rpc_method_count")),
            "chains": capabilities.get("chains", []),
            "client_metric_profiles": capabilities.get("client_metric_profiles", []),
        }
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "framework_context_loaded",
                chains=self.state.framework_summary.get("chain_count", "?"),
                families=self.state.framework_summary.get("family_count", "?"),
                methods=self.state.framework_summary.get("unique_rpc_method_count", "?"),
                fixtures=summary.get("fake_node_fixture_file_count", "?"),
            ),
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AnyChain Benchmark Agent product terminal.")
    parser.add_argument("--prompt", action="append", help="Run one or more scripted user turns, then exit.")
    parser.add_argument("--language", choices=["zh", "en"], default=None, help="Initial terminal language.")
    parser.add_argument("--state-file", help="Use an alternate terminal state file for tests or isolated sessions.")
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_ID", DEFAULT_AGENT_SESSION_ID),
        help="Agent session id. Tests and live matrices should isolate it per scenario.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("ANYCHAIN_AGENT_CHECKPOINT_PATH"),
        help="Use an explicit LangGraph checkpoint database. Tests must isolate this per scenario.",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Reset the selected Harness thread before handling user turns.",
    )
    parser.add_argument(
        "--session-purpose",
        choices=["user", "chaos", "dynamic-dual-ai-chaos", "live-matrix", "dev"],
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_PURPOSE", "user"),
        help="Label the runtime session so test/dev checkpoints cannot be resumed as user state.",
    )
    return parser.parse_args(argv)


def _format_environment_inference(env: dict[str, Any]) -> str:
    cloud = env.get("cloud", {}) or {}
    deployment = env.get("deployment", {}) or {}
    host = env.get("host", {}) or {}
    network = env.get("network", {}) or {}
    disks = env.get("disks", {}) or {}
    deployment_value = _known_value(cloud.get("platform")) or _known_value(deployment.get("type")) or "<needs confirmation>"
    lines = [
        f"- CLOUD_PROVIDER: {cloud.get('provider') or '<needs confirmation>'}",
        f"- deployment: {deployment_value}",
        f"- CLOUD_REGION: {cloud.get('region') or '<needs confirmation>'}",
        f"- CLOUD_ZONE: {cloud.get('zone') or '<needs confirmation>'}",
        f"- MACHINE_TYPE: {cloud.get('machine_type') or '<needs confirmation>'}",
        f"- CPU: {host.get('cpu_count') or '<unknown>'}",
        f"- Memory: {_format_memory(host.get('memory_gib'))}",
        f"- NETWORK_INTERFACE: {network.get('default_interface') or '<needs confirmation>'}",
        f"- LEDGER_DEVICE candidate: {disks.get('proposed_ledger_device') or '<needs confirmation>'}",
        f"- ACCOUNTS_DEVICE candidate: {disks.get('proposed_accounts_device') or '<none detected>'}",
    ]
    candidates = disks.get("candidates") or []
    if candidates:
        lines.append("- Disk candidates:")
        for index, item in enumerate(candidates[:8], start=1):
            lines.append(
                f"  [{index}] {item.get('name') or '<unknown>'} "
                f"type={item.get('type') or '<unknown>'} "
                f"size={item.get('size') or '<unknown>'} "
                f"mount={item.get('mountpoint') or '<none>'} "
                f"label={item.get('label') or '<none>'}"
            )
        if len(candidates) > 8:
            lines.append(f"  ... {len(candidates) - 8} more")
    else:
        lines.append("- Disk candidates: <none detected; Agent should ask manually if needed>")
    interfaces = _usable_interfaces(list(network.get("interfaces") or []), str(network.get("default_interface") or ""))
    if interfaces:
        lines.append("- Network interface candidates:")
        for index, name in enumerate(interfaces[:8], start=1):
            suffix = " (default)" if name == network.get("default_interface") else ""
            lines.append(f"  [{index}] {name}{suffix}")
        if len(interfaces) > 8:
            lines.append(f"  ... {len(interfaces) - 8} more")
    return "\n".join(lines)


def _known_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return ""
    return text


def _is_shell_command(stripped: str, lowered: str, aliases: set[str]) -> bool:
    """Return true only for exact stable terminal commands."""

    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases


def _format_memory(value: Any) -> str:
    if value in {None, ""}:
        return "<unknown>"
    return f"{value} GiB"


def _usable_interfaces(interfaces: list[str], default: str = "") -> list[str]:
    blocked_exact = {"lo", "bonding_masters", "sit0", "tunl0", "gre0", "gretap0", "erspan0", "ip_vti0", "ip6_vti0", "ip6gre0", "ip6tnl0"}
    output = []
    for raw in interfaces:
        name = str(raw or "").strip()
        if not name:
            continue
        if name == default:
            output.append(name)
            continue
        if name in blocked_exact:
            continue
        output.append(name)
    if default and default not in output:
        output.insert(0, default)
    return output or interfaces


def _debug_exception(scope: str, exc: Exception) -> None:
    if os.environ.get("ANYCHAIN_DEBUG") not in {"1", "true", "TRUE", "yes", "YES"}:
        return
    message = redact(f"{type(exc).__name__}: {exc}")
    print(f"[anychain-debug] {scope}: {message}", file=sys.stderr)


def _adk_error_message(language: str, exc: Exception) -> str:
    if isinstance(exc, StateInvariantError) or (
        type(exc).__name__ == "StateInvariantError"
        and type(exc).__module__.endswith("harness.invariants")
    ):
        return t(language, "harness_runtime_error")
    if isinstance(exc, LLMProviderError):
        return t(
            language,
            "llm_provider_unavailable",
            reason=_provider_error_summary(exc),
        )
    message = str(exc).lower()
    if "insufficient balance" in message or "insufficient quota" in message:
        return t(language, "llm_billing_error")
    module = type(exc).__module__.casefold()
    provider_markers = (
        "openai",
        "anthropic",
        "google.api_core",
        "google.genai",
        "httpx",
        "httpcore",
        "urllib",
    )
    provider_message_markers = (
        "api key",
        "authentication",
        "rate limit",
        "model provider",
        "connection error",
        "request timeout",
    )
    if any(marker in module for marker in provider_markers) or any(
        marker in message for marker in provider_message_markers
    ):
        return t(language, "adk_runtime_error")
    return t(language, "harness_runtime_error")


def _provider_error_summary(exc: LLMProviderError | None) -> str:
    if exc is None:
        return ""
    identity = "/".join(
        item for item in (exc.provider, exc.model) if str(item).strip()
    )
    status = f", HTTP {exc.status_code}" if exc.status_code else ""
    return f"{identity or 'model provider'}: {exc.category or 'provider'}{status}"


@contextmanager
def _terminal_sigint_scope(app: AnyChainTerminal) -> Iterator[None]:
    """Give one SIGINT a stable meaning for the lifetime of the interactive CLI."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def _dispatch(_signum: int, _frame: Any) -> None:
        if app._turn_active:
            raise LLMTurnCancelledError("active Agent turn cancelled by SIGINT")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _dispatch)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@contextmanager
def _interactive_turn_scope(timeout_seconds: float) -> Iterator[None]:
    """Own SIGINT/SIGALRM while one synchronous product turn is active."""

    can_manage_signals = threading.current_thread() is threading.main_thread()
    can_manage_alarm = can_manage_signals and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    if not can_manage_signals:
        with llm_turn_scope(timeout_seconds):
            yield
        return

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigalrm = signal.getsignal(signal.SIGALRM) if can_manage_alarm else None
    previous_timer = signal.getitimer(signal.ITIMER_REAL) if can_manage_alarm else (0.0, 0.0)
    started = time.monotonic()

    def _cancel_turn(_signum: int, _frame: Any) -> None:
        raise LLMTurnCancelledError("active Agent turn cancelled by SIGINT")

    def _expire_turn(_signum: int, _frame: Any) -> None:
        raise LLMTurnTimeoutError(f"Agent turn exceeded its {timeout_seconds:g}s deadline")

    signal.signal(signal.SIGINT, _cancel_turn)
    if can_manage_alarm:
        signal.signal(signal.SIGALRM, _expire_turn)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        with llm_turn_scope(timeout_seconds):
            yield
    finally:
        if can_manage_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_sigalrm)
            old_delay, old_interval = previous_timer
            if old_delay > 0:
                remaining = max(0.000001, old_delay - (time.monotonic() - started))
                signal.setitimer(signal.ITIMER_REAL, remaining, old_interval)
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
