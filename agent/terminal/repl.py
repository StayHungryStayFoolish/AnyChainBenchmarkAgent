#!/usr/bin/env python3
"""Product terminal for AnyChain Benchmark Agent.

The terminal owns stable input/output, startup diagnostics, dependency
installation consent, and job recovery notices. Conversation planning and
workflow orchestration are delegated to the in-process Google ADK Runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from adk_app.models import adk_status  # noqa: E402
from adk_app.runner_bridge import ADKRunnerBridge, runner_bridge_status  # noqa: E402
from adk_app.state import load_startup_state  # noqa: E402
from adk_app.tools.web_research import web_research_status  # noqa: E402
from diagnostics.doctor import run_doctor  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from knowledge.framework_context import load_framework_context  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from terminal.input_classifier import classify_input_mode as _classify_input_mode  # noqa: E402
from terminal.input_classifier import compose_evidence_question  # noqa: E402
from terminal.input_classifier import split_pasted_evidence_question  # noqa: E402
from terminal.io import OutputOnlyIO, TerminalIO  # noqa: E402
from terminal.job_commands import JobCommandHandler  # noqa: E402
from terminal.language import detect_language, t  # noqa: E402
from terminal.pending_answers import compose_pending_answer_followup  # noqa: E402
from terminal.pending_answers import is_structural_pending_answer  # noqa: E402
from terminal.pending_answers import is_unbound_structural_answer  # noqa: E402
from utils.redaction import redact  # noqa: E402
from workflows.conversation_state import DEFAULT_SESSION_ID  # noqa: E402
from workflows.conversation_state import answer_pending_question as apply_pending_answer  # noqa: E402
from workflows.conversation_state import load_workflow_state  # noqa: E402
from workflows.conversation_state import reset_workflow_state  # noqa: E402
from workflows.conversation_state import update_workflow_state  # noqa: E402
from workflows.transition_executor import advance_after_pending_answer  # noqa: E402
from workflows.transition_executor import ensure_next_benchmark_setup_question  # noqa: E402
from workflows.transition_executor import render_pending_question  # noqa: E402


@dataclass
class TerminalSession:
    """Small terminal shell state, not a benchmark workflow state machine."""

    language: str = "en"
    current_question_id: str = ""
    latest_job_id: str = ""
    discovery: dict[str, Any] = field(default_factory=dict)
    framework_summary: dict[str, Any] = field(default_factory=dict)
    pending_missing_dependencies: list[str] = field(default_factory=list)
    pending_evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TerminalSession":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})


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
        self.path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = TerminalSessionStore(args.state_file)
    state = store.load()
    if args.language:
        state.language = args.language
    elif args.prompt:
        state.language = detect_language(args.prompt[0], state.language)
    io = OutputOnlyIO() if args.prompt else TerminalIO()
    app = AnyChainTerminal(state=state, store=store, io=io, session_id=args.session_id)
    if args.prompt:
        app.startup()
        for prompt in args.prompt:
            app.handle_user_text(prompt)
            store.save(state)
        return 0
    return app.run()


class AnyChainTerminal:
    def __init__(
        self,
        state: TerminalSession | None = None,
        store: TerminalSessionStore | None = None,
        io: TerminalIO | OutputOnlyIO | None = None,
        bridge_factory: Callable[[], ADKRunnerBridge] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.state = state or TerminalSession()
        self.store = store or TerminalSessionStore()
        self.io = io or TerminalIO()
        self.session_id = session_id or DEFAULT_SESSION_ID
        self._bridge_factory = bridge_factory or (lambda: ADKRunnerBridge(session_id=self.session_id))
        self._bridge: ADKRunnerBridge | None = None
        self._startup_state: dict[str, Any] = {}
        self._llm_config = load_llm_config()
        self._web_research_status: dict[str, Any] = {}
        self._adk_available = False
        self._job_commands = JobCommandHandler(self.state, self.io)

    def run(self) -> int:
        self.startup()
        while True:
            try:
                text = self.io.input(self.state.language).strip()
            except KeyboardInterrupt:
                self.io.agent(self.state.language, t(self.state.language, "ctrl_c_exit"))
                self.store.save(self.state)
                return 130
            except EOFError:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                self.store.save(self.state)
                return 0

            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                self.store.save(self.state)
                return 0
            self.handle_user_text(text)
            self.store.save(self.state)

    def startup(self) -> None:
        self._llm_config = load_llm_config()
        self._startup_state = load_startup_state()
        latest_job = self._startup_state.get("latest_job") or {}
        if latest_job:
            self.state.latest_job_id = latest_job.get("job_id", "")
        else:
            self.state.latest_job_id = ""
        self._clear_stale_pending_question_after_terminal_job(latest_job)
        self._clear_stale_benchmark_context_after_terminal_job(latest_job)

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
        bridge_status = runner_bridge_status().as_dict()
        self._adk_available = bool(status.get("available") and bridge_status.get("available"))
        self.io.agent(self.state.language, t(self.state.language, "adk", status=bridge_status.get("reason", status)))
        if self._adk_available and self.state.current_question_id == "install_agent_runtime":
            self.state.current_question_id = ""
            self.state.pending_missing_dependencies = []
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

        if not self._adk_available:
            self.state.current_question_id = "install_agent_runtime"
            self.state.pending_missing_dependencies = ["google-adk"]
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
        else:
            self._ensure_bridge()
        self.io.agent(self.state.language, t(self.state.language, "help"))

    def _clear_stale_pending_question_after_terminal_job(self, latest_job: dict[str, Any]) -> None:
        status = str((latest_job or {}).get("status") or "").strip().lower()
        if status not in {"completed", "failed", "cancelled", "canceled"}:
            return
        workflow_state = load_workflow_state(session_id=self.session_id)
        question = workflow_state.get("pending_question") or {}
        question_id = str(question.get("id") or "")
        if not question_id or question_id in {"previous_job_resume", "job_follow_logs"}:
            return
        update_workflow_state(
            {
                "pending_question": {},
                "allowed_next_actions": [],
            },
            reason=f"startup_clear_stale_pending_after_{status}_job",
            session_id=self.session_id,
        )

    def _clear_stale_benchmark_context_after_terminal_job(self, latest_job: dict[str, Any]) -> None:
        """Keep completed jobs discoverable without reusing their benchmark plan.

        A completed job is history, not the current benchmark setup. Leaving
        chain/RPC/workload/profile fields active caused a greeting to resume
        the old Solana mixed flow and skip chain selection.
        """
        status = str((latest_job or {}).get("status") or "").strip().lower()
        if status and status not in {"completed", "failed", "cancelled", "canceled"}:
            return
        workflow_state = load_workflow_state(session_id=self.session_id)
        if workflow_state.get("pending_question"):
            return
        active_fields = (
            "active_intent",
            "active_workflow",
            "workflow_step",
            "target_mode",
            "chain",
            "rpc_mode",
            "rpc_methods",
            "mixed_weights",
            "custom_rpc",
            "custom_rpc_methods",
            "benchmark_profile",
            "assumed_for_smoke",
            "allowed_next_actions",
            "approval",
            "preflight_result",
            "smoke_result",
            "blockers",
        )
        if not any(workflow_state.get(key) for key in active_fields):
            return
        reset_workflow_state(
            reason="startup_clear_completed_job_benchmark_context",
            session_id=self.session_id,
        )

    def handle_user_text(self, text: str) -> None:
        self.state.language = detect_language(text, self.state.language)
        update_workflow_state(
            {"language": self.state.language},
            reason="terminal_language_detected",
            session_id=self.session_id,
        )
        stripped = text.strip()
        lowered = stripped.lower()
        input_mode = _classify_input_mode(text)
        if _is_shell_command(stripped, lowered, {"exit", "quit", "q"}):
            self.io.agent(self.state.language, t(self.state.language, "bye"))
            return
        if _is_shell_command(stripped, lowered, {"help", "?", "帮助", "？"}):
            self.io.agent(self.state.language, t(self.state.language, "help"))
            return
        if _is_shell_command(stripped, lowered, {"doctor", "环境检查"}):
            self._doctor()
            return
        if self._job_commands.handle_jobs_command(stripped, lowered):
            return
        if self._job_commands.handle_status_command(stripped, lowered):
            return
        if self._job_commands.handle_log_command(stripped, lowered):
            return
        if self._handle_workflow_pending_answer(stripped):
            return
        if self._handle_pending_confirmation(lowered):
            return
        if not load_workflow_state(session_id=self.session_id).get("pending_question") and is_unbound_structural_answer(stripped):
            self.io.agent(self.state.language, t(self.state.language, "unbound_structural_answer"))
            return
        if not self._adk_available:
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.state.current_question_id = "install_agent_runtime"
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
            return
        try:
            outbound_text = text
            before_workflow_state = load_workflow_state(session_id=self.session_id)
            if input_mode == "pasted_evidence":
                evidence, question = split_pasted_evidence_question(text)
                if question:
                    outbound_text = compose_evidence_question([evidence], question)
                    self.io.agent(self.state.language, t(self.state.language, "pasted_evidence_detected"))
                else:
                    self.state.pending_evidence.append(text)
                    if len(self.state.pending_evidence) == 1:
                        self.io.agent(self.state.language, t(self.state.language, "pasted_evidence_buffered"))
                    return
            elif self.state.pending_evidence:
                input_mode = "evidence_question"
                outbound_text = compose_evidence_question(self.state.pending_evidence, text)
                self.state.pending_evidence = []
                self.io.agent(self.state.language, t(self.state.language, "pasted_evidence_detected"))
            elif input_mode == "normal_user_turn":
                current_workflow_state = load_workflow_state(session_id=self.session_id)
                pending_question = current_workflow_state.get("pending_question") or {}
                if pending_question:
                    if pending_question.get("blocks_execution") is False:
                        update_workflow_state(
                            {"pending_question": {}},
                            reason="terminal_supersede_nonblocking_pending_question",
                            session_id=self.session_id,
                        )
                    else:
                        input_mode = "pending_question_interruption"
            self.io.agent(self.state.language, t(self.state.language, "thinking"))
            response = self._ensure_bridge().run_text(outbound_text, state_delta=self._state_delta(input_mode=input_mode))
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.io.agent(self.state.language, t(self.state.language, "turn_cancelled"))
            return
        except Exception as exc:
            _debug_exception("adk_turn", exc)
            self.io.agent(self.state.language, _adk_error_message(self.state.language, exc))
            return
        state_override = self._terminal_state_override_after_adk()
        if state_override.get("suppress_response"):
            self.io.agent(self.state.language, str(state_override.get("message") or t(self.state.language, "unknown")))
            return
        setup_question = self._setup_question_after_adk(before_workflow_state)
        if setup_question.get("suppress_response"):
            message = str(setup_question.get("message") or "").strip()
            if message:
                self.io.agent(self.state.language, message)
            else:
                self.io.agent(self.state.language, t(self.state.language, "unknown"))
        elif response:
            self.io.agent(self.state.language, response)
        else:
            self.io.agent(self.state.language, t(self.state.language, "unknown"))
        if not setup_question.get("suppress_response"):
            message = str(setup_question.get("message") or "").strip()
            if message:
                self.io.agent(self.state.language, message)

    def _terminal_state_override_after_adk(self) -> dict[str, Any]:
        workflow_state = load_workflow_state(session_id=self.session_id)
        fixture_status = workflow_state.get("fixture_status") if isinstance(workflow_state.get("fixture_status"), dict) else {}
        if str(fixture_status.get("status") or "") != "needs_review":
            return {"suppress_response": False}
        step = str(workflow_state.get("workflow_step") or "")
        if step not in {"unsupported_chain_handoff_requested", "custom_rpc_handoff_requested"}:
            return {"suppress_response": False}
        chain = str(workflow_state.get("chain") or workflow_state.get("confirmed_config", {}).get("BLOCKCHAIN_NODE") or "selected-chain").strip()
        blockers = [str(item) for item in (workflow_state.get("blockers") or []) if str(item).strip()]
        if self.state.language.startswith("zh"):
            if step == "custom_rpc_handoff_requested":
                header = f"{chain} 的自定义 RPC method 目前只能进入 `needs_review` 交接。"
            else:
                header = f"{chain} 目前只能进入 `needs_review` 新链接入交接。"
            lines = [
                header,
                "原因：缺少已验证的 endpoint、真实 request/response、fixture 录制和 smoke 证据，不能声明已经支持或可以压测。",
            ]
            if blockers:
                lines.append("阻塞项：")
                lines.extend(f"- {item}" for item in blockers)
            lines.append("下一步：把这个 needs_review 交接给编码 AI 或开发者；补齐 endpoint 和样本后再回到验证流程。")
        else:
            if step == "custom_rpc_handoff_requested":
                header = f"Custom RPC onboarding for {chain} is `needs_review`."
            else:
                header = f"New-chain onboarding for {chain} is `needs_review`."
            lines = [
                header,
                "Reason: validated endpoint, live request/response evidence, fixture recording, and smoke evidence are missing. The Agent must not claim support or benchmark readiness yet.",
            ]
            if blockers:
                lines.append("Blockers:")
                lines.extend(f"- {item}" for item in blockers)
            lines.append("Next: hand this needs_review brief to a coding AI or developer, then return with endpoint and samples for validation.")
        return {"suppress_response": True, "message": "\n".join(lines)}

    def _handle_workflow_pending_answer(self, stripped: str) -> bool:
        workflow_state = load_workflow_state(session_id=self.session_id)
        question = workflow_state.get("pending_question") or {}
        if not question or not is_structural_pending_answer(stripped, question):
            if question and stripped:
                update_workflow_state(
                    {
                        "last_user_change": {
                            "type": (
                                "unsupported_chain_candidate"
                                if question.get("id") == "chain_selection"
                                else "pending_question_interruption"
                            ),
                            "raw": stripped,
                            "pending_question_id": question.get("id"),
                        }
                    },
                    reason="terminal_preserve_raw_pending_interruption",
                    session_id=self.session_id,
                )
            return False
        result = apply_pending_answer(
            stripped,
            reason="terminal_pending_answer",
            session_id=self.session_id,
        )
        if not result.get("applied"):
            blockers = _format_pending_answer_blockers(self.state.language, result.get("blockers") or [])
            self.io.agent(self.state.language, t(self.state.language, "pending_answer_blocked", blockers=blockers))
            prompt = render_pending_question(question, language=self.state.language)
            if prompt:
                self.io.agent(self.state.language, str(prompt))
            return True
        self.state.pending_evidence = []
        next_step = advance_after_pending_answer(
            result,
            discovery=self.state.discovery,
            language=self.state.language,
            session_id=self.session_id,
        )
        if next_step.get("message"):
            self.io.agent(self.state.language, str(next_step["message"]))
        elif (
            result.get("question_id") == "smoke_run_confirm"
            and result.get("answer_kind") == "yes"
            and (load_workflow_state(session_id=self.session_id).get("target_mode") == "fake-node")
        ):
            self._run_confirmed_fake_node_smoke()
        elif (
            result.get("question_id") == "quick_assumed_smoke_confirm"
            and result.get("answer_kind") == "yes"
        ):
            self._run_quick_assumed_fake_node_smoke()
        elif self._adk_available and next_step.get("delegate_to_adk"):
            try:
                self.io.agent(self.state.language, t(self.state.language, "thinking"))
                response = self._ensure_bridge().run_text(
                    compose_pending_answer_followup(stripped, result),
                    state_delta=self._state_delta(input_mode="pending_answer_applied"),
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.io.agent(self.state.language, t(self.state.language, "turn_cancelled"))
                return True
            except Exception as exc:
                _debug_exception("pending_answer_followup", exc)
                self.io.agent(self.state.language, _adk_error_message(self.state.language, exc))
                return True
            self.io.agent(self.state.language, response or t(self.state.language, "pending_answer_recorded"))
        else:
            self.io.agent(self.state.language, t(self.state.language, "pending_answer_recorded"))
        return True

    def _run_quick_assumed_fake_node_smoke(self) -> None:
        """Execute the approved quick assumed smoke path deterministically."""
        from adk_app.tools.actions import run_quick_assumed_fake_node_smoke

        workflow_state = load_workflow_state(session_id=self.session_id)
        chain = str(workflow_state.get("chain") or "solana").strip().lower() or "solana"
        rpc_mode = str(workflow_state.get("rpc_mode") or "single").strip().lower() or "single"
        if rpc_mode not in {"single", "mixed"}:
            rpc_mode = "single"

        self.io.agent(self.state.language, t(self.state.language, "thinking"))
        smoke = run_quick_assumed_fake_node_smoke(
            source_prompt="terminal approved quick assumed fake-node smoke",
            chain=chain,
            rpc_mode=rpc_mode,
            approved=True,
        )
        data = smoke.get("data") or {}
        nested_smoke = data.get("smoke") or {}
        job = {}
        if isinstance(nested_smoke, dict):
            job = nested_smoke.get("job") or ((nested_smoke.get("data") or {}).get("job") or {})
        job_id = str(job.get("job_id") or "")
        if job_id:
            self.state.latest_job_id = job_id
        status = str(job.get("status") or smoke.get("status") or "unknown")
        if status not in {"completed", "running"}:
            update_workflow_state(
                {
                    "latest_job_id": job_id,
                    "pending_question": {},
                    "allowed_next_actions": ["ask:logs", "ask:status", "ask:fix blockers"],
                    "workflow_step": "job_failed",
                    "active_workflow": "job_monitoring",
                    "smoke_result": smoke,
                    "assumed_for_smoke": True,
                },
                reason="terminal_quick_assumed_smoke_failed",
                session_id=self.session_id,
            )
            commands = data.get("terminal_commands") or {}
            warnings = [item for item in smoke.get("warnings", []) if item]
            message = "\n".join(warnings) if warnings else "benchmark job did not start successfully"
            if self.state.language.startswith("zh"):
                self.io.agent(
                    self.state.language,
                    "quick fake-node smoke 未能启动或执行失败。"
                    f"\n- job_id: {job_id or '<none>'}"
                    f"\n- 状态: {status}"
                    f"\n- 原因: {message}"
                    f"\n- 日志: {commands.get('logs', f'logs {job_id}') if job_id else 'logs <job_id>'}",
                )
            else:
                self.io.agent(
                    self.state.language,
                    "quick fake-node smoke did not start or failed."
                    f"\n- job_id: {job_id or '<none>'}"
                    f"\n- status: {status}"
                    f"\n- reason: {message}"
                    f"\n- logs: {commands.get('logs', f'logs {job_id}') if job_id else 'logs <job_id>'}",
                )
            return
        update_workflow_state(
            {
                "latest_job_id": job_id,
                "pending_question": {},
                "allowed_next_actions": ["ask:status", "ask:logs", "ask:follow", "ask:analyze"],
                "workflow_step": "job_submitted",
                "active_workflow": "job_monitoring",
                "smoke_result": smoke,
                "assumed_for_smoke": True,
            },
            reason="terminal_quick_assumed_smoke_submitted",
            session_id=self.session_id,
        )
        commands = data.get("terminal_commands") or {}
        if self.state.language.startswith("zh"):
            self.io.agent(
                self.state.language,
                "quick fake-node smoke 已提交。"
                f"\n- job_id: {job_id or '<unknown>'}"
                f"\n- 状态: {status}"
                f"\n- 标记: assumed_for_smoke=true"
                f"\n- 日志: {commands.get('logs', f'logs {job_id}')}"
                f"\n- 实时日志: {commands.get('follow', f'follow {job_id}')}"
                f"\n- 分析: {commands.get('analyze', 'analyze latest job')}",
            )
        else:
            self.io.agent(
                self.state.language,
                "quick fake-node smoke submitted."
                f"\n- job_id: {job_id or '<unknown>'}"
                f"\n- status: {status}"
                f"\n- marker: assumed_for_smoke=true"
                f"\n- logs: {commands.get('logs', f'logs {job_id}')}"
                f"\n- follow: {commands.get('follow', f'follow {job_id}')}"
                f"\n- analyze: {commands.get('analyze', 'analyze latest job')}",
            )

    def _run_confirmed_fake_node_smoke(self) -> None:
        """Execute the final approved fake-node smoke gate deterministically.

        The LLM may infer intent and collect fields, but once the typed
        `smoke_run_confirm` gate receives `Y`, the terminal must launch the
        canonical toolchain instead of asking the model to reinterpret the
        already-confirmed configuration.
        """
        from adk_app.tools.actions import run_fake_node_smoke_benchmark
        from adk_app.tools.planning import prepare_benchmark_run

        workflow_state = load_workflow_state(session_id=self.session_id)
        confirmed = _effective_confirmed_config(workflow_state)
        chain = str(workflow_state.get("chain") or confirmed.get("chain") or confirmed.get("BLOCKCHAIN_NODE") or "").strip()
        rpc_mode = str(workflow_state.get("rpc_mode") or confirmed.get("rpc_mode") or confirmed.get("RPC_MODE") or "single").strip().lower()
        if not chain:
            self.io.agent(self.state.language, _localized_simple(self.state.language, "缺少链名，无法生成 smoke plan。请先选择链。", "Missing chain name; choose a chain before smoke."))
            return

        self.io.agent(self.state.language, t(self.state.language, "thinking"))
        prepared = prepare_benchmark_run(
            source_prompt="terminal approved fake-node smoke",
            chain=chain,
            goal="smoke",
            rpc_mode=rpc_mode or "single",
            use_fake_node=True,
            ledger_device=str(confirmed.get("LEDGER_DEVICE") or confirmed.get("ledger_device") or ""),
            accounts_device=str(confirmed.get("ACCOUNTS_DEVICE") or confirmed.get("accounts_device") or ""),
            blockchain_process_names=_process_names_from_confirmed(confirmed),
            cloud_region=str(confirmed.get("CLOUD_REGION") or confirmed.get("cloud_region") or ""),
            cloud_zone=str(confirmed.get("CLOUD_ZONE") or confirmed.get("cloud_zone") or ""),
            machine_type=str(confirmed.get("MACHINE_TYPE") or confirmed.get("machine_type") or ""),
            data_vol_type=str(confirmed.get("DATA_VOL_TYPE") or confirmed.get("data_vol_type") or ""),
            data_vol_size=str(confirmed.get("DATA_VOL_SIZE") or confirmed.get("data_vol_size") or ""),
            data_vol_max_iops=str(confirmed.get("DATA_VOL_MAX_IOPS") or confirmed.get("data_vol_max_iops") or ""),
            data_vol_max_throughput=str(confirmed.get("DATA_VOL_MAX_THROUGHPUT") or confirmed.get("data_vol_max_throughput") or ""),
            accounts_vol_type=str(confirmed.get("ACCOUNTS_VOL_TYPE") or confirmed.get("accounts_vol_type") or ""),
            accounts_vol_size=str(confirmed.get("ACCOUNTS_VOL_SIZE") or confirmed.get("accounts_vol_size") or ""),
            accounts_vol_max_iops=str(confirmed.get("ACCOUNTS_VOL_MAX_IOPS") or confirmed.get("accounts_vol_max_iops") or ""),
            accounts_vol_max_throughput=str(confirmed.get("ACCOUNTS_VOL_MAX_THROUGHPUT") or confirmed.get("accounts_vol_max_throughput") or ""),
            network_interface=str(confirmed.get("NETWORK_INTERFACE") or confirmed.get("network_interface") or ""),
            network_max_bandwidth_gbps=str(confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS") or confirmed.get("network_max_bandwidth_gbps") or ""),
            rpc_methods=list(workflow_state.get("rpc_methods") or []),
            mixed_weights=dict(workflow_state.get("mixed_weights") or {}),
            observability_mode=str(confirmed.get("OBSERVABILITY_STACK_MODE") or confirmed.get("observability_choice_confirmed") or "disabled"),
            observability_enabled=str(confirmed.get("OBSERVABILITY_STACK_MODE") or "disabled") != "disabled",
            confirmations=_terminal_smoke_confirmations(confirmed),
        )
        data = prepared.get("data", {})
        preflight = data.get("preflight", {})
        if not preflight.get("passed"):
            update_workflow_state(
                {
                    "preflight_result": preflight,
                    "blockers": list(preflight.get("blockers") or prepared.get("warnings") or []),
                    "pending_question": data.get("configuration_questions", {}).get("next_question") or {},
                    "workflow_step": "preflight_blocked",
                    "allowed_next_actions": [],
                },
                reason="terminal_fake_node_smoke_preflight_blocked",
                session_id=self.session_id,
            )
            blockers = "; ".join(str(item) for item in (preflight.get("blockers") or prepared.get("warnings") or []))
            self.io.agent(self.state.language, _localized_simple(self.state.language, f"preflight 未通过：{blockers or '存在未满足配置'}", f"Preflight did not pass: {blockers or 'configuration is incomplete'}"))
            question = load_workflow_state(session_id=self.session_id).get("pending_question") or {}
            prompt = render_pending_question(question, language=self.state.language)
            if prompt:
                self.io.agent(self.state.language, prompt)
            return

        smoke = run_fake_node_smoke_benchmark(
            str(data.get("plan_file") or ""),
            approved=True,
        )
        job = ((smoke.get("data") or {}).get("job") or {})
        job_id = str(job.get("job_id") or "")
        if job_id:
            self.state.latest_job_id = job_id
        status = str(job.get("status") or smoke.get("status") or "unknown")
        if status not in {"completed", "running"}:
            update_workflow_state(
                {
                    "latest_plan_file": str((smoke.get("data") or {}).get("smoke_plan_file") or data.get("plan_file") or ""),
                    "latest_job_id": job_id,
                    "pending_question": {},
                    "allowed_next_actions": ["ask:logs", "ask:status", "ask:fix blockers"],
                    "workflow_step": "job_failed",
                    "active_workflow": "job_monitoring",
                    "preflight_result": preflight,
                    "smoke_result": smoke,
                },
                reason="terminal_fake_node_smoke_failed",
                session_id=self.session_id,
            )
            commands = (smoke.get("data") or {}).get("terminal_commands") or {}
            warnings = [item for item in smoke.get("warnings", []) if item]
            message = "\n".join(warnings) if warnings else "benchmark job did not start successfully"
            if self.state.language.startswith("zh"):
                self.io.agent(
                    self.state.language,
                    "fake-node smoke 未能启动或执行失败。"
                    f"\n- job_id: {job_id or '<none>'}"
                    f"\n- 状态: {status}"
                    f"\n- 原因: {message}"
                    f"\n- 日志: {commands.get('logs', f'logs {job_id}') if job_id else 'logs <job_id>'}",
                )
            else:
                self.io.agent(
                    self.state.language,
                    "fake-node smoke did not start or failed."
                    f"\n- job_id: {job_id or '<none>'}"
                    f"\n- status: {status}"
                    f"\n- reason: {message}"
                    f"\n- logs: {commands.get('logs', f'logs {job_id}') if job_id else 'logs <job_id>'}",
                )
            return
        update_workflow_state(
            {
                "latest_plan_file": str((smoke.get("data") or {}).get("smoke_plan_file") or data.get("plan_file") or ""),
                "latest_job_id": job_id,
                "pending_question": {},
                "allowed_next_actions": ["ask:status", "ask:logs", "ask:follow", "ask:analyze"],
                "workflow_step": "job_submitted",
                "active_workflow": "job_monitoring",
                "preflight_result": preflight,
                "smoke_result": smoke,
            },
            reason="terminal_fake_node_smoke_submitted",
            session_id=self.session_id,
        )
        commands = (smoke.get("data") or {}).get("terminal_commands") or {}
        if self.state.language.startswith("zh"):
            self.io.agent(
                self.state.language,
                "fake-node smoke 已提交。"
                f"\n- job_id: {job_id or '<unknown>'}"
                f"\n- 状态: {job.get('status', smoke.get('status', 'unknown'))}"
                f"\n- 日志: {commands.get('logs', f'logs {job_id}')}"
                f"\n- 实时日志: {commands.get('follow', f'follow {job_id}')}"
                f"\n- 分析: {commands.get('analyze', 'analyze latest job')}",
            )
        else:
            self.io.agent(
                self.state.language,
                "fake-node smoke submitted."
                f"\n- job_id: {job_id or '<unknown>'}"
                f"\n- status: {job.get('status', smoke.get('status', 'unknown'))}"
                f"\n- logs: {commands.get('logs', f'logs {job_id}')}"
                f"\n- follow: {commands.get('follow', f'follow {job_id}')}"
                f"\n- analyze: {commands.get('analyze', 'analyze latest job')}",
            )

    def _handle_pending_confirmation(self, lowered: str) -> bool:
        if self.state.current_question_id == "install_agent_runtime":
            if lowered in {"y", "yes", ""}:
                self._install_agent_runtime()
                return True
            if lowered in {"n", "no"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "agent_runtime_declined"))
                return True
        if self.state.current_question_id == "install_dependencies":
            if lowered in {"y", "yes", ""}:
                self._install_dependencies()
                return True
            if lowered in {"n", "no"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "dependency_declined"))
                return True
        return False

    def _ensure_bridge(self) -> ADKRunnerBridge:
        if self._bridge is None:
            self._bridge = self._bridge_factory()
        return self._bridge

    def _setup_question_after_adk(self, before_state: dict[str, Any]) -> dict[str, Any]:
        """Return the typed setup question that must own this model turn.

        ADK owns natural-language understanding. This guard does not infer
        intent from text; it only checks the persisted workflow state. If ADK
        entered benchmark setup without registering a typed pending question,
        the deterministic group engine registers the next blocking question so
        the user's next short answer has a real state target.
        """
        workflow_state = load_workflow_state(session_id=self.session_id)
        before_pending = before_state.get("pending_question") if isinstance(before_state, dict) else {}
        before_pending_id = str((before_pending or {}).get("id") or "")
        pending = workflow_state.get("pending_question") or {}
        pending_id = str(pending.get("id") or "")
        setup_state = (
            str(workflow_state.get("active_workflow") or "") == "benchmark_setup"
            or str(workflow_state.get("target_mode") or "") in {"fake-node", "real-node"}
        )
        if setup_state and pending and pending_id != before_pending_id:
            return {
                "message": render_pending_question(pending, language=self.state.language),
                "pending_question": pending,
                "suppress_response": True,
            }
        if workflow_state.get("pending_question"):
            return {"message": "", "pending_question": workflow_state.get("pending_question") or {}, "suppress_response": False}
        if str(workflow_state.get("active_workflow") or "") != "benchmark_setup" and str(workflow_state.get("target_mode") or "") not in {
            "fake-node",
            "real-node",
        }:
            return {"message": "", "pending_question": {}, "suppress_response": False}
        next_step = ensure_next_benchmark_setup_question(
            workflow_state,
            discovery=self.state.discovery,
            language=self.state.language,
            session_id=self.session_id,
        )
        message = str(next_step.get("message") or "").strip()
        return {
            "message": message,
            "pending_question": next_step.get("pending_question") or {},
            "suppress_response": bool(message),
        }

    def _state_delta(self, input_mode: str = "normal_user_turn") -> dict[str, Any]:
        return {
            "terminal_language": self.state.language,
            "input_mode": input_mode,
            "session_id": self.session_id,
            "framework_summary": self.state.framework_summary,
        }

    def _startup_doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "startup_doctor_start"))
        self._emit_doctor_summary(run_doctor(), startup=True)

    def _doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "doctor_start"))
        self._emit_doctor_summary(run_doctor(), startup=False)

    def _emit_doctor_summary(self, report: dict[str, Any], startup: bool) -> None:
        caps = report.get("capabilities", {})
        env = report.get("environment", {})
        self.state.discovery = env
        cloud = env.get("cloud", {})
        deployment = env.get("deployment", {})
        missing = env.get("dependencies", {}).get("missing_required", [])
        if not missing and self.state.current_question_id == "install_dependencies":
            self.state.current_question_id = ""
        self.state.pending_missing_dependencies = list(missing)
        if startup:
            self.io.agent(
                self.state.language,
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
                self.io.agent(self.state.language, t(self.state.language, "environment_inference_summary", summary=summary))
        else:
            self.io.agent(
                self.state.language,
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
            self.io.agent(self.state.language, t(self.state.language, "dependency_offer", missing=", ".join(missing)))

    def _install_agent_runtime(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "agent_runtime_install_start"))
        config = load_llm_config()
        command = ["bash", "scripts/install_agent_deps.sh", "--yes"]
        if (
            config.provider in {"gemini", "claude"}
            and config.auth_mode in {"google_adc", "service_account_impersonation"}
            and not shutil.which("gcloud")
        ):
            command.append("--with-gcloud")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.state.current_question_id = ""
        self.io.agent(self.state.language, t(self.state.language, "agent_runtime_install_done", exit_code=completed.returncode))
        self._adk_available = bool(adk_status().available and runner_bridge_status().available)
        if self._adk_available:
            self._bridge = None
            self._ensure_bridge()
        self._startup_doctor()

    def _install_dependencies(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "dependency_install_start"))
        completed = subprocess.run(
            ["bash", "scripts/install_deps.sh", "--yes"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.state.current_question_id = ""
        self.io.agent(self.state.language, t(self.state.language, "dependency_install_done", exit_code=completed.returncode))
        self._startup_doctor()

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
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_ID", DEFAULT_SESSION_ID),
        help="ADK/workflow session id. Tests and live matrices should isolate it per scenario.",
    )
    return parser.parse_args(argv)


def _format_pending_answer_blockers(language: str, blockers: list[Any]) -> str:
    rendered: list[str] = []
    zh = language.startswith("zh")
    for blocker in blockers:
        text = str(blocker)
        if text == "expected a chain name, not a numbered choice":
            rendered.append("当前需要输入链名，不能只输入编号" if zh else "Type a chain name; a bare number is not valid here.")
        elif text.startswith("expected a Linux block device"):
            rendered.append(
                "当前需要输入有效的 Linux 块设备名，例如 sdb、nvme1n1 或 /dev/sdd"
                if zh
                else "Type a valid Linux block device name, such as sdb, nvme1n1, or /dev/sdd."
            )
        elif text in {"expected yes or no for pending_question", "expected confirmation yes/no for pending_question"}:
            rendered.append("当前需要回复 Y 或 N" if zh else "Reply Y or N for the current question.")
        elif text == "yes requires pending_question.default_option for this choice":
            rendered.append(
                "当前问题没有默认选项，请回复选项编号或直接输入自定义值"
                if zh
                else "This question has no default option; choose an option number or type a custom value."
            )
        elif text == "answer does not match pending_question options":
            rendered.append(
                "这条回复不匹配当前选项，请回复选项编号或直接输入允许的自定义值"
                if zh
                else "That reply does not match the current options; choose an option number or type an allowed custom value."
            )
        elif text == "no active pending_question; ask what the user wants to confirm":
            rendered.append(
                "当前没有等待确认的问题，请直接说明你的目标"
                if zh
                else "There is no active question to confirm; describe your goal directly."
            )
        elif text.startswith("empty answer for pending_question"):
            rendered.append("请输入当前问题的答案" if zh else "Type an answer for the current question.")
        elif text.startswith("unsupported pending_question kind:"):
            rendered.append("当前问题类型无法处理，请重新说明你的目标" if zh else "This question type cannot be handled; restate your goal.")
        elif text.startswith("ACCOUNTS_DEVICE must be different from LEDGER_DEVICE"):
            rendered.append(
                "独立 accounts/state 磁盘不能和 LEDGER_DEVICE 使用同一块设备；如果没有独立 accounts/state 磁盘，请回退并选择 N。"
                if zh
                else "The separate accounts/state disk must be different from the ledger/data disk. If accounts/state uses the same disk, go back and choose N."
            )
        else:
            rendered.append(text)
    return "; ".join(rendered)


def _contains_token(text: str, token: str) -> bool:
    escaped = re.escape(token.lower())
    return bool(re.search(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", text))


def _effective_confirmed_config(workflow_state: dict[str, Any]) -> dict[str, Any]:
    confirmed = dict(workflow_state.get("confirmed_config") or {})
    chain = str(workflow_state.get("chain") or "").strip()
    if chain:
        confirmed.setdefault("chain", chain)
        confirmed.setdefault("BLOCKCHAIN_NODE", chain)
    rpc_mode = str(workflow_state.get("rpc_mode") or "").strip()
    if rpc_mode:
        confirmed.setdefault("rpc_mode", rpc_mode)
        confirmed.setdefault("RPC_MODE", rpc_mode)
    target_mode = str(workflow_state.get("target_mode") or "").strip()
    if target_mode == "fake-node":
        confirmed.setdefault("TARGET_MODE", "fake-node")
        confirmed.setdefault("use_fake_node", True)
        confirmed.setdefault("blockchain_process_names", ["fake-node"])
        confirmed.setdefault("BLOCKCHAIN_PROCESS_NAMES", ["fake-node"])
        confirmed.setdefault("BLOCKCHAIN_PROCESS_NAMES_STR", "fake-node")
    return confirmed


def _process_names_from_confirmed(confirmed: dict[str, Any]) -> list[str]:
    raw = (
        confirmed.get("blockchain_process_names")
        or confirmed.get("BLOCKCHAIN_PROCESS_NAMES")
        or confirmed.get("BLOCKCHAIN_PROCESS_NAMES_STR")
        or ["fake-node"]
    )
    if isinstance(raw, str):
        values = [item.strip() for item in re.split(r"[, ]+", raw) if item.strip()]
    else:
        values = [str(item).strip() for item in list(raw or []) if str(item).strip()]
    return values or ["fake-node"]


def _terminal_smoke_confirmations(confirmed: dict[str, Any]) -> list[str]:
    confirmations = {
        "benchmark_mode_confirmed",
        "qps_profile_confirmed",
        "observability_choice_confirmed",
        "chain_template_reviewed",
        "rpc_workload_confirmed",
        "rpc_workload_confirmation",
        "rpc_param_samples_confirmed",
        "rpc_param_samples_confirmation",
        "custom_rpc_method_review",
        "advanced_config_review",
        "disk_inventory_confirmation",
        "ledger_device_confirmation",
        "has_accounts_device",
        "blockchain_process_names",
        "data_vol_type",
        "data_vol_size",
        "data_vol_max_iops",
        "data_vol_max_throughput",
        "network_interface",
        "network_max_bandwidth_gbps",
    }
    if confirmed.get("ACCOUNTS_DEVICE") or confirmed.get("accounts_device"):
        confirmations.update({
            "accounts_device",
            "accounts_vol_type",
            "accounts_vol_size",
            "accounts_vol_max_iops",
            "accounts_vol_max_throughput",
        })
    return sorted(confirmations)


def _localized_simple(language: str, zh: str, en: str) -> str:
    return zh if (language or "en").startswith("zh") else en


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
        lines.append("- Disk candidates: <none detected; ADK should ask manually if needed>")
    interfaces = list(network.get("interfaces") or [])
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
    """Return true only for exact stable terminal commands.

    Natural-language requests that merely contain words such as "status" or
    "环境检查" must go to ADK for intent handling.
    """
    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases


def _format_memory(value: Any) -> str:
    if value in {None, ""}:
        return "<unknown>"
    return f"{value} GiB"


def _debug_exception(scope: str, exc: Exception) -> None:
    if os.environ.get("ANYCHAIN_DEBUG") not in {"1", "true", "TRUE", "yes", "YES"}:
        return
    message = redact(f"{type(exc).__name__}: {exc}")
    print(f"[anychain-debug] {scope}: {message}", file=sys.stderr)


def _adk_error_message(language: str, exc: Exception) -> str:
    message = str(exc).lower()
    if "insufficient balance" in message or "insufficient quota" in message:
        return t(language, "llm_billing_error")
    return t(language, "adk_runtime_error")


if __name__ == "__main__":
    raise SystemExit(main())
