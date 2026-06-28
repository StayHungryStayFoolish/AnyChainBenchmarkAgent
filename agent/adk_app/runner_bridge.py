"""ADK Runner boundary for the product terminal.

The product terminal must not shell out to ``adk run``. This bridge owns the
optional in-process ADK Runner integration. When ADK is not installed, it
reports an unavailable status instead of falling back to the old custom Agent
brain.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any


warnings.filterwarnings(
    "ignore",
    message=r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL is enabled\.",
    category=UserWarning,
)


@dataclass(frozen=True)
class RunnerBridgeStatus:
    available: bool
    reason: str
    runner_import: str = "google.adk.runners.Runner"

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "runner_import": self.runner_import,
        }


def runner_bridge_status() -> RunnerBridgeStatus:
    try:
        from google.adk.runners import Runner  # type: ignore  # noqa: F401
    except Exception as exc:
        return RunnerBridgeStatus(False, f"ADK Runner is unavailable: {type(exc).__name__}: {exc}")
    return RunnerBridgeStatus(True, "ADK Runner is importable")


def require_runner_available() -> None:
    status = runner_bridge_status()
    if not status.available:
        raise RuntimeError(status.reason)


def run_text_once(
    text: str,
    state_delta: dict[str, Any] | None = None,
    user_id: str = "terminal-user",
    session_id: str = "terminal-session",
) -> str:
    """Run one text turn through the in-process ADK Runner.

    This is the product-terminal bridge to ADK. It deliberately avoids shelling
    out to ``adk run`` so the terminal can keep stable prompts, language
    behavior, workflow gates, and job recovery.
    """
    bridge = ADKRunnerBridge(user_id=user_id, session_id=session_id)
    return bridge.run_text(text, state_delta=state_delta)


class ADKRunnerBridge:
    """Small in-process ADK Runner wrapper with one persistent session."""

    def __init__(
        self,
        user_id: str = "terminal-user",
        session_id: str = "terminal-session",
        app_name: str = "anychain",
        turn_timeout_seconds: int | None = None,
    ) -> None:
        require_runner_available()
        try:
            from google.adk.runners import Runner  # type: ignore
            from google.adk.sessions import InMemorySessionService  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard.
            raise RuntimeError(f"ADK runtime imports failed: {type(exc).__name__}: {exc}") from exc

        from .root_agent import build_root_agent

        self.user_id = user_id
        self.session_id = session_id
        self.app_name = app_name
        self.turn_timeout_seconds = turn_timeout_seconds or _turn_timeout_seconds()
        self._session_service = InMemorySessionService()
        asyncio.run(
            self._session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
        )
        self._runner = Runner(
            app_name=app_name,
            agent=build_root_agent(),
            session_service=self._session_service,
        )

    def run_text(self, text: str, state_delta: dict[str, Any] | None = None) -> str:
        """Run one user message through the persistent ADK session."""
        try:
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard.
            raise RuntimeError(f"google.genai runtime import failed: {type(exc).__name__}: {exc}") from exc

        language = (state_delta or {}).get("terminal_language") or "en"
        message = types.Content(role="user", parts=[types.Part(text=_wrap_terminal_turn(text, state_delta))])
        run_kwargs: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "new_message": message,
        }
        if state_delta:
            run_kwargs["state_delta"] = state_delta
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            response = asyncio.run(asyncio.wait_for(self._run_text_async(run_kwargs), timeout=self.turn_timeout_seconds))
            if _terminal_contract_violations(response, language):
                rewrite_kwargs = {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "new_message": types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=_wrap_rewrite_turn(response, language),
                            )
                        ],
                    ),
                }
                response = asyncio.run(asyncio.wait_for(self._run_text_async(rewrite_kwargs), timeout=self.turn_timeout_seconds))
            return response

    async def _run_text_async(self, run_kwargs: dict[str, Any]) -> str:
        """Collect text from one async ADK Runner invocation."""
        output_parts: list[str] = []
        async for event in self._runner.run_async(**run_kwargs):
            content = getattr(event, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    output_parts.append(str(part_text))
        return sanitize_adk_text("\n".join(item.strip() for item in output_parts if item.strip()).strip())


def sanitize_adk_text(text: str) -> str:
    """Redact implementation leakage without routing or style patching.

    This is not an intent classifier and must not become a phrase-level patch
    system for model style. Natural-language behavior belongs in ADK
    instructions, sub-agent design, tool schemas, validators, and live tests.
    """
    blocked_fragments = (
        "prepare_benchmark_run",
        "draft_chain_template",
        "run_smoke",
        "submit_benchmark_job",
        "validate_required_config",
        "load_framework_context",
        "load_framework_index",
        "load_workflow_state",
        "update_workflow_state",
        "reset_workflow_state",
        "knowledge_search",
        "intent_router_agent",
        "execution_agent",
        "benchmark_configuration_agent",
        "chain_rpc_onboarding_agent",
        "sub-agent",
        "子代理",
        "sk-",
    )
    kept: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        visible_segments = []
        for segment in _split_visible_segments(raw_line):
            lowered = segment.lower()
            if any(fragment in lowered for fragment in blocked_fragments):
                continue
            visible_segments.append(segment.replace("**", ""))
        if visible_segments:
            kept.append(" ".join(visible_segments))
    return "\n".join(kept).strip()


def _wrap_terminal_turn(text: str, state_delta: dict[str, Any] | None) -> str:
    """Attach the product-terminal contract to one ADK turn.

    This is a prompt boundary, not an intent router. Business routing remains
    inside ADK and the configured model; this wrapper only reminds the model
    about terminal UX constraints that must hold for every natural-language
    turn.
    """
    language = (state_delta or {}).get("terminal_language") or "en"
    if language == "zh":
        contract = (
            "本轮终端输出契约：用中文回答；不要输出英文叙述；不要描述内部动作；"
            "可见回答禁止包含这些子串：我先、让我、让我先、让我看看、现在让我、我先查看、我先加载、调用工具；"
            "第一句必须直接回答用户问题或提出确认问题；"
            "如果输出上述禁用子串，本轮回答无效，必须先重写再返回；"
            "示例：不要说“我先检查框架能力”，直接说“可以，当前支持...”；"
            "不要说“你想先让我看看...”，直接问“是否准备运行 preflight 和 smoke？”；"
            "只给用户结论、选项、确认问题或证据路径；一次只问一个阻塞确认问题。"
        )
    else:
        contract = (
            "Terminal response contract for this turn: answer in English; do not narrate internal actions; "
            "the visible answer must not contain these substrings: let me, I need to, I should, I'll first, I will first, call a tool; "
            "the first visible sentence must directly answer the user or ask a confirmation question; "
            "if the answer contains those forbidden substrings, rewrite it before returning; "
            "invalid example: 'Let me check the framework capabilities'; valid example: 'Yes, the framework supports...'; "
            "show conclusions, options, "
            "confirmation questions, or evidence paths only; ask one blocking confirmation question at a time."
        )
    return (
        f"{contract}\n\n"
        f"User message:\n{text}\n\n"
        f"MANDATORY FINAL SELF-CHECK BEFORE RESPONDING: ensure the visible answer contains no forbidden substrings from the terminal contract. "
        f"The visible answer must start with the user-facing result or one confirmation question. "
        f"Do not describe checking, loading, inspecting, or starting first."
    )


def _wrap_rewrite_turn(text: str, language: str) -> str:
    """Ask ADK to repair a terminal response-contract violation once.

    This is a response guardrail, not business routing. It does not classify
    user intent or decide tools. The model keeps the same facts and rewrites
    only the visible answer to satisfy the terminal contract.
    """
    if language == "zh":
        return (
            "你的上一条可见回答违反了终端输出契约。不要调用工具，不要新增事实，"
            "只重写上一条回答。禁止包含：我先、让我、让我先、让我看看、现在让我、我先查看、我先加载、调用工具。\n\n"
            f"上一条回答：\n{text}"
        )
    return (
        "Your previous visible answer violated the terminal response contract. Do not call tools, do not add facts, "
        "only rewrite the previous answer. Forbidden substrings: let me, I need to, I should, I'll first, I will first, call a tool.\n\n"
        f"Previous answer:\n{text}"
    )


def _terminal_contract_violations(text: str, language: str) -> list[str]:
    """Return terminal response-contract violations without changing content."""
    if language == "zh":
        forbidden = ("我先", "让我", "让我先", "让我看看", "现在让我", "我先查看", "我先加载", "调用工具")
        return [item for item in forbidden if item in text]
    lowered = text.lower()
    forbidden = ("let me", "i need to", "i should", "i'll first", "i will first", "call a tool")
    return [item for item in forbidden if item in lowered]


def _split_visible_segments(line: str) -> list[str]:
    """Split a model line into displayable sentence-like segments.

    ADK models sometimes place a useful answer and internal implementation
    leakage in the same physical line. Sanitizing by segment preserves the
    useful answer while hiding tool names, sub-agent names, or secrets.
    """
    parts = re.split(r"(?<=[。！？!?.])\s*", line.strip())
    return [part for part in parts if part.strip()]


def _turn_timeout_seconds() -> int:
    raw = os.environ.get("ANYCHAIN_AGENT_TURN_TIMEOUT_SECONDS", "90").strip()
    try:
        value = int(raw)
    except ValueError:
        return 90
    return max(10, min(value, 600))
