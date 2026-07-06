"""ADK Runner boundary for the product terminal.

The product terminal must not shell out to ``adk run``. This bridge owns the
in-process ADK Runner integration. When ADK is not installed, it reports an
unavailable status and lets the terminal ask for runtime installation.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import warnings
from dataclasses import dataclass
from typing import Any

from .instructions import build_terminal_turn_prompt
from utils.redaction import redact
from workflows.conversation_state import load_workflow_state


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
        from .tools.workflow_state import workflow_tool_session

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
        self._workflow_tool_session = workflow_tool_session

    def run_text(self, text: str, state_delta: dict[str, Any] | None = None) -> str:
        """Run one user message through the persistent ADK session."""
        try:
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard.
            raise RuntimeError(f"google.genai runtime import failed: {type(exc).__name__}: {exc}") from exc

        turn_state = self._turn_state(state_delta)
        turn_state["user_text"] = text
        message = types.Content(role="user", parts=[types.Part(text=build_terminal_turn_prompt(text, turn_state))])
        run_kwargs: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "new_message": message,
        }
        run_kwargs["state_delta"] = turn_state
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            with self._workflow_tool_session(self.session_id):
                raw_response = self._run_once_with_retry(run_kwargs)
            return sanitize_adk_text(raw_response)

    def _turn_state(self, state_delta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build one ADK turn state from transport metadata and file-backed state.

        The terminal must not inject full benchmark workflow state. Until ADK
        mutable session state is proven stable for this runtime, the
        file-backed conversation state is the single operational source and the
        bridge loads it by session_id.
        """
        incoming = dict(state_delta or {})
        session_id = str(incoming.get("session_id") or self.session_id)
        incoming["session_id"] = session_id
        incoming.setdefault("terminal_language", "en")
        incoming.setdefault("input_mode", "normal_user_turn")
        incoming["workflow_state"] = load_workflow_state(session_id=session_id)
        return incoming

    async def _run_text_async(self, run_kwargs: dict[str, Any]) -> str:
        """Collect text from one async ADK Runner invocation."""
        output_parts: list[str] = []
        async for event in self._runner.run_async(**run_kwargs):
            content = getattr(event, "content", None)
            if not content:
                continue
            parts = list(getattr(content, "parts", []) or [])
            if _event_contains_tool_exchange(parts):
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    output_parts.append(str(part_text))
        raw_text = "\n".join(item.strip() for item in output_parts if item.strip()).strip()
        return sanitize_adk_text(raw_text)

    def _run_once_with_timeout(self, run_kwargs: dict[str, Any]) -> str:
        return asyncio.run(asyncio.wait_for(self._run_text_async(run_kwargs), timeout=self.turn_timeout_seconds))

    def _run_once_with_retry(self, run_kwargs: dict[str, Any]) -> str:
        try:
            return self._run_once_with_timeout(run_kwargs)
        except (asyncio.TimeoutError, TimeoutError):
            raise
        except Exception:
            return self._run_once_with_timeout(run_kwargs)

def sanitize_adk_text(text: str) -> str:
    """Apply mechanical terminal-safe cleanup to ADK text chunks.

    This function must not classify intent, infer workflow state, rewrite model
    prose, or filter process narration with phrase lists. Visible contract
    quality must come from prompts, ADK callbacks, workflow tools, and typed
    workflow state rather than a second hidden LLM rewrite turn.
    """
    text = _extract_visible_response(text)
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
        kept.append(raw_line.replace("**", "").strip())
    return str(redact("\n".join(kept).strip()))


def _event_contains_tool_exchange(parts: list[Any]) -> bool:
    """Return whether an ADK event is part of tool planning/execution.

    ADK/LiteLLM providers can emit explanatory text in the same event as a
    function_call. That text is intermediate model work, not terminal-visible
    output. The product terminal should collect only final text-only events.
    """
    for part in parts:
        if getattr(part, "function_call", None) is not None:
            return True
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def _extract_visible_response(text: str) -> str:
    """Return the model-declared terminal-visible section when present.

    The marker is an output envelope, not a business intent rule. It prevents
    model scratchpad/tool planning text from becoming terminal output while
    preserving the requirement that routing stays in ADK agents and tools.
    """
    marker = "VISIBLE_RESPONSE:"
    if marker not in text:
        return text
    return text.rsplit(marker, 1)[1].strip()

def _turn_timeout_seconds() -> int:
    raw = os.environ.get("ANYCHAIN_AGENT_TURN_TIMEOUT_SECONDS", "90").strip()
    try:
        value = int(raw)
    except ValueError:
        return 90
    return max(10, min(value, 600))
