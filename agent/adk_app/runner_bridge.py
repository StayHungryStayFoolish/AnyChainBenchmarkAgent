"""ADK Runner boundary for the product terminal.

The product terminal must not shell out to ``adk run``. This bridge owns the
optional in-process ADK Runner integration. When ADK is not installed, it
reports an unavailable status instead of falling back to the old custom Agent
brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    require_runner_available()
    try:
        from google.adk.runners import Runner  # type: ignore
        from google.adk.sessions import InMemorySessionService  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard.
        raise RuntimeError(f"ADK runtime imports failed: {type(exc).__name__}: {exc}") from exc

    from .root_agent import build_root_agent

    session_service = InMemorySessionService()
    session_service.create_session_sync(app_name="anychain", user_id=user_id, session_id=session_id)
    runner = Runner(
        app_name="anychain",
        agent=build_root_agent(),
        session_service=session_service,
    )
    message = types.Content(role="user", parts=[types.Part(text=text)])
    output_parts: list[str] = []
    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
        state_delta=state_delta or {},
    ):
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                output_parts.append(str(part_text))
    return "\n".join(item.strip() for item in output_parts if item.strip()).strip()
