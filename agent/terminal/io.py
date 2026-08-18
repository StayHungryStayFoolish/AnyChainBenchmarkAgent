"""Terminal input/output helpers.

The product CLI owns terminal interaction instead of delegating it to
``adk run``. ``prompt_toolkit`` is required because reliable Ctrl+C handling
and wide-character editing are baseline Agent terminal requirements.
"""

from __future__ import annotations

from ..harness.terminal_protocol import presentation_hash
from .language import t


class TerminalIO:
    def __init__(self) -> None:
        try:
            from prompt_toolkit import PromptSession  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "prompt-toolkit is required for AnyChain Agent terminal input. "
                "Run: bash scripts/install_agent_deps.sh --yes"
            ) from exc

        self._prompt_session = PromptSession()
        self._frame_fragments: list[str] = []
        self._session_fragments: list[str] = []

    def input(self, language: str) -> str:
        prompt = t(language, "prompt")
        return self._prompt_session.prompt(prompt)

    def begin_frame(self) -> None:
        self._frame_fragments = []

    def begin_session(self) -> None:
        self._session_fragments = []

    def agent(self, language: str, message: str) -> str:
        rendered = t(language, "agent", message=message)
        self._frame_fragments.append(rendered)
        self._session_fragments.append(rendered)
        print(rendered, flush=True)
        return rendered

    def rendered_frame(self) -> str:
        return "\n".join(getattr(self, "_frame_fragments", ())).strip()

    def presentation_hash(self) -> str:
        return presentation_hash(self.rendered_frame())

    def rendered_session(self) -> str:
        return "\n".join(getattr(self, "_session_fragments", ())).strip()


class OutputOnlyIO:
    """Non-interactive IO for scripted prompts and tests."""

    def input(self, language: str) -> str:
        raise EOFError()

    def __init__(self) -> None:
        self._frame_fragments: list[str] = []
        self._session_fragments: list[str] = []

    def begin_frame(self) -> None:
        self._frame_fragments = []

    def begin_session(self) -> None:
        self._session_fragments = []

    def agent(self, language: str, message: str) -> str:
        rendered = t(language, "agent", message=message)
        self._frame_fragments.append(rendered)
        self._session_fragments.append(rendered)
        print(rendered, flush=True)
        return rendered

    def rendered_frame(self) -> str:
        return "\n".join(getattr(self, "_frame_fragments", ())).strip()

    def presentation_hash(self) -> str:
        return presentation_hash(self.rendered_frame())

    def rendered_session(self) -> str:
        return "\n".join(getattr(self, "_session_fragments", ())).strip()
