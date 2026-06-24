"""Terminal input/output helpers.

The product CLI owns terminal interaction instead of delegating it to
``adk run``. ``prompt_toolkit`` is required because reliable Ctrl+C handling
and wide-character editing are baseline Agent terminal requirements.
"""

from __future__ import annotations

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

    def input(self, language: str) -> str:
        prompt = t(language, "prompt")
        return self._prompt_session.prompt(prompt)

    def agent(self, language: str, message: str) -> None:
        print(t(language, "agent", message=message))


class OutputOnlyIO:
    """Non-interactive IO for scripted prompts and tests."""

    def input(self, language: str) -> str:
        raise EOFError()

    def agent(self, language: str, message: str) -> None:
        print(t(language, "agent", message=message))
