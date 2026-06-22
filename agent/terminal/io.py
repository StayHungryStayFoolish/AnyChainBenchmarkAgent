"""Terminal input/output helpers.

The product CLI owns terminal interaction instead of delegating it to
``adk run``. ``prompt_toolkit`` is used when available because it handles
wide-character editing better than the basic Python readline path.
"""

from __future__ import annotations

from .language import t


class TerminalIO:
    def __init__(self) -> None:
        self._prompt_session = None
        try:
            from prompt_toolkit import PromptSession  # type: ignore

            self._prompt_session = PromptSession()
        except Exception:
            self._prompt_session = None

    def input(self, language: str) -> str:
        prompt = t(language, "prompt")
        if self._prompt_session is not None:
            return self._prompt_session.prompt(prompt)
        return input(prompt)

    def agent(self, language: str, message: str) -> None:
        print(t(language, "agent", message=message))
