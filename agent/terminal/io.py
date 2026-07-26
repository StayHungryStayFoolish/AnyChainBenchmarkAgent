"""Terminal input/output helpers.

The product CLI owns terminal interaction instead of delegating it to
``adk run``. ``prompt_toolkit`` is preferred for richer terminal behavior,
but we keep a deterministic fallback for environments without it (eg.
some harness/PTY sessions).
"""

from __future__ import annotations

import os
import sys

from .language import t


class TerminalIO:
    def __init__(self) -> None:
        self._prompt_session = None
        self._prompt_toolkit_available = False
        try:
            from prompt_toolkit import PromptSession  # type: ignore

            self._prompt_session = PromptSession()
            self._prompt_toolkit_available = True
        except Exception:
            # Keep CLI usable even when prompt_toolkit is not present, so PTY
            # and integration tests can still exercise contract behavior.
            self._prompt_session = None
            self._prompt_toolkit_available = False

    def input(self, language: str) -> str:
        prompt = t(language, "prompt")
        if self._prompt_toolkit_available and self._prompt_session is not None:
            return self._prompt_session.prompt(prompt)

        print(prompt, end="", flush=True)
        raw = self._read_line_bytes(sys.stdin.fileno())
        cleaned = self._sanitize_input(raw)
        return cleaned

    @staticmethod
    def _read_line_bytes(fd: int) -> str:
        if not hasattr(sys.stdin, "buffer"):
            raise EOFError()

        start = b"\x1b[200~"
        end = b"\x1b[201~"
        in_bracketed = False
        payload: list[bytes] = []
        while True:
            chunk = os.read(fd, 1)
            if not chunk:
                break
            payload.append(chunk)
            buf = b"".join(payload)
            if in_bracketed:
                if buf.endswith(end):
                    break
                continue

            if buf.endswith(start):
                payload = payload[: -len(start)]
                in_bracketed = True
                continue

            if chunk in {b"\r", b"\n"}:
                break

        if not payload:
            raise EOFError()

        return b"".join(payload).decode("utf-8", errors="replace")

    @staticmethod
    def _sanitize_input(raw: str) -> str:
        start_marker = "\x1b[200~"
        end_marker = "\x1b[201~"
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")

        start_index = normalized.find(start_marker)
        if start_index < 0:
            return normalized

        before_marker = normalized[:start_index]
        remaining = normalized[start_index + len(start_marker) :]
        end_index = remaining.find(end_marker)
        if end_index < 0:
            return before_marker + remaining

        pasted = remaining[:end_index]
        trailing = remaining[end_index + len(end_marker) :]
        if trailing:
            # If bracketed paste is immediately followed by content,
            # keep anything after the end marker (typically newline from PTY).
            return before_marker + pasted + trailing
        return before_marker + pasted

    def agent(self, language: str, message: str) -> None:
        print(t(language, "agent", message=message))


class OutputOnlyIO:
    """Non-interactive IO for scripted prompts and tests."""

    def input(self, language: str) -> str:
        raise EOFError()

    def agent(self, language: str, message: str) -> None:
        print(t(language, "agent", message=message))
