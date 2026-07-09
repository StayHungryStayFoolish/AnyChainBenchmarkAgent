"""Terminal-visible events emitted by the LangGraph Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HarnessEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


def message(text: str) -> HarnessEvent:
    return HarnessEvent(type="message", payload={"text": text})


def question(payload: dict[str, Any]) -> HarnessEvent:
    return HarnessEvent(type="question", payload=payload)

