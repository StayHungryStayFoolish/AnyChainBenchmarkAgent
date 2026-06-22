"""Persistent workflow state for product Agent sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_FILE = REPO_ROOT / ".agent" / "session" / "terminal_state.json"


@dataclass
class WorkflowState:
    language: str = "en"
    intent: str = ""
    stage: str = "start"
    current_question_id: str = ""
    confirmed_values: dict[str, Any] = field(default_factory=dict)
    defaulted_values: dict[str, Any] = field(default_factory=dict)
    skipped_values: dict[str, Any] = field(default_factory=dict)
    missing_blockers: list[str] = field(default_factory=list)
    pending_confirmations: list[str] = field(default_factory=list)
    plan_file: str = ""
    runtime_env_file: str = ""
    job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowState":
        fields = cls().__dict__.keys()
        return cls(**{key: value for key, value in payload.items() if key in fields})


class WorkflowStateStore:
    def __init__(self, path: str | Path = DEFAULT_STATE_FILE) -> None:
        self.path = Path(path)

    def load(self) -> WorkflowState:
        if not self.path.is_file():
            return WorkflowState()
        try:
            return WorkflowState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return WorkflowState()

    def save(self, state: WorkflowState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
