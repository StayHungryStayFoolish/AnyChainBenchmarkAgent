"""Turn adjudication primitives for the AnyChain Agent Harness.

This module intentionally does not parse business intent. It only names the
outer turn shape so the compiled LangGraph keeps a clear control boundary.
Ambiguous natural language remains the responsibility of the LLM action
resolver and deterministic group workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TurnKind = Literal[
    "empty",
    "evidence_continuation",
    "pending",
    "free_text",
]


@dataclass(frozen=True)
class TurnAdjudication:
    kind: TurnKind
    has_pending_question: bool
    evidence_collection_active: bool
    pending_question_id: str = ""
    pending_kind: str = ""


def adjudicate_turn(state: dict[str, Any], text: str) -> TurnAdjudication:
    """Return the non-mutating outer turn category.

    The result is deliberately small. Do not add chain, mode, RPC, QPS, or
    product-specific phrase checks here; those belong to LLM action resolution
    or deterministic group handlers.
    """

    pending = state.get("pending_question") or {}
    collecting = state.get("evidence_collection") or {}
    pending_id = str(pending.get("id") or "").strip()
    pending_kind = str(pending.get("kind") or "").strip()
    if collecting and str(collecting.get("status") or "active") == "active":
        return TurnAdjudication(
            kind="evidence_continuation",
            has_pending_question=bool(pending),
            evidence_collection_active=True,
            pending_question_id=pending_id,
            pending_kind=pending_kind,
        )
    if not str(text or "").strip():
        return TurnAdjudication(
            kind="empty",
            has_pending_question=bool(pending),
            evidence_collection_active=False,
            pending_question_id=pending_id,
            pending_kind=pending_kind,
        )
    if pending:
        return TurnAdjudication(
            kind="pending",
            has_pending_question=True,
            evidence_collection_active=False,
            pending_question_id=pending_id,
            pending_kind=pending_kind,
        )
    return TurnAdjudication(
        kind="free_text",
        has_pending_question=False,
        evidence_collection_active=False,
    )
