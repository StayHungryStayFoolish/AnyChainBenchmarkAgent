"""Local framework Q&A grounded in repository documentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "README.md",
    "README_ZH.md",
    "config/README.md",
    "agent/README.md",
    "docs/en/framework-flow.md",
    "docs/en/module-guide.md",
    "docs/en/how-to-add-chain.md",
    "docs/en/local-closed-loop-testing.md",
    "docs/zh/framework-flow.md",
    "docs/zh/module-guide.md",
    "docs/zh/how-to-add-chain.md",
    "docs/zh/local-closed-loop-testing.md",
)


def answer_framework_question(question: str, limit: int = 5) -> dict[str, Any]:
    tokens = _tokens(question)
    matches = []
    for rel in SOURCE_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(lines, 1):
            score = _score(tokens, line)
            if score <= 0:
                continue
            matches.append({
                "score": score,
                "path": str(path),
                "line": idx,
                "text": line.strip(),
            })
    matches.sort(key=lambda item: item["score"], reverse=True)
    sources = matches[:limit]
    if not sources:
        return {
            "intent": "framework_question",
            "answer": "I could not find a grounded answer in the local framework documentation.",
            "confidence": 0.2,
            "sources": [],
        }
    answer = _compose_answer(question, sources)
    return {
        "intent": "framework_question",
        "answer": answer,
        "confidence": min(0.9, 0.45 + 0.08 * len(sources)),
        "sources": sources,
    }


def out_of_scope_response(question: str) -> dict[str, Any]:
    return {
        "intent": "out_of_scope",
        "answer": (
            "This Agent is scoped to blockchain node benchmarking, fake-node closed-loop tests, "
            "monitoring, reports, and related configuration. The request appears outside that scope."
        ),
        "confidence": 0.8,
        "sources": [],
    }


def _compose_answer(question: str, sources: list[dict[str, Any]]) -> str:
    topic = question.strip()
    bullets = []
    for source in sources[:3]:
        text = source["text"]
        if len(text) > 180:
            text = text[:177] + "..."
        bullets.append(f"- {text} ({source['path']}:{source['line']})")
    return "Grounded local documentation matches for: " + topic + "\n" + "\n".join(bullets)


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]{2,}", text.lower())
    stop = {"the", "and", "or", "how", "what", "where", "why", "to", "a", "an", "is", "are"}
    return {item for item in raw if item not in stop and len(item) >= 2}


def _score(tokens: set[str], line: str) -> int:
    lowered = line.lower()
    score = 0
    for token in tokens:
        if token in lowered:
            score += 2 if len(token) > 4 else 1
    if line.startswith("#"):
        score += 1
    return score
