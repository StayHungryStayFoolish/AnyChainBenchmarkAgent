#!/usr/bin/env python3
"""Guard AnyChain Agent against non-ADK routing regressions.

This check intentionally covers only mechanical boundaries. It does not judge
LLM quality; live model matrices do that. Keep this script small and targeted.
"""

from __future__ import annotations

import argparse
from pathlib import Path


LEGACY_AGENT_FILES = [
    "agent/adk_app/agents/router.py",
    "agent/adk_app/workflow/root_workflow.py",
    "agent/onboarding/request_answers.py",
    "agent/terminal/responder.py",
    "agent/workflows/benchmark_wizard.py",
    "agent/workflows/planning_bridge.py",
    "agent/workflows/state.py",
]

TERMINAL_FORBIDDEN = [
    "_looks_like_",
    "route_user_intent",
    "BenchmarkWizard",
    "planning_bridge",
    "terminal.responder",
    "request_answers",
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Check AnyChain Agent boundary regressions.")
    parser.add_argument("--root", default=".", help="Repository root.")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    failures: list[str] = []
    for rel in LEGACY_AGENT_FILES:
        if (root / rel).exists():
            failures.append(f"legacy Agent file must not exist: {rel}")

    terminal = root / "agent" / "terminal" / "repl.py"
    if terminal.exists():
        text = terminal.read_text(encoding="utf-8", errors="replace")
        for needle in TERMINAL_FORBIDDEN:
            if needle in text:
                failures.append(f"terminal must not contain business router marker {needle!r}: {terminal}")

    if failures:
        for item in failures:
            print(item)
        return 1
    print("agent boundary check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
