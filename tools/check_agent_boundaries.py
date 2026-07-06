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
    "agent/adk_app/terminal_contract.py",
]

TERMINAL_FORBIDDEN = [
    "_looks_like_",
    "route_user_intent",
    "BenchmarkWizard",
    "planning_bridge",
    "terminal.responder",
    "request_answers",
]

RUNNER_BRIDGE_FORBIDDEN = [
    "terminal_presenter",
    "present_terminal_text",
    "_run_with_runner_retry",
    "_format_terminal_envelope_if_needed",
    "_needs_terminal_envelope_formatting",
    "_contains_process_narration",
    "_terminal_formatter_system_prompt",
    "provider.complete(",
    "process_openers",
    "_LEADING_PROCESS",
    "language == \"zh\"",
    "language == 'zh'",
]

WORKFLOW_TOOL_FORBIDDEN = [
    "_safe_discovery_summary",
    "process_openers",
    "_LEADING_PROCESS",
]

PRODUCT_TEST_FORBIDDEN = [
    "removes_leading_process_narration",
    "removes_chinese_leading_process_narration",
    "removes_chinese_handle_process_narration",
]

PUBLIC_SCHEMA_FORBIDDEN = [
    '"mock"',
    "lifecycle-only mock",
]

CLI_FORBIDDEN = [
    '"--mock"',
    "'--mock'",
    'sub.add_parser("chat"',
    "sub.add_parser('chat'",
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

    presenter = root / "agent" / "adk_app" / "terminal_presenter.py"
    if presenter.exists():
        failures.append(f"obsolete terminal presenter file must not exist: {presenter}")

    runner_bridge = root / "agent" / "adk_app" / "runner_bridge.py"
    if runner_bridge.exists():
        text = runner_bridge.read_text(encoding="utf-8", errors="replace")
        for needle in RUNNER_BRIDGE_FORBIDDEN:
            if needle in text:
                failures.append(f"runner bridge must not call terminal presenter rewrite marker {needle!r}: {runner_bridge}")

    workflow_tools = root / "agent" / "adk_app" / "tools" / "workflow_state.py"
    if workflow_tools.exists():
        text = workflow_tools.read_text(encoding="utf-8", errors="replace")
        for needle in WORKFLOW_TOOL_FORBIDDEN:
            if needle in text:
                failures.append(f"workflow-state tools must not contain output phrase-repair marker {needle!r}: {workflow_tools}")

    product_terminal_tests = root / "tests" / "test_agent_product_terminal.py"
    if product_terminal_tests.exists():
        text = product_terminal_tests.read_text(encoding="utf-8", errors="replace")
        for needle in PRODUCT_TEST_FORBIDDEN:
            if needle in text:
                failures.append(f"product terminal tests must not assert phrase-repair behavior {needle!r}: {product_terminal_tests}")

    public_schema = root / "agent" / "tools" / "schema.py"
    if public_schema.exists():
        text = public_schema.read_text(encoding="utf-8", errors="replace")
        for needle in PUBLIC_SCHEMA_FORBIDDEN:
            if needle in text:
                failures.append(f"public Agent tool schema must not expose mock execution marker {needle!r}: {public_schema}")

    cli = root / "agent" / "cli.py"
    if cli.exists():
        text = cli.read_text(encoding="utf-8", errors="replace")
        for needle in CLI_FORBIDDEN:
            if needle in text:
                failures.append(f"Agent CLI must use explicit developer mock naming, not {needle!r}: {cli}")

    if failures:
        for item in failures:
            print(item)
        return 1
    print("agent boundary check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
