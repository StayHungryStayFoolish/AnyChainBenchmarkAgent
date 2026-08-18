"""Regression tests for the canonical ``agent.*`` package identity."""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = REPO_ROOT / "agent"
INTERNAL_ROOTS = {
    "analyzers",
    "diagnostics",
    "discovery",
    "harness",
    "knowledge",
    "llm",
    "onboarding",
    "planners",
    "runners",
    "terminal",
    "tools",
    "utils",
    "validators",
    "workflows",
}

EXCLUDED_PATHS: set[str] = set()

PUBLIC_DOCS = (
    "README.md",
    "README_ZH.md",
    "AGENTS.md",
    "agent/README.md",
    "config/README.md",
    "docs/en/agent-cli-verification-guide.md",
    "docs/en/secondary-development-guide.md",
    "docs/zh/secondary-development-guide.md",
)


class AgentPackageIdentityTests(unittest.TestCase):
    def test_non_excluded_modules_do_not_import_internal_top_level_packages(self) -> None:
        violations: list[str] = []
        for path in sorted(AGENT_ROOT.rglob("*.py")):
            relative = path.relative_to(AGENT_ROOT).as_posix()
            if relative in EXCLUDED_PATHS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                root = node.module.split(".", 1)[0]
                if root in INTERNAL_ROOTS:
                    violations.append(f"{relative}:{node.lineno}: from {node.module}")
        self.assertEqual([], violations)

    def test_programmatic_cli_runs_as_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "agent.cli", "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("AnyChain Benchmark Agent control-plane CLI", result.stdout)

    def test_public_docs_do_not_advertise_script_execution(self) -> None:
        stale = []
        for relative in PUBLIC_DOCS:
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            if "python3 agent/cli.py" in text:
                stale.append(relative)
        self.assertEqual([], stale)


if __name__ == "__main__":
    unittest.main()
