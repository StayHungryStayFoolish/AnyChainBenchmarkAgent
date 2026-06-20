#!/usr/bin/env python3
"""Check for internal execution markers that should not appear in public code."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_ROOTS = (
    "README.md",
    "README_ZH.md",
    "agent",
    "analysis",
    "bin",
    "blockchain_node_benchmark.sh",
    "ci",
    "config",
    "core",
    "deploy",
    "docs",
    "lib",
    "monitoring",
    "scripts",
    "tests",
    "tools",
    "utils",
    "visualization",
)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "blockchain-node-benchmark-result",
    "fixtures",
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".svg",
    ".pyc",
}

MARKER_RE = re.compile(
    r"("
    r"\bS\d+(?:\.\d+)?\s+W\d+(?:\.\d+)?\b|"
    r"\bW\d+(?:\.\d+)?\s+(?:output|\u8f93\u51fa)\b|"
    r"\bCP-\d+\b|"
    r"\bY\+\b|"
    r"\b(?:wave|proposal|writer-first)\b|"
    r"hard gate|"
    r"\u65b9\u6848\u7532|\u65b9\u6848\u4e59|\u65b9\u6848\u4e19|\u4e2d\u7acb\u547d\u540d|\u9009\u7532|"
    r"\u786c\u95e8|\u5b88\u62a4|\u94c1\u5f8b|\u88c1\u51b3|\u6ce2\u6b21|\u8303\u5f0f|\u7ea0\u6b63|\u5b9e\u8bc1\u6765\u6e90|\u8fc1\u79fb\u53c2\u8003|"
    r"subagent|Hermes|Claude|Opus|Kiro|round-05|fix_wave|pre-S0|"
    r"plan §|§S\d+"
    r")",
    re.IGNORECASE,
)

RUNTIME_OUTPUT_CJK_RE = re.compile(
    r"\b(?:echo|printf|print|log_info|log_warn|log_error|log_debug|logger\.[a-zA-Z_]+|fmt\.Print(?:f|ln)?|fmt\.Fprint(?:f|ln)?)\b.*[\u4e00-\u9fff]"
)


def should_skip_path(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if path.name == "go.sum":
        return True
    if path.name == "check_public_repo_markers.py":
        return True
    if path.match("tools/proxy/proxy"):
        return True
    return False


def should_skip_line(path: Path, line: str) -> bool:
    stripped = line.strip()
    if '"target_address"' in stripped:
        return True
    if path.match("config/chains/*.json") and '"system_addresses"' in stripped:
        return True
    if "MISSING_PY" in stripped:
        return True
    if re.search(
        r"(vertex_claude|VertexClaudeProvider|Gemini/Claude|Gemini or Claude|Gemini, Claude|Gemini 或 Claude|"
        r"claude-\d|claude-opus|claude_|\"claude\"|`claude`|Claude partner|Claude API|Claude on Vertex|"
        r"Claude with Anthropic|Claude through Vertex|ANTHROPIC_API_KEY|AnthropicAPIKeyProvider)",
        stripped,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\bclaude\b", stripped, re.IGNORECASE) and re.search(
        r"(LLM_PROVIDER|provider|SUPPORTED_LLM_PROVIDERS|VertexClaudeProvider|"
        r"partner models?|model|GOOGLE_CLOUD_LOCATION|claude-\d|"
        r'"claude"|`claude`|vertex_claude)',
        stripped,
        re.IGNORECASE,
    ):
        return True
    return False


def iter_files(root: Path, roots: tuple[str, ...]):
    for item in roots:
        p = root / item
        if not p.exists():
            continue
        if p.is_file():
            if not should_skip_path(p):
                yield p
            continue
        for child in p.rglob("*"):
            if child.is_file() and not should_skip_path(child):
                yield child


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("paths", nargs="*", help="optional paths to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    roots = tuple(args.paths) if args.paths else DEFAULT_ROOTS
    findings: list[tuple[Path, int, str]] = []

    for path in iter_files(root, roots):
        rel = path.relative_to(root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for line_no, line in enumerate(lines, 1):
            if should_skip_line(rel, line):
                continue
            if MARKER_RE.search(line):
                findings.append((rel, line_no, line.strip()))
                continue
            if RUNTIME_OUTPUT_CJK_RE.search(line):
                findings.append((rel, line_no, line.strip()))

    if findings:
        for rel, line_no, line in findings:
            print(f"{rel}:{line_no}: {line}")
        print(f"\nFound {len(findings)} public-repo marker(s).", file=sys.stderr)
        return 1

    print("public repo marker check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
