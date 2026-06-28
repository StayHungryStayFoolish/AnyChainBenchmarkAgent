#!/usr/bin/env python3
"""Run live AnyChain Agent product-terminal prompt matrices.

This runner is intentionally separate from normal unit tests. It requires a
real model credential when ``--require-live`` is used and writes redacted logs
under /tmp by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = Path(__file__).with_name("agent_intent_smoke_scenarios.json")
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live Agent prompt matrix against the product terminal.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--log-dir", default="/tmp/anychain-agent-live-matrix")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--agent-python", default=os.environ.get("ANYCHAIN_AGENT_PYTHON", ""))
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "LLM_PROVIDER": args.provider,
        "LLM_MODEL": args.model,
        "LLM_AUTH_MODE": "api_key",
        "ANYCHAIN_AGENT_TURN_TIMEOUT_SECONDS": str(max(10, min(args.timeout, 600))),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    })
    if args.agent_python:
        env["ANYCHAIN_AGENT_PYTHON"] = args.agent_python
    if args.require_live and args.provider == "deepseek" and not env.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is required for --require-live", file=sys.stderr)
        return 2

    default_forbidden = matrix.get("default_forbidden_regex", [])
    results: list[dict[str, Any]] = []
    failures = 0
    for scenario in matrix.get("scenarios", []):
        result = run_scenario(scenario, default_forbidden, log_dir, env, args.timeout)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['name']} -> {result['log_file']}")
        if not result["passed"]:
            failures += 1
            for issue in result["issues"]:
                print(f"  - {issue}")

    summary = {
        "matrix": matrix.get("name", Path(args.matrix).name),
        "provider": args.provider,
        "model": args.model,
        "scenario_count": len(results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "failed_count": failures,
        "results": results,
    }
    summary_file = log_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary={summary_file}")
    return 0 if failures == 0 else 1


def run_scenario(
    scenario: dict[str, Any],
    default_forbidden: list[str],
    log_dir: Path,
    env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    name = scenario["name"]
    state_file = log_dir / f"{name}.state.json"
    session_id = f"live-{re.sub(r'[^A-Za-z0-9_-]+', '_', name)}"
    raw_log = log_dir / f"{name}.raw.log"
    redacted_log = log_dir / f"{name}.log"
    prompts = scenario.get("prompts", [])
    if not prompts and scenario.get("prompt"):
        prompts = [scenario["prompt"]]
    command = ["./bin/anychain-agent", "--state-file", str(state_file), "--session-id", session_id]
    language = scenario.get("language")
    if language:
        command.extend(["--language", language])
    for prompt in prompts:
        command.extend(["--prompt", str(prompt)])

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = round(time.time() - started, 3)
    raw = completed.stdout
    redacted = SECRET_RE.sub("<redacted>", raw)
    raw_log.write_text(redacted, encoding="utf-8")
    redacted_log.write_text(redacted, encoding="utf-8")

    issues: list[str] = []
    if completed.returncode != 0:
        issues.append(f"exit_code={completed.returncode}")
    for needle in scenario.get("must_contain", []):
        if needle not in redacted:
            issues.append(f"missing required text: {needle}")
    any_needles = [needle for needle in scenario.get("must_include_any", []) if needle]
    if any_needles and not any(needle in redacted for needle in any_needles):
        issues.append(f"missing any required text: {', '.join(any_needles)}")
    for needle in scenario.get("must_not_contain", []):
        if needle and needle in redacted:
            issues.append(f"forbidden text present: {needle}")
    for pattern in default_forbidden + scenario.get("forbidden_regex", []) + scenario.get("must_not_include_regex", []):
        if re.search(pattern, redacted, flags=re.IGNORECASE):
            issues.append(f"forbidden regex matched: {pattern}")
    # Scripted prompt mode intentionally uses OutputOnlyIO and does not echo
    # ``User>`` lines. Scenario-specific positive assertions are therefore the
    # source of truth for whether the model answered the prompt.

    return {
        "name": name,
        "passed": not issues,
        "issues": issues,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "log_file": str(redacted_log),
    }


if __name__ == "__main__":
    raise SystemExit(main())
