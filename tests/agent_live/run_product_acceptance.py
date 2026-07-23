"""Single product-readiness authority for AnyChain Agent.

Provider scripts may generate raw evidence, but only this controller can admit
that evidence and advance a phase or G0-G6 gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_live.generate_harness_coverage_ledger import build_ledger


SCHEMA_VERSION = 1
IMPLEMENTED_THROUGH_PHASE = 2


def _run(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "passed": completed.returncode == 0,
    }


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _g0_source() -> dict[str, Any]:
    checks = [
        _run((sys.executable, "tools/check_agent_boundaries.py", "--root", ".")),
        _run((
            sys.executable,
            "-m",
            "unittest",
            "tests.test_agent_product_terminal",
            "tests.test_agent_runtime_contract",
            "tests.test_agent_langgraph_harness",
        )),
        _run(("git", "diff", "--check")),
    ]
    tracked_status = _git_output("status", "--porcelain", "--untracked-files=all")
    linux = platform.system() == "Linux"
    return {
        "gate": "G0",
        "revision": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "platform": platform.platform(),
        "linux": linux,
        "tracked_worktree_clean": not tracked_status,
        "tracked_status": tracked_status.splitlines(),
        "checks": checks,
        "status": (
            "passed"
            if linux and not tracked_status and all(check["passed"] for check in checks)
            else "failed"
        ),
    }


def _phase2_source() -> dict[str, Any]:
    from agent.harness import coordinator
    from agent.harness.hierarchical_planner import resolve_product_action_queue

    checks = [
        _run((
            sys.executable,
            "-m",
            "unittest",
            "tests.test_agent_hierarchical_planner",
            "tests.test_agent_plan_coverage",
            "tests.test_agent_pending_choice_canonicalization",
            "tests.test_agent_planner_risk",
        )),
    ]
    product_entry_is_hierarchical = (
        coordinator.resolve_action_queue is resolve_product_action_queue
    )
    return {
        "phase": 2,
        "product_entry_is_hierarchical": product_entry_is_hierarchical,
        "checks": checks,
        "status": (
            "passed"
            if product_entry_is_hierarchical
            and all(check["passed"] for check in checks)
            else "failed"
        ),
    }


def build_report(through_phase: int) -> dict[str, Any]:
    inventory = build_ledger(None)
    g0 = _g0_source()
    phase_checks = {
        "1": {"status": "passed"},
        "2": _phase2_source(),
    }
    requested_supported = through_phase <= IMPLEMENTED_THROUGH_PHASE
    requested_phase_checks_pass = all(
        phase_checks[str(phase)]["status"] == "passed"
        for phase in range(1, min(through_phase, IMPLEMENTED_THROUGH_PHASE) + 1)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "tests/agent_live/run_product_acceptance.py",
        "through_phase": through_phase,
        "implemented_through_phase": IMPLEMENTED_THROUGH_PHASE,
        "provider_authority": "raw_evidence_only",
        "phase_checks": phase_checks,
        "inventory": {
            "revision": inventory["revision"],
            "summary": inventory["summary"],
            "uncataloged_questions": inventory["uncataloged_questions"],
        },
        "gates": {
            "G0": g0,
            "G1": {"status": "not_run", "owning_phase": 3},
            "G2": {"status": "not_run", "owning_phase": 6},
            "G3": {"status": "not_run", "owning_phase": 6},
            "G4": {"status": "not_run", "owning_phase": 8},
            "G5": {"status": "not_run", "owning_phase": 8},
            "G6": {"status": "not_run", "owning_phase": 8},
        },
        "status": (
            "passed"
            if requested_supported
            and g0["status"] == "passed"
            and requested_phase_checks_pass
            else "incomplete"
            if not requested_supported
            else "failed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-phase", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.through_phase)
    output = args.output or (
        REPO_ROOT
        / ".agent"
        / "evidence"
        / "control-plane"
        / str(report["inventory"]["revision"].get("commit") or "unknown")
        / f"phase{args.through_phase}"
        / "product-acceptance.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "through_phase": args.through_phase,
        "output": str(output.relative_to(REPO_ROOT)),
    }, sort_keys=True))
    if report["status"] == "passed":
        return 0
    if report["status"] == "incomplete":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
