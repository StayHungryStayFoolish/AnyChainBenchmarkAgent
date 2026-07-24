"""Single product-readiness authority for AnyChain Agent.

Provider scripts may generate raw evidence, but only this controller can admit
that evidence and advance a phase or G0-G6 gate.
"""

from __future__ import annotations

import argparse
import hashlib
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
from tests.agent_live.execute_harness_contract_ledger import execute_ledger


SCHEMA_VERSION = 1
IMPLEMENTED_THROUGH_PHASE = 7


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


def _checked_phase(phase: int, *test_modules: str) -> dict[str, Any]:
    checks = [
        _run((sys.executable, "-m", "unittest", *test_modules)),
        _run((sys.executable, "tools/check_agent_boundaries.py", "--root", ".")),
    ]
    return {
        "phase": phase,
        "checks": checks,
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
    }


def _phase3_source() -> dict[str, Any]:
    return _checked_phase(
        3,
        "tests.test_agent_turn_lifecycle",
        "tests.test_agent_state_authority",
        "tests.test_agent_execution_application_service",
    )


def _phase4_source() -> dict[str, Any]:
    return _checked_phase(
        4,
        "tests.test_agent_harness_architecture",
        "tests.test_agent_contract_authority",
        "tests.test_agent_group_readiness_authority",
    )


def _phase5_source() -> dict[str, Any]:
    return _checked_phase(
        5,
        "tests.test_agent_state_authority",
        "tests.test_agent_contract_projection",
        "tests.test_agent_legacy_issue_map",
    )


def _retained_regression_inventory() -> dict[str, Any]:
    manifest_path = (
        REPO_ROOT
        / "tests"
        / "agent_live"
        / "fixtures"
        / "real_user_regressions"
        / "manifest.json"
    )
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "case_count": 0,
            "errors": [f"cannot load manifest: {exc}"],
        }
    case_count = 0
    for row in manifest.get("files") or []:
        relative_path = Path(str(row.get("path") or ""))
        path = (REPO_ROOT / relative_path).resolve()
        if REPO_ROOT not in path.parents or not path.is_file():
            errors.append(f"fixture path is missing or outside repository: {relative_path}")
            continue
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_hash != str(row.get("sha256") or ""):
            errors.append(f"fixture hash mismatch: {relative_path}")
        if row.get("sanitized") is not True:
            errors.append(f"fixture is not declared sanitized: {relative_path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load fixture {relative_path}: {exc}")
            continue
        observed_count = len(document.get("cases") or [])
        case_count += observed_count
        if observed_count != int(row.get("case_count") or -1):
            errors.append(f"fixture case count mismatch: {relative_path}")
    if case_count <= 0:
        errors.append("retained regression inventory is empty")
    return {
        "status": "passed" if not errors else "failed",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "case_count": case_count,
        "errors": errors,
    }


def _phase6_complete(
    *,
    ledger: dict[str, Any],
    closure: dict[str, Any],
    observed_domain_owners: set[str],
    expected_domain_owners: set[str],
    regressions: dict[str, Any],
    checks: list[dict[str, Any]],
) -> bool:
    return (
        closure.get("status") == "complete"
        and int(closure.get("required_denominator") or 0) > 0
        and int(closure.get("open_required", -1)) == 0
        and int((ledger.get("summary") or {}).get("groups") or 0) == 20
        and observed_domain_owners == expected_domain_owners
        and not ledger.get("uncataloged_questions")
        and regressions["status"] == "passed"
        and all(check["passed"] for check in checks)
    )


def _phase6_source(inventory: dict[str, Any]) -> dict[str, Any]:
    from agent.workflows.group_registry import GROUPS

    revision = dict(inventory.get("revision") or {})
    revision_id = str(revision.get("worktree_hash") or revision.get("commit") or "unknown")
    phase_root = (
        REPO_ROOT
        / ".agent"
        / "evidence"
        / "control-plane"
        / revision_id
        / "phase6"
    )
    artifact_dir = phase_root / "deterministic"
    ledger = execute_ledger(
        artifact_dir=artifact_dir,
        revision=revision,
    )
    ledger_path = phase_root / "deterministic-ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    closure = dict(
        (ledger.get("summary") or {})
        .get("execution_closure", {})
        .get("deterministic", {})
    )
    expected_domain_owners = {group.owner for group in GROUPS}
    observed_domain_owners = {
        str(group.get("owner") or "")
        for group in ledger.get("groups") or []
        if str(group.get("owner") or "")
    }
    action_owners = {
        str(action.get("owner") or "")
        for action in ledger.get("actions") or []
        if str(action.get("owner") or "")
    }
    regressions = _retained_regression_inventory()
    checks = [
        _run((sys.executable, "-m", "pytest", "-q")),
        _run((sys.executable, "tools/check_agent_boundaries.py", "--root", ".")),
    ]
    complete = _phase6_complete(
        ledger=ledger,
        closure=closure,
        observed_domain_owners=observed_domain_owners,
        expected_domain_owners=expected_domain_owners,
        regressions=regressions,
        checks=checks,
    )
    return {
        "phase": 6,
        "ledger": str(ledger_path.relative_to(REPO_ROOT)),
        "deterministic_closure": closure,
        "groups": int((ledger.get("summary") or {}).get("groups") or 0),
        "domain_owners": sorted(observed_domain_owners),
        "action_owners": sorted(action_owners),
        "retained_regressions": regressions,
        "checks": checks,
        "status": "passed" if complete else "failed",
    }


def _phase7_source() -> dict[str, Any]:
    return _checked_phase(
        7,
        "tests.test_agent_dependency_and_docs_contract",
        "tests.test_agent_legacy_issue_map",
    )


def build_report(through_phase: int) -> dict[str, Any]:
    inventory = build_ledger(None)
    g0 = _g0_source()
    phase_sources = {
        1: lambda: {"phase": 1, "status": "passed", "checks": []},
        2: _phase2_source,
        3: _phase3_source,
        4: _phase4_source,
        5: _phase5_source,
        6: lambda: _phase6_source(inventory),
        7: _phase7_source,
    }
    phase_checks = {
        str(phase): phase_sources[phase]()
        for phase in range(1, min(through_phase, IMPLEMENTED_THROUGH_PHASE) + 1)
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
            "G1": {
                "status": (
                    "passed"
                    if through_phase >= 3
                    and all(
                        phase_checks[str(phase)]["status"] == "passed"
                        for phase in range(3, min(through_phase, 5) + 1)
                    )
                    else "not_run"
                ),
                "owning_phase": 3,
            },
            "G2": {
                "status": (
                    "passed"
                    if through_phase >= 6
                    and phase_checks.get("6", {}).get("status") == "passed"
                    else "not_run"
                ),
                "owning_phase": 6,
            },
            "G3": {"status": "not_run", "owning_phase": 8},
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
