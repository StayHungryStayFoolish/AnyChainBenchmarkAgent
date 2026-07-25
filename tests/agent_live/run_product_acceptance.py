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
import uuid
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_live.generate_harness_coverage_ledger import build_ledger
from tests.agent_live.generate_harness_coverage_ledger import ingest_evidence_artifacts
from tests.agent_live.execute_harness_contract_ledger import execute_ledger
from tests.agent_live.coverage_evidence import (
    validate_real_execution_ledger_artifacts,
)
from tests.agent_live.execute_real_execution_ledger import (
    validate_g5_failure_artifact,
)
from tests.agent_live.g5_collection_manifest import (
    load_active_g5_collection,
)
from tests.agent_live.export_approved_plan import (
    cleanup_approved_plan_evidence,
    validate_completed_cleanup_receipt,
)
from agent.runners.execution_scenarios import scenario_by_id
from tests.agent_live.product_chaos_obligations import (
    build_product_chaos_obligations,
)
from tests.agent_live.product_obligation_evidence import (
    admit_product_chaos_rounds,
)
from tests.agent_live.completed_journey_batch import (
    G4_ARTIFACT_TYPE,
    load_completed_journey_batch_evidence,
)
from tests.agent_live.generate_product_review_evidence import (
    generate_product_review_evidence,
)
from tests.agent_live.product_review_contract import (
    PRODUCT_REVIEW_SCHEMA_VERSION,
    content_hash as product_review_content_hash,
)
from tests.agent_live.product_review_scope import DEFAULT_SCOPE_MANIFEST
from tests.agent_live.linux_shell_gates import (
    execute_required_linux_shell_gates,
    load_linux_shell_gate_manifest,
)
from tests.agent_live.retained_regression_evidence_set import (
    load_retained_regression_evidence_set,
)
from tests.agent_live.retained_regression_obligations import (
    build_retained_regression_obligations,
)
from tests.agent_live.validate_product_review_evidence import (
    validate_product_review_evidence,
)


SCHEMA_VERSION = 1
IMPLEMENTED_THROUGH_PHASE = 8
EMPTY_WORKTREE_HASH = hashlib.sha256(b"").hexdigest()
FULL_PYTHON_SUITE_COMMAND = (
    sys.executable,
    "tests/run_offline_python_suite.py",
)


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


def _revision_id(revision: dict[str, Any]) -> str:
    commit = str(revision.get("commit") or "unknown")
    worktree_hash = str(revision.get("worktree_hash") or "")
    if not worktree_hash or worktree_hash == EMPTY_WORKTREE_HASH:
        return commit
    return f"{commit}-{worktree_hash[:16]}"


def _json_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


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
    from agent.harness import coordinator, hierarchical_planner

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
        not hasattr(coordinator, "resolve_action_queue")
        and not hasattr(coordinator, "plan_turn_step")
        and not hasattr(hierarchical_planner, "resolve_product_action_queue")
        and all(
            callable(getattr(hierarchical_planner, name, None))
            for name in (
                "begin_semantic_partition",
                "compile_next_owner",
                "review_semantic_plan",
            )
        )
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


def _phase1_source() -> dict[str, Any]:
    return _checked_phase(
        1,
        "tests.test_agent_contract_projection",
        "tests.test_agent_question_prompts",
    )


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
    shell_gates: dict[str, Any],
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
        and shell_gates.get("status") == "passed"
        and all(check["passed"] for check in checks)
    )


def _phase6_source(inventory: dict[str, Any]) -> dict[str, Any]:
    from agent.workflows.group_registry import GROUPS

    revision = dict(inventory.get("revision") or {})
    revision_id = _revision_id(revision)
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
    try:
        shell_manifest = load_linux_shell_gate_manifest(REPO_ROOT)
        shell_gates = execute_required_linux_shell_gates(
            REPO_ROOT,
            shell_manifest,
        )
    except (OSError, TypeError, ValueError) as exc:
        shell_gates = {
            "status": "failed",
            "reason": str(exc),
            "classified_denominator": 0,
            "required_denominator": 0,
            "passed": 0,
            "failed": 0,
            "results": [],
        }
    shell_receipt_unsigned = {
        "schema_version": PRODUCT_REVIEW_SCHEMA_VERSION,
        "source_type": "linux_shell_gate_receipt",
        "revision": revision,
        "execution": shell_gates,
    }
    shell_receipt_path = _write_json(
        phase_root / "linux-shell-gates.json",
        {
            **shell_receipt_unsigned,
            "source_hash": product_review_content_hash(
                shell_receipt_unsigned
            ),
        },
    )
    checks = [
        _run(FULL_PYTHON_SUITE_COMMAND),
        _run((sys.executable, "tools/check_agent_boundaries.py", "--root", ".")),
    ]
    complete = _phase6_complete(
        ledger=ledger,
        closure=closure,
        observed_domain_owners=observed_domain_owners,
        expected_domain_owners=expected_domain_owners,
        regressions=regressions,
        shell_gates=shell_gates,
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
        "linux_shell_gates": shell_gates,
        "linux_shell_gate_receipt": str(
            shell_receipt_path.relative_to(REPO_ROOT)
        ),
        "checks": checks,
        "status": "passed" if complete else "failed",
    }


def _phase7_source() -> dict[str, Any]:
    return _checked_phase(
        7,
        "tests.test_agent_dependency_and_docs_contract",
        "tests.test_agent_legacy_issue_map",
    )


def _phase8_g3_gate(
    *,
    phase_root: Path,
    obligations: Sequence[dict[str, Any]],
    revision: dict[str, str],
) -> dict[str, Any]:
    provider_path = phase_root / "g3" / "provider.json"
    manifest_path = phase_root / "g3" / "evidence-set" / "manifest.json"
    missing = [
        _display_path(path)
        for path in (provider_path, manifest_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "status": "incomplete",
            "complete": False,
            "reason": "immutable G3 evidence set is not available",
            "missing": missing,
        }
    try:
        provider = _load_json_object(
            provider_path,
            "G3 retained-regression provider",
        )
        manifest = load_retained_regression_evidence_set(
            manifest_path,
            provider=provider,
            obligations=obligations,
            revision=revision,
        )
        passed = int(manifest["obligation_count"])
        exact_count = int(manifest["exact_count"])
        open_count = int(manifest["open_count"])
        manifest_hash = str(manifest["manifest_hash"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "complete": False,
            "reason": f"immutable G3 evidence set was rejected: {exc}",
            "manifest": _display_path(manifest_path),
        }
    return {
        "status": "passed",
        "complete": True,
        "passed": passed,
        "failed": 0,
        "external": 0,
        "not_run": 0,
        "exact_count": exact_count,
        "open_count": open_count,
        "manifest": _display_path(manifest_path),
        "manifest_hash": manifest_hash,
    }


def _phase8_g4_gate(
    *,
    phase_root: Path,
    obligations: Sequence[dict[str, Any]],
    revision: dict[str, str],
) -> dict[str, Any]:
    evidence_by_round: dict[str, tuple[Path, ...]] = {}
    missing: list[str] = []
    try:
        for round_id in ("round-1", "round-2"):
            index_path = (
                phase_root
                / "g4"
                / round_id
                / "evidence-set"
                / "manifest.json"
            )
            if not index_path.is_file():
                missing.append(_display_path(index_path))
                evidence_by_round[round_id] = ()
                continue
            evidence_by_round[round_id] = load_completed_journey_batch_evidence(
                index_path,
                obligations=obligations,
                revision=revision,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id=round_id,
            )
        if missing:
            return {
                "status": "incomplete",
                "complete": False,
                "reason": "immutable G4 round evidence is not available",
                "missing": missing,
            }
        return admit_product_chaos_rounds(
            obligations=obligations,
            evidence_by_round=evidence_by_round,
            revision=revision,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "complete": False,
            "reason": f"immutable G4 evidence set was rejected: {exc}",
        }


def _phase8_g5_gate(
    *,
    phase_root: Path,
    revision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_ledger = build_ledger(revision=revision)
    g5_root = phase_root / "g5"
    pointer_exists = (g5_root / "active-collection.json").exists()
    manifest, manifest_reason = load_active_g5_collection(
        g5_root,
        revision=revision,
    )
    if manifest is None:
        return execution_ledger, {
            "status": "failed" if pointer_exists else "incomplete",
            "global_ledger_valid": False,
            "global_ledger_reason": (
                manifest_reason or "G5 collection has not been published"
            ),
        }

    summary: dict[str, Any] = {
        "attempt_id": str(manifest.get("attempt_id") or ""),
        "collection_status": str(manifest.get("status") or ""),
    }
    try:
        evidence_paths = [
            Path(str(item.get("path") or ""))
            for item in manifest.get("evidence") or ()
        ]
        if evidence_paths:
            execution_ledger = ingest_evidence_artifacts(
                execution_ledger,
                evidence_paths,
            )
        summary.update(
            dict(
                (execution_ledger.get("summary") or {})
                .get("execution_closure", {})
                .get("real_execution", {})
            )
        )
        if manifest.get("status") == "failed":
            failure = manifest.get("failure_evidence")
            reason = "G5 attempt failed before completing all required lanes"
            if isinstance(failure, dict):
                failure_path = Path(str(failure.get("path") or ""))
                failure_payload = _load_json_object(
                    failure_path,
                    "G5 failure evidence",
                )
                if failure_payload.get("artifact_type") == "g5_real_execution_failure":
                    valid, failure_reason = validate_g5_failure_artifact(
                        failure_path,
                        revision=revision,
                    )
                    if not valid:
                        raise ValueError(failure_reason)
                    reason = "G5 attempt failed with validated stage evidence"
                elif failure_payload.get("evidence_class") == "real_execution":
                    reason = "G5 attempt retained an observed execution failure"
                else:
                    raise ValueError("G5 failure evidence type is invalid")
            return execution_ledger, {
                **summary,
                "status": "failed",
                "global_ledger_valid": False,
                "global_ledger_reason": reason,
            }

        artifacts = [
            _load_json_object(path, "G5 real-execution evidence")
            for path in evidence_paths
        ]
        artifacts.sort(
            key=lambda artifact: int(
                ((artifact.get("request") or {}).get("ledger_sequence") or 0)
            )
        )
        globally_valid, global_reason = (
            validate_real_execution_ledger_artifacts(
                artifacts,
                revision=revision,
            )
        )
        complete = (
            summary.get("status") == "complete"
            and int(summary.get("required_denominator") or 0) == 4
            and int(summary.get("open_required", -1)) == 0
        )
        return execution_ledger, {
            **summary,
            "status": "passed" if complete and globally_valid else "failed",
            "global_ledger_valid": globally_valid,
            "global_ledger_reason": global_reason,
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return execution_ledger, {
            **summary,
            "status": "failed",
            "global_ledger_valid": False,
            "global_ledger_reason": f"G5 collection was rejected: {exc}",
        }


def _phase8_source(inventory: dict[str, Any]) -> dict[str, Any]:
    revision = dict(inventory.get("revision") or {})
    phase_root = (
        REPO_ROOT
        / ".agent"
        / "evidence"
        / "control-plane"
        / _revision_id(revision)
        / "phase8"
    )

    g3_obligations = build_retained_regression_obligations(
        repo_root=REPO_ROOT,
        revision=revision,
    )
    g4_obligations = build_product_chaos_obligations(revision=revision)
    g3_catalog = _write_json(
        phase_root / "g3" / "obligations.json",
        {"revision": revision, "obligations": g3_obligations},
    )
    g4_catalog = _write_json(
        phase_root / "g4" / "obligations.json",
        {"revision": revision, "obligations": g4_obligations},
    )
    g3_summary = _phase8_g3_gate(
        phase_root=phase_root,
        obligations=g3_obligations,
        revision=revision,
    )
    g4_summary = _phase8_g4_gate(
        phase_root=phase_root,
        obligations=g4_obligations,
        revision=revision,
    )

    execution_ledger, g5_summary = _phase8_g5_gate(
        phase_root=phase_root,
        revision=revision,
    )
    _write_json(phase_root / "g5" / "execution-ledger.json", execution_ledger)

    g3_status = str(g3_summary["status"])
    g4_status = "passed" if g4_summary["complete"] else g4_summary["status"]
    g5_status = str(g5_summary["status"])
    prerequisites_passed = all(
        status == "passed" for status in (g3_status, g4_status, g5_status)
    )
    g6 = _phase8_product_review(
        revision=revision,
        phase_root=phase_root,
        prerequisites_passed=prerequisites_passed,
    )
    statuses = (g3_status, g4_status, g5_status, g6["status"])
    phase_status = (
        "passed"
        if all(status == "passed" for status in statuses)
        else "failed"
        if "failed" in statuses
        else "incomplete"
    )
    return {
        "phase": 8,
        "obligation_catalogs": {
            "G3": str(g3_catalog.relative_to(REPO_ROOT)),
            "G4": str(g4_catalog.relative_to(REPO_ROOT)),
        },
        "gates": {
            "G3": {**g3_summary, "status": g3_status},
            "G4": {**g4_summary, "status": g4_status},
            "G5": {
                **g5_summary,
                "status": g5_status,
            },
            "G6": g6,
        },
        "status": phase_status,
    }


def _phase8_product_review(
    *,
    revision: dict[str, str],
    phase_root: Path,
    prerequisites_passed: bool,
) -> dict[str, Any]:
    if not prerequisites_passed:
        return {
            "status": "not_run",
            "reason": "G3-G5 must pass before final product review",
        }
    config_path = phase_root / "g6" / "generator-config.json"
    if not config_path.is_file():
        return {
            "status": "incomplete",
            "reason": "typed G6 generator config is missing",
            "config": _display_path(config_path),
        }
    try:
        config = _load_g6_generator_config(config_path, revision=revision)
        output_dir = (
            phase_root
            / "g6"
            / "generated"
            / f"review-{uuid.uuid4().hex}"
        )
        manifest_path = generate_product_review_evidence(
            output_dir=output_dir,
            revision=revision,
            planner_receipt_paths=config["planner_receipt_paths"],
            migration_cutoff_path=config["migration_cutoff_path"],
            bilingual_docs_path=config["bilingual_docs_path"],
            runtime_hygiene_path=config["runtime_hygiene_path"],
            external_capability_paths=config["external_capability_paths"],
            severity_ledger_path=config["severity_ledger_path"],
            shell_gate_receipt_path=(
                phase_root.parent / "phase6" / "linux-shell-gates.json"
            ),
            scope_repo_root=REPO_ROOT,
            scope_manifest_path=DEFAULT_SCOPE_MANIFEST,
        )
        decision = validate_product_review_evidence(
            manifest_path,
            revision=revision,
        )
        if (
            not isinstance(decision, dict)
            or decision.get("gate") != "G6"
            or decision.get("status") not in {"passed", "failed"}
        ):
            raise ValueError("independent G6 validator returned an invalid decision")
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "status": "failed",
            "reason": f"typed G6 evidence was rejected: {exc}",
            "config": _display_path(config_path),
        }
    return {
        **decision,
        "config": _display_path(config_path),
        "manifest": _display_path(manifest_path),
        "generation": "generated",
    }


def _cleanup_phase8_confidential_evidence(
    report: dict[str, Any],
) -> dict[str, Any]:
    revision = dict((report.get("inventory") or {}).get("revision") or {})
    phase_root = (
        REPO_ROOT
        / ".agent"
        / "evidence"
        / "control-plane"
        / _revision_id(revision)
        / "phase8"
    )
    contracts: list[dict[str, Any]] = []
    collection, reason = load_active_g5_collection(
        phase_root / "g5",
        revision=revision,
    )
    if collection is None or collection.get("status") != "complete":
        raise ValueError(
            "terminal confidential cleanup requires a complete G5 collection: "
            + reason
        )
    evidence_paths = [
        Path(str(item.get("path") or ""))
        for item in collection.get("evidence") or ()
    ]
    for path in evidence_paths:
        artifact = _load_json_object(path, "G5 real-execution evidence")
        if artifact.get("evidence_class") != "real_execution":
            continue
        request = dict(artifact.get("request") or {})
        scenario = scenario_by_id(str(artifact.get("scenario_id") or ""))
        target_mode = (
            "sync-observe"
            if scenario.workflow_type == "sync_observe"
            else "fake-node"
            if scenario.operation == "fake_node_smoke"
            else "real-node"
        )
        contracts.append({
            "artifact_file": str(
                request.get("approved_plan_provenance_file") or ""
            ),
            "plan_file": str(request.get("approved_plan_file") or ""),
            "approval_artifact_sha256": str(
                request.get("approved_plan_provenance_sha256") or ""
            ),
            "plan_sha256": str(
                request.get("approved_plan_sha256") or ""
            ),
            "workflow": scenario.workflow_type,
            "target_mode": target_mode,
            "approval_action": scenario.action_type,
        })
    if len(contracts) != 4:
        raise ValueError(
            "terminal confidential cleanup requires four validated G5 lanes"
        )
    receipt = cleanup_approved_plan_evidence(
        contracts,
        expected_revision=revision,
        output_dir=phase_root / "g5" / "confidential-cleanup",
    )
    payload = validate_completed_cleanup_receipt(
        receipt,
        expected_revision=revision,
        expected_evidence_root=Path(
            str(contracts[0].get("artifact_file") or "")
        ).resolve().parent,
        expected_contracts=contracts,
        output_dir=(phase_root / "g5" / "confidential-cleanup").resolve(),
    )
    return {
        "status": "passed",
        "receipt": _display_path(receipt),
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "cleanup_outcome": str(payload.get("status") or ""),
        "deleted_file_count": int(payload.get("deleted_file_count") or 0),
    }


def _load_g6_generator_config(
    path: Path,
    *,
    revision: dict[str, str],
) -> dict[str, Any]:
    config = _load_json_object(path, "G6 generator config")
    expected_fields = {
        "revision",
        "planner_receipt_paths",
        "migration_cutoff_path",
        "bilingual_docs_path",
        "runtime_hygiene_path",
        "external_capability_paths",
        "severity_ledger_path",
    }
    if set(config) != expected_fields:
        raise ValueError("G6 generator config schema is invalid")
    if config.get("revision") != revision:
        raise ValueError("G6 generator config is bound to a stale revision")
    for field in ("planner_receipt_paths", "external_capability_paths"):
        values = config.get(field)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"G6 generator config {field} is invalid")
    for field in (
        "migration_cutoff_path",
        "bilingual_docs_path",
        "runtime_hygiene_path",
        "severity_ledger_path",
    ):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"G6 generator config {field} is invalid")
    return config


def _aggregate_status(
    *,
    requested_supported: bool,
    g0_status: str,
    phase_statuses: Sequence[str],
) -> str:
    """Preserve hard failures before representing unfinished later work."""

    statuses = tuple(str(status) for status in phase_statuses)
    if g0_status == "failed" or "failed" in statuses:
        return "failed"
    if (
        not requested_supported
        or any(status in {"incomplete", "not_run"} for status in statuses)
    ):
        return "incomplete"
    return "passed"


def build_report(through_phase: int) -> dict[str, Any]:
    inventory = build_ledger(None)
    g0 = _g0_source()
    phase_sources = {
        1: _phase1_source,
        2: _phase2_source,
        3: _phase3_source,
        4: _phase4_source,
        5: _phase5_source,
        6: lambda: _phase6_source(inventory),
        7: _phase7_source,
        8: lambda: _phase8_source(inventory),
    }
    phase_checks = {
        str(phase): phase_sources[phase]()
        for phase in range(1, min(through_phase, IMPLEMENTED_THROUGH_PHASE) + 1)
    }
    requested_supported = through_phase <= IMPLEMENTED_THROUGH_PHASE
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
            "G3": {
                **phase_checks.get("8", {}).get("gates", {}).get(
                    "G3", {"status": "not_run"}
                ),
                "owning_phase": 8,
            },
            "G4": {
                **phase_checks.get("8", {}).get("gates", {}).get(
                    "G4", {"status": "not_run"}
                ),
                "owning_phase": 8,
            },
            "G5": {
                **phase_checks.get("8", {}).get("gates", {}).get(
                    "G5", {"status": "not_run"}
                ),
                "owning_phase": 8,
            },
            "G6": {
                **phase_checks.get("8", {}).get("gates", {}).get(
                    "G6", {"status": "not_run"}
                ),
                "owning_phase": 8,
            },
        },
        "status": _aggregate_status(
            requested_supported=requested_supported,
            g0_status=str(g0.get("status") or "failed"),
            phase_statuses=tuple(
                str(phase_checks[str(phase)].get("status") or "failed")
                for phase in range(
                    1,
                    min(through_phase, IMPLEMENTED_THROUGH_PHASE) + 1,
                )
            ),
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
        / _revision_id(dict(report["inventory"]["revision"] or {}))
        / f"phase{args.through_phase}"
        / "product-acceptance.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if report["status"] == "passed" and args.through_phase == 8:
        provisional = {
            **report,
            "status": "finalizing",
            "confidential_evidence_cleanup": {"status": "pending"},
        }
        output.write_text(
            json.dumps(provisional, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            report["confidential_evidence_cleanup"] = (
                _cleanup_phase8_confidential_evidence(report)
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            report["status"] = "failed"
            report["confidential_evidence_cleanup"] = {
                "status": "failed",
                "reason": str(exc),
            }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "through_phase": args.through_phase,
        "output": _display_path(output),
    }, sort_keys=True))
    if report["status"] == "passed":
        return 0
    if report["status"] == "incomplete":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
