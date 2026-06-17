"""Generate human-readable Agent runbooks."""

from __future__ import annotations

from typing import Any

from utils.redaction import redact


def render_runbook(plan: dict[str, Any], preflight: dict[str, Any] | None = None) -> str:
    safe_plan = redact(plan)
    lines = [
        "# AnyChain Benchmark Agent Runbook",
        "",
        f"Plan ID: {safe_plan.get('plan_id', '<unknown>')}",
        f"Chain: {safe_plan.get('chain') or '<missing>'}",
        f"Strategy: {safe_plan.get('strategy')}",
        f"RPC mode: {safe_plan.get('rpc_mode')}",
        f"fake-node: {safe_plan.get('use_fake_node')}",
        f"Dependency mode: {safe_plan.get('dependency_mode', 'audit')}",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(safe_plan.get("execution", {}).get("command", [])),
        "```",
        "",
        "## Environment",
        "",
    ]

    env = safe_plan.get("execution", {}).get("environment", {})
    for key in sorted(env):
        lines.append(f"- {key}={env[key]}")

    lines.extend(["", "## Approval Checkpoints", ""])
    for checkpoint in safe_plan.get("approval_checkpoints", []):
        lines.append(f"- {checkpoint}")

    risk = safe_plan.get("risk", {})
    if risk:
        lines.extend(["", "## Risk", ""])
        lines.append(f"- level: {risk.get('risk_level', '<unknown>')}")
        lines.append(f"- score: {risk.get('risk_score', '<unknown>')}")
        for finding in risk.get("findings", []):
            lines.append(f"- [{finding.get('severity')}] {finding.get('message')}")
        for recommendation in risk.get("recommendations", []):
            lines.append(f"- recommendation: {recommendation}")

    lines.extend(["", "## Required Inputs", ""])
    required = safe_plan.get("required_inputs", [])
    if required:
        for item in required:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    questions = safe_plan.get("required_questions", [])
    if questions:
        lines.extend(["", "## Required Questions", ""])
        for question in questions:
            lines.append(f"- [{question.get('severity')}] {question.get('id')}: {question.get('prompt')}")
            if question.get("candidates"):
                lines.append(f"  candidates: {', '.join(question['candidates'])}")
            if question.get("missing"):
                lines.append(f"  missing: {', '.join(question['missing'])}")

    checklist = safe_plan.get("configuration_checklist", {})
    if checklist:
        lines.extend(["", "## Configuration Checklist", ""])
        lines.append(f"- summary: {checklist.get('summary', '<unknown>')}")
        for section in ("agent", "benchmark", "advanced"):
            items = checklist.get(section, [])
            if not items:
                continue
            lines.append(f"- {section}:")
            for item in items:
                status = "OK" if item.get("present") else "MISSING"
                lines.append(f"  - {status} [{item.get('severity')}] {item.get('id')}: {item.get('description')}")

    materialized = safe_plan.get("materialized_config", {})
    if materialized:
        lines.extend(["", "## Materialized Config", ""])
        lines.append("These values are written to `<job_run_dir>/runtime.env` when the job is submitted.")
        for key in sorted(materialized):
            lines.append(f"- {key}={materialized[key] or '<unset>'}")

    discovery = safe_plan.get("discovery", {})
    if discovery:
        cloud = discovery.get("cloud", {})
        deployment = discovery.get("deployment", {})
        disks = discovery.get("disks", {})
        dependencies = discovery.get("dependencies", {})
        lines.extend(["", "## Discovery", ""])
        lines.append(f"- source: {discovery.get('source', '<unknown>')}")
        lines.append(f"- mode: {discovery.get('mode', '<unknown>')}")
        if cloud:
            lines.append(
                f"- cloud: {cloud.get('platform', '<unknown>')} "
                f"({cloud.get('provider', '<unknown>')}, confidence={cloud.get('confidence', '<unknown>')})"
            )
        if deployment:
            lines.append(f"- deployment: {deployment.get('type', '<unknown>')}")
        if disks:
            lines.append(f"- proposed ledger device: {disks.get('proposed_ledger_device') or '<needs confirmation>'}")
            lines.append(f"- proposed accounts device: {disks.get('proposed_accounts_device') or '<none>'}")
            if disks.get("ambiguous_candidates"):
                lines.append(f"- ambiguous disks: {', '.join(disks['ambiguous_candidates'])}")
        if dependencies:
            missing = dependencies.get("missing_required", [])
            lines.append(f"- dependency mode: {dependencies.get('mode', 'audit')}")
            lines.append(f"- missing required dependencies: {', '.join(missing) if missing else 'none'}")
        for warning in discovery.get("warnings", []):
            lines.append(f"- warning: {warning}")

    if preflight is not None:
        lines.extend(["", "## Preflight", ""])
        for check in preflight.get("checks", []):
            status = "PASS" if check.get("passed") else "FAIL"
            detail = f" - {check.get('detail')}" if check.get("detail") else ""
            lines.append(f"- {status}: {check.get('name')}{detail}")

    lines.extend([
        "",
        "## Expected Artifacts",
        "",
    ])
    for key, value in safe_plan.get("artifacts", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Stop / Rollback",
        "",
        "- Use `python3 agent/cli.py status --job-id <job_id>` to inspect a submitted job.",
        "- Use the benchmark framework cleanup path for any active benchmark process.",
    ])

    return "\n".join(lines) + "\n"
