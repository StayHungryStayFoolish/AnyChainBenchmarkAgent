"""Risk scoring for generated benchmark plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge.gap_analyzer import analyze_capability_gap


REPO_ROOT = Path(__file__).resolve().parents[2]


def score_plan_risk(plan: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, str | int]] = []
    score = 0

    if plan.get("required_inputs"):
        score += 30
        findings.append({"severity": "high", "points": 30, "message": "Plan has missing required inputs."})

    if "stress_execution" in plan.get("approval_checkpoints", []):
        score += 20
        findings.append({"severity": "medium", "points": 20, "message": "Stress/intensive execution requires explicit approval."})

    if plan.get("use_fake_node"):
        gap = analyze_capability_gap(plan.get("chain", ""))
        if gap.get("fixture_count", 0) == 0:
            score += 25
            findings.append({"severity": "high", "points": 25, "message": "fake-node requested but no fixtures were found for the chain."})

    if plan.get("rpc_mode") == "mixed":
        weighted = _chain_weighted_methods(plan.get("chain", ""))
        total = sum(float(item.get("weight", 0) or 0) for item in weighted)
        if weighted and abs(total - 100.0) > 0.01:
            score += 20
            findings.append({"severity": "medium", "points": 20, "message": f"mixed_weighted total is {total}, expected 100."})
        if not weighted:
            score += 20
            findings.append({"severity": "medium", "points": 20, "message": "mixed mode selected but no mixed_weighted methods found."})

    confidence = plan.get("confidence", {})
    if float(confidence.get("ledger_device", 1.0) or 0.0) < 0.6:
        score += 15
        findings.append({"severity": "medium", "points": 15, "message": "Ledger/data disk confidence is low."})

    env = plan.get("execution", {}).get("environment", {})
    if not plan.get("use_fake_node") and not env.get("LOCAL_RPC_URL"):
        score += 30
        findings.append({"severity": "high", "points": 30, "message": "Real benchmark selected without LOCAL_RPC_URL."})

    if env.get("OBSERVABILITY_STACK_ENABLED") != "true":
        score += 5
        findings.append({"severity": "low", "points": 5, "message": "Prometheus/Grafana observability is disabled."})

    grade = "low"
    if score >= 60:
        grade = "high"
    elif score >= 30:
        grade = "medium"

    return {
        "risk_score": min(score, 100),
        "risk_level": grade,
        "findings": findings,
        "recommendations": _recommendations(findings),
    }


def _chain_weighted_methods(chain: str) -> list[dict[str, Any]]:
    if not chain:
        return []
    path = REPO_ROOT / "config" / "chains" / f"{chain}.json"
    if not path.is_file():
        return []
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("rpc_methods", {}).get("mixed_weighted", [])
    return value if isinstance(value, list) else []


def _recommendations(findings: list[dict[str, str | int]]) -> list[str]:
    if not findings:
        return ["Plan risk is low. Run preflight and a fake-node smoke test before production traffic."]
    recs = []
    messages = " ".join(str(item["message"]) for item in findings)
    if "missing required inputs" in messages:
        recs.append("Answer blocker questions before submitting the job.")
    if "fixtures" in messages:
        recs.append("Record or validate fake-node fixtures before relying on closed-loop results.")
    if "mixed_weighted" in messages:
        recs.append("Normalize mixed workload weights to 100 and re-run plan validation.")
    if "LOCAL_RPC_URL" in messages:
        recs.append("Provide LOCAL_RPC_URL or switch to fake-node for local closed-loop validation.")
    if not recs:
        recs.append("Review findings and confirm the plan before execution.")
    return recs
