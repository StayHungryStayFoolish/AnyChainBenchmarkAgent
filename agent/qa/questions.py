"""Generate minimal QA questions from plans and discovery results."""

from __future__ import annotations

from typing import Any


def required_questions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    for item in plan.get("required_inputs", []):
        questions.append({
            "id": item,
            "category": "required_input",
            "severity": "blocker",
            "prompt": _required_prompt(item),
        })

    confidence = plan.get("confidence", {})
    confirmed = set(plan.get("confirmed_inputs", []))
    if confidence.get("ledger_device", 1.0) < 0.6 and "ledger_device_confirmation" not in confirmed:
        disks = plan.get("discovery", {}).get("disks", {})
        questions.append({
            "id": "ledger_device_confirmation",
            "category": "environment",
            "severity": "blocker",
            "prompt": "Confirm the ledger/data disk device before running the benchmark.",
            "candidates": disks.get("ambiguous_candidates") or _candidate_disk_names(disks),
        })

    discovery = plan.get("discovery", {})
    missing_required = discovery.get("dependencies", {}).get("missing_required", [])
    if missing_required and "dependency_mode_confirmation" not in confirmed:
        questions.append({
            "id": "dependency_mode_confirmation",
            "category": "dependency",
            "severity": "warning",
            "prompt": "Required dependencies are missing. Choose audit-only, isolated Docker/.venv, or managed install after approval.",
            "missing": missing_required,
        })

    if "mixed_weights" in plan.get("requires_confirmation", []):
        questions.append({
            "id": "mixed_weights_confirmation",
            "category": "workload",
            "severity": "blocker",
            "prompt": "Confirm mixed RPC method weights and parameter samples.",
        })

    if "stress_execution" in plan.get("requires_confirmation", []):
        questions.append({
            "id": "stress_execution_confirmation",
            "category": "safety",
            "severity": "blocker",
            "prompt": "Confirm stress/intensive benchmark execution; it may affect the target node.",
        })

    return _dedupe_questions(questions)


def _required_prompt(item: str) -> str:
    prompts = {
        "chain": "Which blockchain node should be tested?",
        "local_rpc_url": "Provide the local RPC endpoint, or choose fake-node for closed-loop testing.",
        "blockchain_process_names": "Provide blockchain node process names or command keywords for resource attribution.",
        "ledger_device": "Confirm the ledger/data disk device used by the node.",
        "data_vol_max_iops": "Provide the provisioned ledger/data disk IOPS baseline.",
        "data_vol_max_throughput": "Provide the provisioned ledger/data disk throughput baseline in MiB/s.",
        "network_max_bandwidth_gbps": "Provide the instance or pod network bandwidth baseline in Gbps.",
        "rpc_mode": "Choose single or mixed RPC workload mode.",
    }
    return prompts.get(item, f"Provide required value: {item}")


def _candidate_disk_names(disks: dict[str, Any]) -> list[str]:
    names = []
    for item in disks.get("candidates", []):
        name = item.get("name")
        if name and item.get("mountpoint") not in {"", "/", "/boot", "/boot/efi"}:
            names.append(name)
    return names


def _dedupe_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for question in questions:
        qid = question["id"]
        if qid in seen:
            continue
        seen.add(qid)
        result.append(question)
    return result
