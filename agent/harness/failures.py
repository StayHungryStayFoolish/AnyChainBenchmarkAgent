"""Structured failure facts and policy-constrained recovery contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from agent.utils.redaction import redact


@dataclass(frozen=True)
class RecoveryPolicy:
    code: str
    affected_group: str
    invalidated_groups: tuple[str, ...]
    invalidated_fields: tuple[str, ...] = ()
    allow_correction: bool = True
    allow_retry: bool = False
    llm_analysis_useful: bool = True


RECOVERY_POLICIES: dict[str, RecoveryPolicy] = {
    "DOMAIN_ACTION_BLOCKED": RecoveryPolicy(
        "DOMAIN_ACTION_BLOCKED",
        "opening",
        (),
        allow_retry=True,
    ),
    "PREFLIGHT_CHECK_FAILED": RecoveryPolicy(
        "PREFLIGHT_CHECK_FAILED",
        "preflight_smoke_execution",
        ("preflight_smoke_execution", "job_monitoring"),
    ),
    "FIXTURE_MISSING": RecoveryPolicy(
        "FIXTURE_MISSING",
        "target_samples_fixtures",
        ("target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
    ),
    "ENDPOINT_UNREACHABLE": RecoveryPolicy(
        "ENDPOINT_UNREACHABLE",
        "endpoint_process",
        ("endpoint_process", "preflight_smoke_execution", "job_monitoring"),
    ),
    "RPC_METHOD_OR_SCHEMA_INVALID": RecoveryPolicy(
        "RPC_METHOD_OR_SCHEMA_INVALID",
        "workload_rpc",
        ("workload_rpc", "target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
    ),
    "WORKLOAD_PROCESS_FAILED": RecoveryPolicy(
        "WORKLOAD_PROCESS_FAILED",
        "preflight_smoke_execution",
        ("preflight_smoke_execution", "job_monitoring"),
    ),
    "WORKLOAD_NO_REQUESTS": RecoveryPolicy(
        "WORKLOAD_NO_REQUESTS",
        "workload_rpc",
        ("workload_rpc", "preflight_smoke_execution", "job_monitoring"),
    ),
    "WORKLOAD_ZERO_SUCCESS": RecoveryPolicy(
        "WORKLOAD_ZERO_SUCCESS",
        "workload_rpc",
        ("workload_rpc", "target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
    ),
    "ARTIFACT_INCOMPLETE": RecoveryPolicy(
        "ARTIFACT_INCOMPLETE",
        "report_artifact_analysis",
        ("report_artifact_analysis",),
        allow_correction=False,
    ),
    "HARNESS_INVARIANT_FAILED": RecoveryPolicy(
        "HARNESS_INVARIANT_FAILED",
        "opening",
        (),
        allow_correction=False,
    ),
    "MODEL_PROVIDER_UNAVAILABLE": RecoveryPolicy(
        "MODEL_PROVIDER_UNAVAILABLE",
        "opening",
        (),
        allow_correction=False,
        allow_retry=True,
        llm_analysis_useful=False,
    ),
}


PREFLIGHT_CODE_BY_CHECK = {
    "effective_workload_fixtures_available": "FIXTURE_MISSING",
    "local_rpc_url_valid": "ENDPOINT_UNREACHABLE",
    "rpc_mode_valid": "RPC_METHOD_OR_SCHEMA_INVALID",
    "mixed_weighted_total_valid": "RPC_METHOD_OR_SCHEMA_INVALID",
}

FAILURE_PRIORITY = {
    "FIXTURE_MISSING": 10,
    "ENDPOINT_UNREACHABLE": 20,
    "RPC_METHOD_OR_SCHEMA_INVALID": 30,
    "WORKLOAD_PROCESS_FAILED": 40,
    "WORKLOAD_NO_REQUESTS": 50,
    "WORKLOAD_ZERO_SUCCESS": 60,
    "ARTIFACT_INCOMPLETE": 90,
    "PREFLIGHT_CHECK_FAILED": 100,
}


def failure_record_from_preflight(
    preflight: Mapping[str, Any],
    *,
    confirmed_config: Mapping[str, Any] | None = None,
    execution_id: str = "",
) -> dict[str, Any]:
    failed_checks = [dict(item) for item in preflight.get("checks", []) if not item.get("passed")]
    primary = failed_checks[0] if failed_checks else {"name": "preflight", "detail": "preflight blocked"}
    code = PREFLIGHT_CODE_BY_CHECK.get(str(primary.get("name") or ""), "PREFLIGHT_CHECK_FAILED")
    facts = [
        {
            "code": PREFLIGHT_CODE_BY_CHECK.get(str(item.get("name") or ""), "PREFLIGHT_CHECK_FAILED"),
            "source": "preflight",
            "check": str(item.get("name") or ""),
            "detail": str(item.get("detail") or ""),
        }
        for item in failed_checks
    ] or [{"code": code, "source": "preflight", "detail": "preflight blocked"}]
    return build_failure_record(
        code,
        source="preflight",
        severity="blocking",
        facts=facts,
        evidence_paths=list(preflight.get("evidence_paths") or []),
        confirmed_config=confirmed_config,
        execution_id=execution_id,
    )


def failure_record_from_job(
    job: Mapping[str, Any],
    *,
    confirmed_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = dict(job.get("result_validation") or {})
    facts = [dict(item) for item in validation.get("failure_facts", []) if isinstance(item, dict)]
    if facts:
        facts.sort(key=lambda item: FAILURE_PRIORITY.get(str(item.get("code") or ""), 1000))
        code = str(facts[0].get("code") or "WORKLOAD_PROCESS_FAILED")
    elif str(job.get("status") or "") == "partial":
        code = "ARTIFACT_INCOMPLETE"
        facts = [{"code": code, "source": "artifact", "detail": str(job.get("error") or "required artifacts are incomplete")}]
    else:
        code = "WORKLOAD_PROCESS_FAILED"
        facts = [{"code": code, "source": "workload", "detail": str(job.get("error") or "benchmark workload failed")}]
    artifacts = dict(job.get("artifacts") or {})
    evidence_paths = [
        str(value) for key, value in artifacts.items()
        if key in {"summary_json", "performance_csv", "proxy_method_csv", "vegeta_json", "html_report"} and value
    ]
    run_dir = str(job.get("run_dir") or "")
    if run_dir:
        evidence_paths.append(f"{run_dir}/benchmark.log")
    return build_failure_record(
        code,
        source=str(facts[0].get("source") or "workload"),
        severity="partial" if str(job.get("status") or "") == "partial" else "blocking",
        facts=facts,
        evidence_paths=evidence_paths,
        confirmed_config=confirmed_config,
        job_id=str(job.get("job_id") or ""),
        execution_id=str(job.get("execution_key") or ""),
    )


def model_provider_failure_record(error_type: str) -> dict[str, Any]:
    return build_failure_record(
        "MODEL_PROVIDER_UNAVAILABLE",
        source="model_provider",
        severity="blocking",
        facts=[{"code": "MODEL_PROVIDER_UNAVAILABLE", "source": "model_provider", "error_type": error_type}],
    )


def domain_blocker_failure_record(
    action: Mapping[str, Any],
    *,
    owner: str,
    validation_detail: str,
    retained_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a rejected domain command without losing its recovery context."""

    action_type = str(action.get("type") or "unknown")
    action_id = str(action.get("action_id") or "")
    arguments = {
        str(key): value
        for key, value in action.items()
        if key not in {"action_id", "type", "confidence", "reason"} and not str(key).startswith("_")
    }
    affected_group = str(
        retained_state.get("affected_group")
        or (retained_state.get("pending_question") or {}).get("group")
        or retained_state.get("active_group")
        or "opening"
    )
    record = build_failure_record(
        "DOMAIN_ACTION_BLOCKED",
        source=f"domain:{owner or 'unknown'}",
        severity="blocking",
        facts=[{
            "code": "DOMAIN_ACTION_BLOCKED",
            "source": f"domain:{owner or 'unknown'}",
            "detail": validation_detail,
        }],
        confirmed_config=retained_state.get("confirmed_config") or {},
    )
    safe_action = redact({
        "action_id": action_id,
        "type": action_type,
        "owner": owner,
        "arguments": arguments,
    })
    record.update({
        "action_identity": safe_action,
        "validation": {"detail": str(redact(validation_detail))},
        "retained_state": redact(dict(retained_state)),
        "affected_group": affected_group,
        "recovery_paths": {
            "retry": {
                "action_type": "retry_failure",
                "retained_action": safe_action,
            },
            "correct": {
                "action_type": "correct_failure",
                "target_group": affected_group,
            },
            "cancel": {
                "action_type": "cancel_failure_recovery",
            },
        },
    })
    return record


def build_failure_record(
    code: str,
    *,
    source: str,
    severity: str,
    facts: list[dict[str, Any]],
    evidence_paths: list[str] | None = None,
    confirmed_config: Mapping[str, Any] | None = None,
    job_id: str = "",
    execution_id: str = "",
) -> dict[str, Any]:
    policy = RECOVERY_POLICIES.get(code, RECOVERY_POLICIES["HARNESS_INVARIANT_FAILED"])
    safe_facts = redact(facts)
    safe_paths = redact([str(item) for item in (evidence_paths or []) if str(item).strip()])
    identity = {
        "code": policy.code,
        "source": source,
        "job_id": job_id,
        "execution_id": execution_id,
        "facts": safe_facts,
    }
    failure_id = "failure_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]
    actions = ["inspect_failure", "cancel_failure_recovery"]
    if policy.allow_correction:
        actions.insert(0, "correct_failure")
    if policy.allow_retry:
        actions.insert(0, "retry_failure")
    return {
        "failure_id": failure_id,
        "code": policy.code,
        "source": source,
        "severity": severity,
        "job_id": job_id,
        "execution_id": execution_id,
        "facts": safe_facts,
        "evidence_paths": safe_paths,
        "affected_group": policy.affected_group,
        "invalidated_groups": list(policy.invalidated_groups),
        "invalidated_fields": list(policy.invalidated_fields),
        "allowed_actions": actions,
        "llm_analysis_useful": policy.llm_analysis_useful,
        "preserved_config_keys": sorted(str(key) for key in (confirmed_config or {}).keys()),
    }


def render_failure_summary(record: Mapping[str, Any], language: str) -> str:
    zh = str(language or "").startswith("zh")
    facts = list(record.get("facts") or [])
    detail = "; ".join(_fact_detail(item) for item in facts if isinstance(item, dict)) or "<none>"
    paths = ", ".join(str(item) for item in record.get("evidence_paths") or []) or "<none>"
    preserved = ", ".join(str(item) for item in record.get("preserved_config_keys") or []) or "<none>"
    action = record.get("action_identity") if isinstance(record.get("action_identity"), Mapping) else {}
    action_type = str(action.get("type") or "").strip()
    recovery_paths = record.get("recovery_paths") if isinstance(record.get("recovery_paths"), Mapping) else {}
    choices = ", ".join(str(name) for name in recovery_paths) or "<none>"
    if zh:
        summary = (
            f"执行恢复：{record.get('severity', 'blocking')} / {record.get('code', 'UNKNOWN')} "
            f"（诊断 ID：`{record.get('failure_id', '<unknown>')}`）。\n"
            f"- 已观察到的证据：{detail}\n"
            f"- 保留的已确认配置：{preserved}\n"
            f"- 证据路径：{paths}"
        )
        if action_type:
            summary += f"\n- 失败 action：`{action_type}`（`{action.get('action_id') or '<unknown>'}`）\n- 恢复选择：{choices}"
        return summary
    summary = (
        f"Execution recovery: {record.get('severity', 'blocking')} / {record.get('code', 'UNKNOWN')} "
        f"(diagnostic id: `{record.get('failure_id', '<unknown>')}`).\n"
        f"- Observed evidence: {detail}\n"
        f"- Preserved confirmed config: {preserved}\n"
        f"- Evidence paths: {paths}"
    )
    if action_type:
        summary += f"\n- Failed action: `{action_type}` (`{action.get('action_id') or '<unknown>'}`)\n- Recovery choices: {choices}"
    return summary


def unresolved_recovery(recovery: Mapping[str, Any] | None) -> bool:
    return bool(recovery and recovery.get("record") and recovery.get("status") in {"pending", "ready_to_revalidate"})


def _fact_detail(fact: Mapping[str, Any]) -> str:
    detail = str(fact.get("detail") or "").strip()
    if detail:
        return detail
    if fact.get("status_codes"):
        return f"status_codes={fact.get('status_codes')}"
    if fact.get("artifact"):
        return f"missing artifact: {fact.get('artifact')}"
    if fact.get("returncode") is not None:
        return f"process exit code {fact.get('returncode')}"
    return str(fact.get("code") or "failure")
