"""Deterministic benchmark runtime invoked by the execution domain."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ...runners.application_service import (
    ExecutionOperation,
    ExecutionRequest,
    execution_service,
)
from ...planners.strategy_planner import (
    materialize_custom_rpc_template,
)

from ..state import AgentGraphState
from ..sync_observe_contract import SyncObserveRequest
from ..contracts import HandlerResult, RecoveryCommand, ResponseFragment, StateDelta
from ..failures import failure_record_from_job, failure_record_from_preflight
from .rpc_catalog import validated_contracts_view
from .rpc_receipts import emit_materialization_receipt


def execute_approved_preflight_and_smoke(state: AgentGraphState) -> HandlerResult:
    """Run deterministic preflight/smoke after explicit graph approval."""

    output: AgentGraphState = deepcopy(state)
    if output.get("preflight", {}).get("status") == "blocked" or output.get("smoke"):
        return HandlerResult()

    if output.get("workflow_mode") == "sync_observe":
        prepared = _prepare_benchmark_with_runtime_contract(output)
        data = prepared.get("data", {})
        preflight = data.get("preflight", {})
        output["plan"] = data.get("plan", {})
        output["plan_file"] = data.get("plan_file", "")
        output["preflight"] = {
            **(output.get("preflight") or {}),
            **preflight,
            "status": "passed" if preflight.get("passed") else "blocked",
            "evidence_paths": prepared.get("evidence_paths", []),
        }
        if not preflight.get("passed"):
            return _execution_result(
                state,
                output,
                recovery_command=_preflight_recovery_command(output),
                completion="blocked",
            )
        recovery_command = _resolved_recovery_command(output, validation_receipt="preflight_passed")
        job_result = execution_service.execute(
            ExecutionRequest(
                operation=ExecutionOperation.SYNC_OBSERVE,
                plan_file=str(data.get("plan_file", "")),
                plan=output.get("plan") or None,
                approved=True,
                idempotency_key=_execution_idempotency_key(output),
            )
        ).to_dict()
        output["job"] = _job_from_execution_result(job_result)
        if _execution_result_failed(job_result, output["job"]):
            return _execution_result(
                state,
                output,
                recovery_command=_job_recovery_command(output),
                completion="blocked",
            )
        return _execution_result(
            state,
            output,
            response_fragment=_job_fragment(
                job_result,
                message_id="execution.response.sync_observe_submitted",
            ),
            recovery_command=recovery_command,
        )

    prepared = _prepare_benchmark_with_runtime_contract(output)
    data = prepared.get("data", {})
    preflight = data.get("preflight", {})
    output["plan"] = data.get("plan", {})
    output["plan_file"] = data.get("plan_file", "")
    output["preflight"] = {
        **(output.get("preflight") or {}),
        **preflight,
        "status": "passed" if preflight.get("passed") else "blocked",
        "evidence_paths": prepared.get("evidence_paths", []),
    }
    if not preflight.get("passed"):
        return _execution_result(
            state,
            output,
            recovery_command=_preflight_recovery_command(output),
            completion="blocked",
        )
    recovery_command = _resolved_recovery_command(output, validation_receipt="preflight_passed")

    if output.get("target_mode") == "fake-node":
        smoke = execution_service.execute(
            ExecutionRequest(
                operation=ExecutionOperation.FAKE_NODE_SMOKE,
                plan_file=str(data.get("plan_file", "")),
                plan=output.get("plan") or None,
                approved=True,
                idempotency_key=_execution_idempotency_key(output),
            )
        ).to_dict()
        output["smoke"] = smoke
        output["job"] = _job_from_execution_result(smoke)
        if _execution_result_failed(smoke, output["job"]):
            return _execution_result(
                state,
                output,
                recovery_command=_job_recovery_command(output),
                completion="blocked",
            )
        if recovery_command is not None:
            recovery_command = RecoveryCommand(operation="resolve", validation_receipt="fake_node_smoke_passed")
        return _execution_result(
            state,
            output,
            response_fragment=_job_fragment(
                smoke,
                message_id="execution.response.fake_node_smoke_submitted",
            ),
            recovery_command=recovery_command,
        )

    smoke = execution_service.execute(
        ExecutionRequest(
            operation=ExecutionOperation.REAL_NODE_SMOKE,
            plan_file=str(data.get("plan_file", "")),
            plan=output.get("plan") or None,
            approved=True,
            idempotency_key=_execution_idempotency_key(output),
        )
    ).to_dict()
    smoke_job = _job_from_execution_result(smoke)
    output["smoke"] = {
        "purpose": "real_node_isolated_smoke",
        "status": str(smoke_job.get("status") or smoke.get("status") or "unknown"),
        "job_id": str(smoke_job.get("job_id") or ""),
        "result": smoke,
    }
    output["job"] = smoke_job
    if _execution_result_failed(smoke, smoke_job):
        return _execution_result(
            state,
            output,
            recovery_command=_job_recovery_command(output),
            completion="blocked",
        )
    return _execution_result(
        state,
        output,
        response_fragment=_job_fragment(
            smoke,
            message_id="execution.response.real_node_smoke_submitted",
        ),
        recovery_command=recovery_command,
    )


def execute_approved_final_benchmark(state: AgentGraphState) -> HandlerResult:
    """Submit the immutable prepared plan after a successful real-node smoke."""

    output: AgentGraphState = deepcopy(state)
    final = output.setdefault("final_benchmark", {})
    if final.get("job_id"):
        return HandlerResult()
    result = execution_service.execute(
        ExecutionRequest(
            operation=ExecutionOperation.FINAL_BENCHMARK,
            plan_file=str(output.get("plan_file") or ""),
            plan=output.get("plan") or None,
            approved=True,
            idempotency_key=_execution_idempotency_key(output),
        )
    ).to_dict()
    job = (result.get("data") or {}).get("job", {})
    if not job:
        warnings = "; ".join(str(item) for item in result.get("warnings") or [] if str(item).strip())
        final.update({"approved": False, "status": "blocked", "error": warnings or "final plan is unavailable"})
        return _execution_result(
            state,
            output,
            response_fragment=ResponseFragment(
                kind="warning",
                message_id="execution.response.final_not_submitted",
                arguments={"reason": str(final["error"])},
                source=__name__,
            ),
            completion="blocked",
        )
    output["job"] = job
    final.update({
        "approved": True,
        "status": str(job.get("status") or result.get("status") or "unknown"),
        "job_id": str(job.get("job_id") or ""),
        "result": result,
    })
    if str(job.get("status") or "") in {"failed", "partial"}:
        return _execution_result(
            state,
            output,
            recovery_command=_job_recovery_command(output),
            completion="blocked",
        )
    return _execution_result(
        state,
        output,
        response_fragment=_job_fragment(
            result,
            message_id="execution.response.final_benchmark_submitted",
        ),
    )


def _execution_result(
    original: AgentGraphState,
    output: AgentGraphState,
    *,
    recovery_command: RecoveryCommand | None = None,
    response_fragment: ResponseFragment | None = None,
    completion: str = "completed",
) -> HandlerResult:
    previous_receipt_ids = {
        str(item.get("receipt_id") or "")
        for item in (original.get("turn_context") or {}).get("control_receipts") or ()
        if isinstance(item, Mapping)
    }
    return HandlerResult(
        delta=StateDelta.between(original, output),
        control_receipts=tuple(
            deepcopy(dict(item))
            for item in (output.get("turn_context") or {}).get("control_receipts") or ()
            if isinstance(item, Mapping)
            and str(item.get("receipt_id") or "") not in previous_receipt_ids
        ),
        recovery_command=recovery_command,
        clear_pending=recovery_command is None,
        response_fragments=(response_fragment,) if response_fragment else (),
        completion=completion,  # type: ignore[arg-type]
        stop_after_response=True,
    )


def _prepare_kwargs(state: AgentGraphState) -> dict[str, Any]:
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    qps_overrides = qps.get("overrides") or {}
    obs = state.get("observability") or {}
    workload = state.get("workload") or {}
    target_mode = state.get("target_mode")
    sync_request = SyncObserveRequest.from_state(state)
    chain = (state.get("chain_identity") or {}).get("canonical") or confirmed.get("BLOCKCHAIN_NODE", "")
    kwargs = {
        "source_prompt": "LangGraph Harness approved preflight/smoke",
        "chain": chain,
        "goal": _goal_from_mode(str(qps.get("mode") or "quick")),
        "rpc_mode": state.get("rpc_mode") or "single",
        "use_fake_node": target_mode == "fake-node",
        "target_rpc_url": str(confirmed.get("LOCAL_RPC_URL") or ""),
        "blockchain_process_names": ["fake-node"] if target_mode == "fake-node" else [],
        "deployment_type": str(((state.get("discovery") or {}).get("deployment") or {}).get("type") or ""),
        "cloud_provider": str(((state.get("discovery") or {}).get("cloud") or {}).get("provider") or ""),
        "ledger_device": str(confirmed.get("LEDGER_DEVICE") or ""),
        "accounts_device": str(confirmed.get("ACCOUNTS_DEVICE") or ""),
        "cloud_region": str(confirmed.get("CLOUD_REGION") or ""),
        "cloud_zone": str(confirmed.get("CLOUD_ZONE") or ""),
        "machine_type": str(confirmed.get("MACHINE_TYPE") or ""),
        "data_vol_type": str(confirmed.get("DATA_VOL_TYPE") or ""),
        "data_vol_size": str(confirmed.get("DATA_VOL_SIZE") or ""),
        "data_vol_max_iops": str(confirmed.get("DATA_VOL_MAX_IOPS") or ""),
        "data_vol_max_throughput": str(confirmed.get("DATA_VOL_MAX_THROUGHPUT") or ""),
        "accounts_vol_type": str(confirmed.get("ACCOUNTS_VOL_TYPE") or ""),
        "accounts_vol_size": str(confirmed.get("ACCOUNTS_VOL_SIZE") or ""),
        "accounts_vol_max_iops": str(confirmed.get("ACCOUNTS_VOL_MAX_IOPS") or ""),
        "accounts_vol_max_throughput": str(confirmed.get("ACCOUNTS_VOL_MAX_THROUGHPUT") or ""),
        "network_interface": str(confirmed.get("NETWORK_INTERFACE") or ""),
        "network_max_bandwidth_gbps": str(confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS") or ""),
        "qps_initial": _int_or_none(qps_overrides.get("INITIAL_QPS")),
        "qps_max": _int_or_none(qps_overrides.get("MAX_QPS")),
        "qps_step": _int_or_none(qps_overrides.get("QPS_STEP")),
        "duration_seconds": _int_or_none(qps_overrides.get("DURATION")),
        "observability_enabled": obs.get("mode") not in {"", "disabled", None},
        "observability_mode": str(obs.get("mode") or "disabled"),
        "workflow_type": "sync_observe" if state.get("workflow_mode") == "sync_observe" else "rpc_benchmark",
        "confirmations": [
            "benchmark_mode_confirmed",
            "qps_profile_confirmed",
            "observability_choice_confirmed",
            "chain_template_reviewed",
            "rpc_workload_confirmed",
            "rpc_workload_confirmation",
            "rpc_param_samples_confirmed",
            "rpc_param_samples_confirmation",
            "advanced_config_review",
            "disk_inventory_confirmation",
            "ledger_device_confirmation",
            "has_accounts_device",
            "sync_observe_stop_condition",
        ],
    }
    kwargs.update(sync_request.execution_values())
    # Mixed RPC mode requires weights to be confirmed. Both the default-workload
    # path (template weights sum to 100) and the custom/adjusted path (validated
    # to sum to 100) set `workload.confirmed`; without this the checklist blocks
    # every mixed run on `mixed_weights_confirmed`.
    if state.get("rpc_mode") == "mixed" and workload.get("confirmed"):
        kwargs["confirmations"].append("mixed_weights_confirmed")
    process_names = str(confirmed.get("BLOCKCHAIN_PROCESS_NAMES") or "").strip()
    if process_names:
        kwargs["blockchain_process_names"] = [process_names]
    if confirmed.get("MAINNET_RPC_URL"):
        kwargs["mainnet_rpc_url"] = str(confirmed.get("MAINNET_RPC_URL") or "")
    methods = workload.get("methods")
    weights = workload.get("mixed_weights")
    if isinstance(methods, list) and methods:
        kwargs["rpc_methods"] = [str(method) for method in methods if str(method).strip()]
    if isinstance(weights, dict) and weights:
        kwargs["mixed_weights"] = {str(method): int(weight) for method, weight in weights.items()}
    return kwargs


def _prepare_benchmark_with_runtime_contract(state: AgentGraphState) -> dict[str, Any]:
    """Prepare one immutable plan containing the validated custom-RPC closure."""

    prepare_kwargs = _prepare_kwargs(state)
    workload = state.get("workload") or {}
    if workload.get("job_local_override"):
        identity = state.get("chain_identity") or {}
        validated = [
            dict(item)
            for item in validated_contracts_view(state)
            if isinstance(item, dict)
        ]
        chain = str(
            identity.get("canonical") or identity.get("raw") or ""
        ).strip().lower()
        family = str(identity.get("adapter_family") or "").strip().lower()
        override = materialize_custom_rpc_template(
            chain=chain,
            adapter_family=family,
            rpc_mode=str(state.get("rpc_mode") or "single"),
            workload=dict(workload),
            validated_methods=validated,
        )
        if override:
            materialization_evidence = (
                (override.get("_meta") or {}).get("materialization_evidence")
                if isinstance(override.get("_meta"), dict)
                else {}
            )
            if isinstance(materialization_evidence, dict):
                emit_materialization_receipt(state, materialization_evidence)
            prepare_kwargs["chain_config_override"] = override
    prepared_result = execution_service.execute(
        ExecutionRequest(
            operation=ExecutionOperation.PREPARE,
            prepare_kwargs=prepare_kwargs,
        )
    )
    prepared = prepared_result.to_dict()
    runtime_override = prepare_kwargs.get("chain_config_override")
    if isinstance(runtime_override, dict) and runtime_override:
        prepared_data = prepared.setdefault("data", {})
        plan = prepared_data.setdefault("plan", {})
        if not isinstance(plan, dict):
            raise RuntimeError("prepare service returned a non-object plan")
        planned_override = plan.get("chain_config_override")
        if planned_override != runtime_override:
            raise RuntimeError(
                "prepare service omitted or changed the validated "
                "chain_config_override"
            )
    if prepared_result.failure and not (prepared.get("data") or {}).get("preflight"):
        prepared.setdefault("data", {})["preflight"] = {
            "passed": False,
            "blockers": [prepared_result.failure.message],
            "checks": [],
            "warnings": list(prepared_result.warnings),
        }
    return prepared


def _execution_idempotency_key(state: AgentGraphState) -> str:
    request_id = str((state.get("preflight") or {}).get("execution_request_id") or "").strip()
    return f"harness:{request_id}" if request_id else ""


def _job_from_execution_result(result: dict[str, Any]) -> dict[str, Any]:
    job = (result.get("data") or {}).get("job")
    if isinstance(job, dict) and job:
        return dict(job)
    failure = result.get("failure") or {}
    return {
        "status": str(result.get("status") or "failed"),
        "error": str(failure.get("message") or "; ".join(result.get("warnings") or []) or "execution failed"),
        "failure": dict(failure) if isinstance(failure, dict) else {},
    }


def _execution_result_failed(result: dict[str, Any], job: dict[str, Any]) -> bool:
    return str(result.get("status") or "") in {"blocked", "failed"} or str(job.get("status") or "") in {
        "failed",
        "partial",
    }


def _goal_from_mode(mode: str) -> str:
    if mode == "quick":
        return "smoke"
    if mode == "intensive":
        return "stress"
    return "baseline"


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _preflight_recovery_command(state: AgentGraphState) -> RecoveryCommand:
    record = failure_record_from_preflight(
        state.get("preflight") or {},
        confirmed_config=state.get("confirmed_config") or {},
        execution_id=str((state.get("preflight") or {}).get("execution_request_id") or ""),
    )
    return RecoveryCommand(operation="activate", record=record)


def _job_recovery_command(state: AgentGraphState) -> RecoveryCommand:
    record = failure_record_from_job(
        state.get("job") or {},
        confirmed_config=state.get("confirmed_config") or {},
    )
    return RecoveryCommand(operation="activate", record=record)


def _resolved_recovery_command(
    state: AgentGraphState,
    *,
    validation_receipt: str,
) -> RecoveryCommand | None:
    recovery = state.get("failure_recovery") or {}
    if recovery.get("status") != "correcting":
        return None
    return RecoveryCommand(operation="resolve", validation_receipt=validation_receipt)


def _job_fragment(result: dict[str, Any], *, message_id: str) -> ResponseFragment:
    data = result.get("data") or {}
    job = data.get("job") or {}
    commands = data.get("terminal_commands") or {}
    command_text = "; ".join(str(value) for value in commands.values())
    return ResponseFragment(
        kind="status",
        message_id=message_id,
        arguments={
            "status": str(result.get("status") or "unknown"),
            "job_id": str(job.get("job_id") or "<unknown>"),
            "commands": command_text or "jobs/status/logs",
        },
        source=__name__,
    )
