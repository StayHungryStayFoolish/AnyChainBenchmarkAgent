"""Authoritative execution scenarios shared by runtime and evidence tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


RPC_BENCHMARK_WORKFLOW = "rpc_benchmark"
SYNC_OBSERVE_WORKFLOW = "sync_observe"


@dataclass(frozen=True)
class ExecutionScenarioSpec:
    scenario_id: str
    action_type: str
    operation: str
    operation_kind: str
    workflow_type: str
    target_modes: frozenset[str]
    required_artifacts: tuple[str, ...]
    real_evidence_required: bool = True
    forbidden_artifacts: tuple[str, ...] = ()
    required_command_tokens: tuple[str, ...] = ()
    forbidden_command_tokens: tuple[str, ...] = ()


EXECUTION_SCENARIOS: tuple[ExecutionScenarioSpec, ...] = (
    ExecutionScenarioSpec(
        scenario_id="rpc_fake_node_smoke",
        action_type="approve_preflight_smoke",
        operation="fake_node_smoke",
        operation_kind="preflight_smoke",
        workflow_type=RPC_BENCHMARK_WORKFLOW,
        target_modes=frozenset({"fake-node"}),
        required_artifacts=(
            "summary_json",
            "performance_csv",
            "html_report",
            "proxy_method_csv",
            "vegeta_json",
        ),
        required_command_tokens=("--fake-node",),
        forbidden_command_tokens=("--sync-observe",),
    ),
    ExecutionScenarioSpec(
        scenario_id="rpc_real_node_smoke",
        action_type="approve_preflight_smoke",
        operation="real_node_smoke",
        operation_kind="preflight_smoke",
        workflow_type=RPC_BENCHMARK_WORKFLOW,
        target_modes=frozenset({"real-node"}),
        required_artifacts=(
            "summary_json",
            "performance_csv",
            "html_report",
            "proxy_method_csv",
            "vegeta_json",
        ),
        forbidden_command_tokens=("--fake-node", "--sync-observe"),
    ),
    ExecutionScenarioSpec(
        scenario_id="rpc_real_node_final",
        action_type="approve_final_benchmark",
        operation="final_benchmark",
        operation_kind="final_benchmark",
        workflow_type=RPC_BENCHMARK_WORKFLOW,
        target_modes=frozenset({"real-node"}),
        required_artifacts=(
            "summary_json",
            "performance_csv",
            "html_report",
            "proxy_method_csv",
            "vegeta_json",
        ),
        forbidden_command_tokens=("--fake-node", "--sync-observe"),
    ),
    ExecutionScenarioSpec(
        scenario_id="sync_observe_bounded",
        action_type="approve_preflight_smoke",
        operation="sync_observe",
        operation_kind="sync_observe",
        workflow_type=SYNC_OBSERVE_WORKFLOW,
        target_modes=frozenset({"sync-observe"}),
        required_artifacts=(
            "summary_json",
            "performance_csv",
            "html_report_en",
            "html_report_zh",
            "sync_timeline_chart",
            "sync_health_csv",
        ),
        forbidden_artifacts=("vegeta_json", "proxy_method_csv"),
        required_command_tokens=("--sync-observe", "--duration"),
        forbidden_command_tokens=("--quick", "--standard", "--intensive", "--single", "--mixed"),
    ),
)


def workflow_type_from_plan(plan: Mapping[str, Any]) -> str:
    raw = plan.get("workflow_type") or plan.get("run_mode")
    if not raw and isinstance(plan.get("request"), Mapping):
        raw = plan["request"].get("workflow_type") or plan["request"].get("run_mode")
    return str(raw or RPC_BENCHMARK_WORKFLOW).strip().lower().replace("-", "_")


def scenarios_for_action(action_type: str) -> tuple[ExecutionScenarioSpec, ...]:
    return tuple(
        spec
        for spec in EXECUTION_SCENARIOS
        if spec.action_type == action_type and spec.real_evidence_required
    )


def scenario_by_id(scenario_id: str) -> ExecutionScenarioSpec:
    matches = [spec for spec in EXECUTION_SCENARIOS if spec.scenario_id == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"unknown execution scenario: {scenario_id or '<empty>'}")
    return matches[0]


def scenario_for_operation(operation: str, plan: Mapping[str, Any]) -> ExecutionScenarioSpec:
    workflow = validate_operation_workflow(operation, plan)
    matches = [
        spec
        for spec in EXECUTION_SCENARIOS
        if spec.operation == operation and spec.workflow_type == workflow
    ]
    if len(matches) != 1:
        raise ValueError(
            f"no unique execution scenario for operation={operation}, workflow={workflow}"
        )
    return matches[0]


def validate_operation_workflow(operation: str, plan: Mapping[str, Any]) -> str:
    """Fail closed when a submitted operation does not own the plan workflow."""

    workflow = workflow_type_from_plan(plan)
    allowed = {
        "fake_node_smoke": {RPC_BENCHMARK_WORKFLOW},
        "real_node_smoke": {RPC_BENCHMARK_WORKFLOW},
        "final_benchmark": {RPC_BENCHMARK_WORKFLOW},
        "sync_observe": {SYNC_OBSERVE_WORKFLOW},
    }
    expected = allowed.get(operation)
    if expected is not None and workflow not in expected:
        choices = ", ".join(sorted(expected))
        raise ValueError(
            f"operation {operation} does not own workflow {workflow}; expected {choices}"
        )
    return workflow
