"""Reviewed concrete inputs shared by Harness evidence producers.

Question scenarios own product state.  Execution cases own the concrete input
and its expected admission.  Keeping these concerns separate prevents one
evidence lane from using another lane's artifact as an input registry.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.harness_contract_scenarios import (
    ManualInputCase,
    QuestionScenario,
    manual_input_case,
    question_scenarios,
)


REACHABLE_RPC_URL_ENV = "ANYCHAIN_AGENT_TEST_REACHABLE_RPC_URL"
UNREACHABLE_RPC_URL = "http://127.0.0.1:1"
LOCAL_INPUT_CLASSES = frozenset({
    "exact_option",
    "valid_literal",
    "invalid_literal",
    "trimmed_whitespace_punctuation",
    "out_of_range_numeric",
    "reachable_url",
    "unreachable_or_mismatched_url",
})


@dataclass(frozen=True)
class ContractExecutionCase:
    case_id: str
    scenario_id: str
    input_class: str
    option_id: str
    exact_input: str
    input_provider: str
    expected_admitted: bool
    expected_value: Any
    setup_capabilities: tuple[str, ...]
    startup_policy: str
    side_effect_policy: str

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "setup_capabilities": list(self.setup_capabilities),
        }

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self.descriptor)

    def resolve_input(self, environment: Mapping[str, str] | None = None) -> str:
        env = environment if environment is not None else os.environ
        if not self.input_provider:
            return self.exact_input
        if self.input_provider == "reachable_rpc_url":
            value = str(env.get(REACHABLE_RPC_URL_ENV) or "").strip()
            if not value:
                raise RuntimeError(
                    f"execution case {self.case_id} requires {REACHABLE_RPC_URL_ENV}"
                )
            return value
        if self.input_provider == "trimmed_reachable_rpc_url":
            value = str(env.get(REACHABLE_RPC_URL_ENV) or "").strip()
            if not value:
                raise RuntimeError(
                    f"execution case {self.case_id} requires {REACHABLE_RPC_URL_ENV}"
                )
            return f"  {value},  "
        raise RuntimeError(f"unknown execution-case input provider: {self.input_provider}")


def reviewed_execution_case(
    edge: Mapping[str, Any],
) -> tuple[QuestionScenario, ContractExecutionCase] | None:
    """Resolve exactly one reviewed scenario and concrete case for an edge."""

    scenario_ids = tuple(str(item) for item in edge.get("executable_scenario_ids") or ())
    scenarios = _reviewed_scenario_registry()
    candidates: list[tuple[QuestionScenario, ContractExecutionCase]] = []
    for scenario_id in scenario_ids:
        scenario = scenarios.get(scenario_id)
        if scenario is None or not _scenario_matches_edge(scenario, edge):
            continue
        case = _case_for_scenario(scenario, edge)
        if case is not None:
            candidates.append((scenario, case))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            f"edge has multiple reviewed execution cases: {edge.get('edge_key')}"
        )
    return candidates[0]


@lru_cache(maxsize=1)
def _reviewed_scenario_registry() -> dict[str, QuestionScenario]:
    return {
        item.scenario_id: item
        for item in question_scenarios("en")
        if item.executable
    }


def bind_reviewed_execution_case(edge: dict[str, Any]) -> None:
    """Attach immutable case identity and honest local applicability."""

    resolved = reviewed_execution_case(edge)
    if resolved is None:
        edge["execution_case_ids"] = []
        edge["execution_case_hash"] = ""
        if str(edge.get("input_class") or "") in LOCAL_INPUT_CLASSES:
            edge["applicable"] = False
            edge["applicability_reason"] = (
                "no reviewed concrete input exists for this local equivalence class"
            )
        return
    _scenario, case = resolved
    edge["execution_case_ids"] = [case.case_id]
    edge["execution_case_hash"] = case.descriptor_hash
    if edge.get("edge_type") == "manual_input":
        edge["expected_admitted"] = case.expected_admitted
        if case.expected_value is not None:
            edge.setdefault("expected_postcondition", {})["value"] = case.expected_value
        edge["deterministic_case_available"] = (
            not case.input_provider
            and case.side_effect_policy == "none"
            and str(edge.get("input_class") or "") not in {
                "reachable_url",
                "unreachable_or_mismatched_url",
            }
        )


def _scenario_matches_edge(scenario: QuestionScenario, edge: Mapping[str, Any]) -> bool:
    question = scenario.question
    return (
        str(question.get("group") or "") == str(edge.get("group") or "")
        and str(question.get("id") or "") == str(edge.get("question_id") or "")
        and scenario.state_fingerprint == str(edge.get("state_fingerprint") or "")
    )


def _case_for_scenario(
    scenario: QuestionScenario,
    edge: Mapping[str, Any],
) -> ContractExecutionCase | None:
    edge_type = str(edge.get("edge_type") or "")
    input_class = str(edge.get("input_class") or "")
    option_id = str(edge.get("option_id") or "")
    if edge_type == "question_option" and input_class == "exact_option":
        if not any(str(item.get("id") or "") == option_id for item in scenario.question.get("options") or ()):
            return None
        action_type = str(edge.get("action_type") or "")
        return _new_case(
            scenario,
            input_class,
            option_id=option_id,
            exact_input=option_id,
            expected_admitted=True,
            side_effect_policy=(
                "isolated_job_submission"
                if action_type in {"approve_preflight_smoke", "approve_final_benchmark"}
                else "none"
            ),
            setup_capabilities=(
                ("prepared_real_node_plan",)
                if action_type == "approve_final_benchmark"
                else ()
            ),
        )
    if edge_type != "manual_input":
        return None

    case = (
        (scenario.manual_input_overrides or {}).get(input_class)
        or manual_input_case(scenario.question, input_class)
        or _domain_manual_case(scenario.question, input_class)
    )
    if case is None:
        return None
    provider = ""
    exact_input = case.text
    setup: tuple[str, ...] = ()
    if exact_input == "<reachable_rpc_url>":
        exact_input = ""
        provider = "reachable_rpc_url"
        setup = ("local_jsonrpc_endpoint",)
    elif exact_input == "<trimmed_reachable_rpc_url>":
        exact_input = ""
        provider = "trimmed_reachable_rpc_url"
        setup = ("local_jsonrpc_endpoint",)
    return _new_case(
        scenario,
        input_class,
        option_id="",
        exact_input=exact_input,
        input_provider=provider,
        expected_admitted=case.expected_admitted,
        expected_value=(
            {"eth_blockNumber": 50, "eth_gasPrice": 50}
            if str((scenario.question.get("validation") or {}).get("input_mode") or "")
            == "rpc_weights"
            else None
        ),
        setup_capabilities=setup,
    )


def _domain_manual_case(
    question: Mapping[str, Any],
    input_class: str,
) -> ManualInputCase | None:
    kind = str(question.get("kind") or "")
    validation = dict(question.get("validation") or {})
    input_mode = str(validation.get("input_mode") or "")

    if kind == "url":
        if input_class in {"valid_literal", "reachable_url"}:
            return ManualInputCase(input_class, "<reachable_rpc_url>", True)
        if input_class == "trimmed_whitespace_punctuation":
            return ManualInputCase(input_class, "<trimmed_reachable_rpc_url>", True)
        if input_class == "unreachable_or_mismatched_url":
            retains_failed_candidate = str(question.get("id") or "") in {
                "custom_rpc_endpoint",
                "new_chain_endpoint",
            }
            return ManualInputCase(
                input_class,
                UNREACHABLE_RPC_URL,
                retains_failed_candidate,
            )
        return None
    if kind == "chain":
        if input_class == "valid_literal":
            return ManualInputCase(input_class, "solana", True)
        if input_class == "trimmed_whitespace_punctuation":
            return ManualInputCase(input_class, "  solana,  ", True)
        return None
    if kind == "confirm_or_value" and str(question.get("field") or "") == "MAINNET_RPC_URL_REVIEWED":
        if input_class == "valid_literal":
            return ManualInputCase(input_class, "<reachable_rpc_url>", True)
        if input_class == "trimmed_whitespace_punctuation":
            return ManualInputCase(input_class, "<trimmed_reachable_rpc_url>", True)
        return None
    if input_mode == "rpc_method_or_schema_evidence":
        if input_class == "valid_literal":
            return ManualInputCase(input_class, "eth_chainId", True)
        if input_class == "trimmed_whitespace_punctuation":
            return ManualInputCase(input_class, "  eth_chainId,  ", True)
        return None
    if input_mode == "rpc_weights":
        if input_class == "valid_literal":
            return ManualInputCase(
                input_class,
                "eth_blockNumber=50,eth_gasPrice=50",
                True,
            )
        if input_class == "trimmed_whitespace_punctuation":
            return ManualInputCase(
                input_class,
                "  eth_blockNumber=50,eth_gasPrice=50  ",
                True,
            )
        return None
    return None


def _new_case(
    scenario: QuestionScenario,
    input_class: str,
    *,
    option_id: str,
    exact_input: str,
    expected_admitted: bool,
    expected_value: Any = None,
    input_provider: str = "",
    setup_capabilities: tuple[str, ...] = (),
    side_effect_policy: str = "none",
) -> ContractExecutionCase:
    identity = {
        "scenario_id": scenario.scenario_id,
        "state_fingerprint": scenario.state_fingerprint,
        "question_id": str(scenario.question.get("id") or ""),
        "input_class": input_class,
        "option_id": option_id,
        "exact_input": exact_input,
        "input_provider": input_provider,
        "expected_admitted": expected_admitted,
        "expected_value": expected_value,
        "setup_capabilities": list(setup_capabilities),
        "startup_policy": "direct_or_resume",
        "side_effect_policy": side_effect_policy,
    }
    return ContractExecutionCase(
        case_id="case-" + content_hash(identity)[:20],
        scenario_id=scenario.scenario_id,
        input_class=input_class,
        option_id=option_id,
        exact_input=exact_input,
        input_provider=input_provider,
        expected_admitted=expected_admitted,
        expected_value=expected_value,
        setup_capabilities=setup_capabilities,
        startup_policy="direct_or_resume",
        side_effect_policy=side_effect_policy,
    )
