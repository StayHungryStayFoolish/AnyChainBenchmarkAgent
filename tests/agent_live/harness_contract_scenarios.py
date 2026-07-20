"""Catalog and explicitly executable Harness contract scenarios.

Catalog scenarios enumerate known question factories.  A scenario is
executable only when it also supplies the exact seed state that produced the
question.  This distinction prevents catalog discovery from becoming path
coverage by assertion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping
from unittest.mock import patch

from agent.harness.domains.environment import question_for_environment
from agent.harness.domains.chain_rpc import question_for_chain_rpc
from agent.harness.domains.execution import question_for_execution
from agent.harness.domains.orientation import opening_question
from agent.harness.domains.performance import question_for_performance
from agent.harness.domains.recovery import question_for_recovery
from agent.harness.domains.sync_observe import question_for_sync_observe
from agent.harness.state import AgentGraphState, new_state
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.graph_turn import invoke_product_graph_turn


def canonical_question_contract(question: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime identity while preserving the complete question contract."""

    stable = deepcopy(dict(question))
    stable.pop("execution_request_id", None)
    return stable


def canonical_scenario_state(seed_state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-generated identity fields from a reviewed scenario seed."""

    stable = deepcopy(dict(seed_state))
    session = stable.get("session")
    if isinstance(session, dict):
        session.pop("created_at", None)
        session.pop("updated_at", None)
    pending = stable.get("pending_question")
    if isinstance(pending, dict):
        stable["pending_question"] = canonical_question_contract(pending)
    return stable


@dataclass(frozen=True)
class QuestionScenario:
    scenario_id: str
    question: Mapping[str, Any]
    seed_state: Mapping[str, Any] | None = None
    source: str = "catalog"
    manual_postcondition_path: str = ""
    option_postcondition_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    option_relation_overrides: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    manual_input_overrides: Mapping[str, "ManualInputCase"] = field(default_factory=dict)
    manual_next_question_ids: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.seed_state is not None

    @property
    def state_fingerprint(self) -> str:
        if self.seed_state is None:
            return content_hash({"catalog_scenario_id": self.scenario_id})
        return content_hash(canonical_scenario_state(self.seed_state))


@dataclass(frozen=True)
class ManualInputCase:
    input_class: str
    text: str
    expected_admitted: bool


@dataclass(frozen=True)
class ActionTransitionScenario:
    """Reviewed seed for a coordinator action without a pending question."""

    scenario_id: str
    action_type: str
    seed_state: Mapping[str, Any]

    @property
    def state_fingerprint(self) -> str:
        return content_hash(canonical_scenario_state(self.seed_state))


def action_transition_scenarios(language: str = "en") -> list[ActionTransitionScenario]:
    """Return minimal valid seeds for semantic coordinator transitions."""

    def base(suffix: str) -> AgentGraphState:
        state = new_state(
            f"coverage-action-{suffix}",
            language=language,
            session_purpose="coverage",
        )
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "status": "confirmed",
        }
        return state

    change_group = base("change-group")

    go_back = base("go-back")
    go_back["active_group"] = "qps_profile"
    go_back["group_history"] = ["workload_rpc"]

    queue_goal = base("queue-workflow-goal")

    queued_goal = {
        "target_mode": "sync-observe",
        "goal": "observe node synchronization after the current benchmark",
        "source_evidence": "run sync-observe after this benchmark",
    }
    activate_goal = base("activate-workflow-goal")
    activate_goal["workflow_goals"] = [deepcopy(queued_goal)]

    discard_goal = base("discard-workflow-goal")
    discard_goal["workflow_goals"] = [deepcopy(queued_goal)]

    scenarios = [
        ActionTransitionScenario("action_change_group", "change_group", change_group),
        ActionTransitionScenario("action_go_back", "go_back", go_back),
        ActionTransitionScenario(
            "action_queue_workflow_goal",
            "queue_workflow_goal",
            queue_goal,
        ),
        ActionTransitionScenario(
            "action_activate_next_workflow_goal",
            "activate_next_workflow_goal",
            activate_goal,
        ),
        ActionTransitionScenario(
            "action_discard_next_workflow_goal",
            "discard_next_workflow_goal",
            discard_goal,
        ),
    ]
    scenario_ids = [item.scenario_id for item in scenarios]
    action_types = [item.action_type for item in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise AssertionError("action-transition scenario registry has duplicate ids")
    if len(action_types) != len(set(action_types)):
        raise AssertionError("action-transition scenario registry has duplicate action types")
    return sorted(scenarios, key=lambda item: item.scenario_id)


def question_scenarios(language: str = "en") -> list[QuestionScenario]:
    """Return the standalone reviewed catalog plus executable seed states."""

    explicit = _explicit_scenarios(language)
    catalog = _catalog_only_scenarios(language)
    overlap = set(explicit) & set(catalog)
    if overlap:
        raise AssertionError(f"scenario registry has duplicate ids: {sorted(overlap)}")
    scenarios = [*explicit.values(), *catalog.values()]
    scenarios.extend(_runtime_execution_scenarios(language))
    return sorted(scenarios, key=lambda item: item.scenario_id)


def _compiled_action_state(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    user_text: str,
) -> AgentGraphState:
    """Return the graph-owned state that rendered a control question."""

    prepared = deepcopy(dict(state))
    prepared["last_user_input"] = user_text
    with patch(
        "agent.harness.coordinator.resolve_action_queue",
        return_value={"actions": [dict(action)]},
    ):
        result = invoke_product_graph_turn(prepared)
    question = dict(result.get("pending_question") or {})
    if not question:
        raise AssertionError(f"compiled action produced no pending question: {action}")
    return result


def manual_input_case(question: Mapping[str, Any], input_class: str) -> ManualInputCase | None:
    """Return a concrete local input only when its expected behavior is known."""

    validation = dict(question.get("validation") or {})
    value_type = str(validation.get("value_type") or "")
    if input_class == "valid_literal":
        value = _valid_literal(validation)
        option_ids = {str(item.get("id") or "") for item in question.get("options") or []}
        if value in option_ids and value_type in {"positive_number", "positive_integer"}:
            value = "7"
        return ManualInputCase(input_class, value, True) if value is not None else None
    if input_class == "trimmed_whitespace_punctuation" and value_type == "scalar_token":
        return ManualInputCase(input_class, "  n2-standard-16,  ", True)
    if (
        input_class == "invalid_literal"
        and value_type == "scalar_token"
        and int(validation.get("max_length") or 0) > 0
    ):
        return ManualInputCase(
            input_class,
            "x" * (int(validation["max_length"]) + 1),
            False,
        )
    if input_class == "invalid_literal" and value_type in {"positive_number", "positive_integer"}:
        return ManualInputCase(input_class, "not-a-number", False)
    if input_class == "trimmed_whitespace_punctuation" and value_type in {"positive_number", "positive_integer"}:
        return ManualInputCase(input_class, "  7  ", True)
    if input_class == "out_of_range_numeric" and value_type in {"positive_number", "positive_integer"}:
        return ManualInputCase(input_class, "0", False)
    if input_class == "structured_json_yaml_env_curl" and value_type == "json":
        return ManualInputCase(input_class, '{"key":"value"}', True)
    return None


def _explicit_scenarios(language: str) -> dict[str, QuestionScenario]:
    scenarios: dict[str, QuestionScenario] = {}

    def add(
        scenario_id: str,
        state: AgentGraphState,
        question: Mapping[str, Any] | None,
        *,
        source: str = "reviewed_seed",
        manual_postcondition_path: str = "",
        manual_next_question_ids: tuple[str, ...] = (),
        option_postcondition_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        option_relation_overrides: Mapping[str, tuple[Mapping[str, Any], ...]] | None = None,
        manual_input_overrides: Mapping[str, ManualInputCase] | None = None,
    ) -> None:
        if not question:
            return
        state["active_group"] = str(question.get("group") or state.get("active_group") or "opening")
        state["pending_question"] = deepcopy(dict(question))
        scenarios[scenario_id] = QuestionScenario(
            scenario_id=scenario_id,
            question=deepcopy(dict(question)),
            seed_state=deepcopy(state),
            source=source,
            manual_postcondition_path=manual_postcondition_path,
            option_postcondition_overrides=deepcopy(dict(option_postcondition_overrides or {})),
            option_relation_overrides=deepcopy(dict(option_relation_overrides or {})),
            manual_input_overrides=deepcopy(dict(manual_input_overrides or {})),
            manual_next_question_ids=tuple(manual_next_question_ids),
        )

    opening_state = new_state("coverage-opening", language=language, session_purpose="coverage")
    opening = opening_question(opening_state)
    add("opening", opening_state, opening)

    from agent.harness.domains.orientation import resume_question

    resume_relations = {
        "1": (
            {
                "kind": "path_equals_before_path",
                "after_path": "active_group",
                "before_path": "resume_context.active_group",
            },
            {
                "kind": "prefix_equals_before_prefix",
                "after_prefix": "pending_question",
                "before_prefix": "resume_context.pending_question",
                "ignored_suffixes": ["created_turn_index", "resume_action_queue"],
            },
            {
                "kind": "path_equals_before_path",
                "after_path": "action_queue",
                "before_path": "action_queue",
            },
        )
    }
    resume_state = new_state("coverage-resume", language=language, session_purpose="coverage")
    resume_state["target_mode"] = "fake-node"
    saved_chain_question = question_for_chain_rpc(resume_state, "chain_identity") or {}
    resume_state["resume_context"] = {
        "active_group": "chain_identity",
        "pending_question": deepcopy(dict(saved_chain_question)),
    }
    add(
        "resume",
        resume_state,
        resume_question(resume_state),
        option_postcondition_overrides={"1": {"resume_context": {}}},
        option_relation_overrides=resume_relations,
    )

    resume_qps_state = new_state("coverage-resume-qps", language=language, session_purpose="coverage")
    resume_qps_state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "qps_profile": {"mode": "quick", "confirmed": False, "default_decision_made": False},
    })
    saved_qps_question = question_for_performance(resume_qps_state, "qps_profile") or {}
    resume_qps_state["resume_context"] = {
        "active_group": "qps_profile",
        "pending_question": deepcopy(dict(saved_qps_question)),
    }
    add(
        "resume_qps",
        resume_qps_state,
        resume_question(resume_qps_state),
        option_postcondition_overrides={"1": {"resume_context": {}}},
        option_relation_overrides=resume_relations,
    )

    detected_state = new_state("coverage-provider-detected", language=language, session_purpose="coverage")
    detected_state["discovery"] = {"cloud": {"region": "test-region"}}
    add(
        "provider_detected",
        detected_state,
        question_for_environment(detected_state, "provider_deployment"),
    )

    zone_state = new_state("coverage-provider-zone", language=language, session_purpose="coverage")
    zone_state["confirmed_config"] = {"CLOUD_REGION": "test-region"}
    add("provider_zone", zone_state, question_for_environment(zone_state, "provider_deployment"))

    provider_state = new_state("coverage-provider-machine", language=language, session_purpose="coverage")
    provider_state["confirmed_config"] = {
        "CLOUD_REGION": "test-region",
        "CLOUD_ZONE": "test-zone",
    }
    provider = question_for_environment(provider_state, "provider_deployment")
    if provider:
        add("provider_machine", provider_state, provider)

    accounts_presence = new_state("coverage-accounts-presence", language=language, session_purpose="coverage")
    add(
        "accounts_presence",
        accounts_presence,
        question_for_environment(accounts_presence, "accounts_disk"),
    )

    environment_seeds = (
        ("ledger_ledger_device", "ledger_disk", {}),
        ("ledger_data_vol_type", "ledger_disk", {"LEDGER_DEVICE": "vda"}),
        (
            "ledger_data_vol_max_iops",
            "ledger_disk",
            {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk", "DATA_VOL_SIZE": "1"},
        ),
        (
            "ledger_data_vol_max_throughput",
            "ledger_disk",
            {
                "LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk",
                "DATA_VOL_SIZE": "1", "DATA_VOL_MAX_IOPS": "1",
            },
        ),
        ("accounts_accounts_device", "accounts_disk", {"has_accounts_device": True}),
        (
            "accounts_accounts_vol_type",
            "accounts_disk",
            {"has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb"},
        ),
        (
            "accounts_accounts_vol_size",
            "accounts_disk",
            {
                "has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb",
                "ACCOUNTS_VOL_TYPE": "test-disk",
            },
        ),
        (
            "accounts_accounts_vol_max_iops",
            "accounts_disk",
            {
                "has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb",
                "ACCOUNTS_VOL_TYPE": "test-disk", "ACCOUNTS_VOL_SIZE": "1",
            },
        ),
        (
            "accounts_accounts_vol_max_throughput",
            "accounts_disk",
            {
                "has_accounts_device": True, "ACCOUNTS_DEVICE": "vdb",
                "ACCOUNTS_VOL_TYPE": "test-disk", "ACCOUNTS_VOL_SIZE": "1",
                "ACCOUNTS_VOL_MAX_IOPS": "1",
            },
        ),
        ("network_interface", "network", {}),
        ("network_bandwidth", "network", {"NETWORK_INTERFACE": "eth0"}),
    )
    for scenario_id, group, confirmed in environment_seeds:
        state = new_state(f"coverage-{scenario_id}", language=language, session_purpose="coverage")
        state["confirmed_config"] = dict(confirmed)
        add(scenario_id, state, question_for_environment(state, group))

    qps_state = new_state("coverage-qps-mode", language=language, session_purpose="coverage")
    qps_state["target_mode"] = "fake-node"
    qps_state["workflow_mode"] = "rpc_benchmark"
    qps = question_for_performance(qps_state, "qps_profile")
    if qps:
        add("qps_mode", qps_state, qps)

    sync_stop_state = new_state(
        "coverage-sync-stop-condition",
        language=language,
        session_purpose="coverage",
    )
    sync_stop_state.update({
        "target_mode": "sync-observe",
        "workflow_mode": "sync_observe",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "sync_observe": {"source": "endpoint_only"},
        "confirmed_config": {
            "SYNC_OBSERVE_RPC_URL": "http://geth-dev:8545",
            "MAINNET_RPC_URL_REVIEWED": True,
        },
        "endpoint_evidence": {"sync_rpc_url_ready": True},
    })
    add(
        "sync_stop",
        sync_stop_state,
        question_for_sync_observe(sync_stop_state),
    )

    sync_duration_state = deepcopy(sync_stop_state)
    sync_duration_state["thread_id"] = "coverage-sync-duration"
    sync_duration_state["sync_observe"] = {
        **dict(sync_duration_state.get("sync_observe") or {}),
        "stop_condition": "duration",
    }
    add(
        "sync_duration",
        sync_duration_state,
        question_for_sync_observe(sync_duration_state),
        manual_postcondition_path="sync_observe.duration_seconds",
    )

    performance_seeds = (
        ("qps_confirm", "qps_profile", {"qps_profile": {"mode": "quick"}}),
        (
            "qps_adjust",
            "qps_profile",
            {"qps_profile": {"mode": "quick", "default_decision_made": True}},
        ),
        (
            "qps_adjust_value",
            "qps_profile",
            {
                "qps_profile": {
                    "mode": "quick", "default_decision_made": True,
                    "adjust_field": "INITIAL_QPS",
                }
            },
        ),
        ("advanced_confirm", "advanced_tuning", {}),
        (
            "advanced_adjust",
            "advanced_tuning",
            {"advanced_tuning": {"default_decision_made": True}},
        ),
        (
            "advanced_adjust_value",
            "advanced_tuning",
            {"advanced_tuning": {"default_decision_made": True, "adjust_field": "MONITOR_INTERVAL"}},
        ),
    )
    for scenario_id, group, updates in performance_seeds:
        state = new_state(f"coverage-{scenario_id}", language=language, session_purpose="coverage")
        state.update(deepcopy(updates))
        manual_path = {
            "qps_adjust_value": "qps_profile.overrides.INITIAL_QPS",
            "advanced_adjust_value": "advanced_tuning.overrides.MONITOR_INTERVAL",
        }.get(scenario_id, "")
        add(
            scenario_id,
            state,
            question_for_performance(state, group),
            manual_postcondition_path=manual_path,
        )

    observability_state = new_state("coverage-observability", language=language, session_purpose="coverage")
    observability = question_for_performance(observability_state, "observability")
    if observability:
        add("observability", observability_state, observability)

    sync_seeds = (
        ("sync_source", {}),
        ("sync_after_setup", {"source": "client_setup"}),
    )
    for scenario_id, sync_values in sync_seeds:
        state = new_state(f"coverage-{scenario_id}", language=language, session_purpose="coverage")
        state["workflow_mode"] = "sync_observe"
        state["sync_observe"] = dict(sync_values)
        add(
            scenario_id,
            state,
            question_for_sync_observe(state),
            manual_postcondition_path=(
                "sync_observe.duration_seconds" if scenario_id == "sync_duration" else ""
            ),
        )

    recovery_state = new_state("coverage-failure-recovery", language=language, session_purpose="coverage")
    recovery_state["failure_recovery"] = {
        "status": "pending",
        "record": {
            "code": "ENDPOINT_UNREACHABLE",
            "summary": "endpoint failed",
            "allowed_actions": ["correct_failure", "inspect_failure", "cancel_failure_recovery"],
        },
    }
    add(
        "failure_recovery",
        recovery_state,
        question_for_recovery(recovery_state, "failure_recovery"),
    )

    chain_seeds: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
        ("target_mode", "target_mode", {}),
        ("chain_manual", "chain_identity", {}),
        (
            "adapter_family",
            "chain_identity",
            {
                "chain_identity": {
                    "raw": "sola",
                    "canonical": "sola",
                    "status": "needs_adapter_family_confirmation",
                }
            },
        ),
        (
            "case3_next",
            "chain_identity",
            {
                "chain_identity": {"status": "case3_collecting_evidence"},
                "secondary_handoff": {"evidence": ["evidence"]},
            },
        ),
        (
            "case3_evidence",
            "chain_identity",
            {"chain_identity": {"status": "case3_needs_evidence"}},
        ),
        (
            "custom_adapter",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {"status": "needs_adapter_family_confirmation"},
            },
        ),
        (
            "custom_schema",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {
                    "status": "schema_needs_confirmation",
                    "schema_draft": {"method": "eth_blockNumber", "params": []},
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [],
                        "finished": False,
                        "draft": {
                            "contract_version": 1,
                            "revision": 1,
                            "phase": "request_confirmation",
                            "method": "eth_blockNumber",
                            "params": [],
                            "params_json": [],
                            "confirmed_parameters": [],
                            "request_confirmed": False,
                            "response_confirmed": False,
                        },
                    },
                },
            },
        ),
        (
            "custom_response",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {
                    "status": "response_needs_confirmation",
                    "observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'},
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [],
                        "finished": False,
                        "draft": {
                            "contract_version": 1,
                            "revision": 1,
                            "phase": "response_confirmation",
                            "method": "eth_blockNumber",
                            "params": [],
                            "params_json": [],
                            "confirmed_parameters": [],
                            "request_confirmed": True,
                            "response_confirmed": False,
                            "probe": {"ready": True},
                            "observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'},
                        },
                    },
                },
            },
        ),
        (
            "custom_continue",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {
                    "status": "method_validated_next",
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [{"method": "eth_blockNumber", "request_confirmed": True, "response_confirmed": True}],
                        "finished": False,
                        "draft": {"phase": "validated", "method": "eth_blockNumber"},
                    },
                },
            },
        ),
        (
            "custom_scope",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {
                    "status": "needs_scope",
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [{"method": "eth_blockNumber", "request_confirmed": True, "response_confirmed": True}],
                        "finished": True,
                        "draft": {},
                    },
                },
            },
        ),
        (
            "custom_single_method",
            "endpoint_process",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "custom_rpc": {
                    "status": "needs_single_method",
                    "catalog": {
                        "contract_version": 1,
                        "revision": 2,
                        "methods": [
                            {"method": "eth_blockNumber"}, {"method": "eth_gasPrice"},
                        ],
                        "finished": True,
                        "draft": {},
                    },
                },
            },
        ),
        (
            "new_chain_schema",
            "endpoint_process",
            {
                "chain_identity": {
                    "canonical": "new-chain",
                    "status": "existing_family_schema_needs_confirmation",
                    "schema_draft": {"method": "eth_blockNumber", "params": []},
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [],
                        "finished": False,
                        "draft": {
                            "contract_version": 1,
                            "revision": 1,
                            "phase": "request_confirmation",
                            "method": "eth_blockNumber",
                            "params": [],
                            "params_json": [],
                            "confirmed_parameters": [],
                            "request_confirmed": False,
                            "response_confirmed": False,
                        },
                    },
                },
            },
        ),
        (
            "new_chain_response",
            "endpoint_process",
            {
                "chain_identity": {
                    "canonical": "new-chain",
                    "status": "existing_family_response_needs_confirmation",
                    "candidate_method": "eth_blockNumber",
                    "observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'},
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [],
                        "finished": False,
                        "draft": {
                            "contract_version": 1,
                            "revision": 1,
                            "phase": "response_confirmation",
                            "method": "eth_blockNumber",
                            "params": [],
                            "params_json": [],
                            "confirmed_parameters": [],
                            "request_confirmed": True,
                            "response_confirmed": False,
                            "probe": {"ready": True},
                            "observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x1"}'},
                        },
                    },
                },
            },
        ),
        (
            "new_chain_continue",
            "endpoint_process",
            {
                "chain_identity": {
                    "canonical": "new-chain",
                    "status": "existing_family_method_validated_next",
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [{"method": "eth_blockNumber", "request_confirmed": True, "response_confirmed": True}],
                        "finished": False,
                        "draft": {"phase": "validated", "method": "eth_blockNumber"},
                    },
                },
            },
        ),
        (
            "new_chain_scope",
            "endpoint_process",
            {
                "chain_identity": {
                    "canonical": "new-chain",
                    "status": "existing_family_needs_workload_scope",
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 1,
                        "methods": [{"method": "eth_blockNumber", "request_confirmed": True, "response_confirmed": True}],
                        "finished": True,
                        "draft": {},
                    },
                },
            },
        ),
        (
            "new_chain_single_method",
            "endpoint_process",
            {
                "chain_identity": {
                    "canonical": "new-chain",
                    "status": "existing_family_needs_single_method",
                },
                "custom_rpc": {
                    "catalog": {
                        "contract_version": 1,
                        "revision": 2,
                        "methods": [
                            {"method": "eth_blockNumber"}, {"method": "eth_gasPrice"},
                        ],
                    },
                },
            },
        ),
        (
            "rpc_mode",
            "workload_rpc",
            {"chain_identity": {"canonical": "bsc", "status": "confirmed"}},
        ),
        (
            "workload_single",
            "workload_rpc",
            {
                "target_mode": "fake-node",
                "workflow_mode": "rpc_benchmark",
                "rpc_mode": "single",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            },
        ),
        (
            "workload_mixed",
            "workload_rpc",
            {
                "target_mode": "fake-node",
                "workflow_mode": "rpc_benchmark",
                "rpc_mode": "mixed",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            },
        ),
        (
            "new_chain_runtime",
            "target_samples_fixtures",
            {
                "target_mode": "fake-node",
                "workflow_mode": "rpc_benchmark",
                "chain_identity": {"status": "existing_family_runtime_choice"},
            },
        ),
        (
            "missing_custom_fixture",
            "target_samples_fixtures",
            {
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "rpc_mode": "single",
                "fixture_evidence": {"status": "missing", "missing": [{"method": "eth_accounts"}]},
            },
        ),
    )
    for scenario_id, group, updates in chain_seeds:
        state = new_state(f"coverage-{scenario_id}", language=language, session_purpose="coverage")
        state.update(deepcopy(dict(updates)))
        add(
            scenario_id,
            state,
            question_for_chain_rpc(state, group),
            manual_postcondition_path=(
                "chain_identity.canonical" if scenario_id == "chain_manual" else ""
            ),
        )
    return scenarios


def _catalog_only_scenarios(language: str) -> dict[str, QuestionScenario]:
    """Construct reviewed runtime variants not reached by the default seeds."""

    from agent.harness.domains.chain_identity import _chain_ambiguity_question
    from agent.harness.domains.chain_rpc_questions import _target_change_scope_question
    from agent.harness.domains.environment import config_proposal_review_question
    from agent.harness.domains.orientation import resume_question

    output: dict[str, QuestionScenario] = {}

    def catalog(
        scenario_id: str,
        state: AgentGraphState,
        question: Mapping[str, Any] | None,
        *,
        manual_postcondition_path: str = "",
        manual_next_question_ids: tuple[str, ...] = (),
        option_postcondition_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        option_relation_overrides: Mapping[str, tuple[Mapping[str, Any], ...]] | None = None,
        manual_input_overrides: Mapping[str, ManualInputCase] | None = None,
    ) -> None:
        if question:
            seed = deepcopy(state)
            seed["active_group"] = str(question.get("group") or seed.get("active_group") or "opening")
            seed["pending_question"] = deepcopy(dict(question))
            output[scenario_id] = QuestionScenario(
                scenario_id,
                deepcopy(dict(question)),
                seed,
                source="reviewed_variant_seed",
                manual_postcondition_path=manual_postcondition_path,
                option_postcondition_overrides=deepcopy(
                    dict(option_postcondition_overrides or {})
                ),
                option_relation_overrides=deepcopy(dict(option_relation_overrides or {})),
                manual_input_overrides=deepcopy(dict(manual_input_overrides or {})),
                manual_next_question_ids=tuple(manual_next_question_ids),
            )

    state = new_state("catalog-resume-quarantine", language=language, session_purpose="coverage")
    state["checkpoint_recovery"] = {"status": "quarantined", "reason": "partial"}
    catalog("resume_quarantine", state, resume_question(state))

    inferred_values = {
        "provider_deployment": {"CLOUD_REGION": "test-region"},
        "ledger_disk": {"LEDGER_DEVICE": "vda"},
        "accounts_disk": {"has_accounts_device": False},
        "network": {"NETWORK_INTERFACE": "eth0"},
    }
    for group, config_values in inferred_values.items():
        proposal = {"config_values": dict(config_values)}
        review_state = new_state(
            f"catalog-inferred-{group}", language=language, session_purpose="coverage"
        )
        review_state["inferred_config"] = {"pending_review": deepcopy(proposal)}
        applied = {
            f"confirmed_config.{key}": value for key, value in config_values.items()
        }
        catalog(
            f"inferred_config_{group}",
            review_state,
            config_proposal_review_question(
                group,
                proposal,
                language=language,
            ),
            option_postcondition_overrides={
                "1": applied,
                "2": {},
            },
            option_relation_overrides={
                "1": ({"kind": "path_absent_after", "path": "inferred_config.pending_review"},),
                "2": ({"kind": "path_absent_after", "path": "inferred_config.pending_review"},),
            },
        )

    state = new_state("catalog-ledger-size", language=language, session_purpose="coverage")
    state["confirmed_config"] = {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk"}
    state["discovery"] = {
        "disks": {"candidates": [{"name": "vda", "size": "100G", "type": "disk"}]}
    }
    catalog(
        "ledger_size_inference",
        state,
        question_for_environment(state, "ledger_disk"),
        manual_postcondition_path="confirmed_config.DATA_VOL_SIZE",
    )

    state = new_state("catalog-target-mode-change", language=language, session_purpose="coverage")
    state.update({"target_mode": "fake-node", "active_group": "qps_profile"})
    state = _compiled_action_state(
        state,
        {
            "type": "choose_target_mode",
            "target_mode": "real-node",
            "target_mode_explicit": True,
            "source_evidence": "real-node",
            "confidence": "high",
        },
        user_text="switch to real-node",
    )
    catalog("target_mode_change", state, state.get("pending_question"))

    state = new_state("catalog-unknown-chain", language=language, session_purpose="coverage")
    state["target_mode"] = "fake-node"
    state["workflow_mode"] = "rpc_benchmark"
    state = _compiled_action_state(
        state,
        {
            "type": "choose_chain",
            "chain_text": "sola",
            "source_evidence": "sola",
            "chain_exists": False,
            "possible_known_chain": "solana",
            "confidence": "high",
        },
        user_text="test sola",
    )
    catalog("unknown_chain_identity", state, state.get("pending_question"))

    state = new_state("catalog-chain-change", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
    })
    state = _compiled_action_state(
        state,
        {
            "type": "change_chain",
            "chain_text": "ethereum",
            "source_evidence": "ethereum",
            "confidence": "high",
        },
        user_text="change chain to ethereum",
    )
    catalog("chain_change", state, state.get("pending_question"))

    state = new_state("catalog-chain-ambiguity", language=language, session_purpose="coverage")
    catalog(
        "chain_ambiguity",
        state,
        _chain_ambiguity_question(state, "bsc", {"chain_candidates": ["bsc", "ethereum"]}),
    )

    endpoint_states = (
        (
            "endpoint_local_rpc_url",
            {
                "target_mode": "real-node",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            },
        ),
        (
            "endpoint_blockchain_process_names",
            {
                "target_mode": "real-node",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "endpoint_evidence": {"local_rpc_url_ready": True},
            },
        ),
        (
            "endpoint_sync_observe_rpc_url",
            {
                "target_mode": "sync-observe",
                "workflow_mode": "sync_observe",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                "sync_observe": {"source": "endpoint_only"},
            },
        ),
    )
    for scenario_id, updates in endpoint_states:
        state = new_state(f"catalog-{scenario_id}", language=language, session_purpose="coverage")
        state.update(deepcopy(updates))
        catalog(scenario_id, state, question_for_chain_rpc(state, "endpoint_process"))

    state = new_state("catalog-chain-change-input", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "rpc_mode": "single",
        "workload": {"confirmed": True},
    })
    state = _compiled_action_state(
        state,
        {"type": "request_chain_selection", "source_evidence": "change chain"},
        user_text="change chain",
    )
    catalog(
        "chain_change_input",
        state,
        state.get("pending_question"),
        manual_postcondition_path="chain_identity.change_candidate.canonical",
        manual_input_overrides={
            "valid_literal": ManualInputCase("valid_literal", "solana", True),
            "trimmed_whitespace_punctuation": ManualInputCase(
                "trimmed_whitespace_punctuation", "  solana  ", True
            ),
        },
    )

    state = new_state("catalog-mainnet-review", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "real-node",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "endpoint_evidence": {"local_rpc_url_ready": True},
        "confirmed_config": {"BLOCKCHAIN_PROCESS_NAMES": "node"},
    })
    catalog("mainnet_review", state, question_for_chain_rpc(state, "endpoint_process"))

    state = new_state("catalog-chain-api-key", language=language, session_purpose="coverage")
    state["chain_identity"] = {"canonical": "litecoin", "status": "confirmed"}
    catalog(
        "chain_auxiliary_api_key",
        state,
        question_for_chain_rpc(state, "chain_auxiliary_endpoints"),
    )

    validated = [{"method": "eth_blockNumber"}, {"method": "eth_gasPrice"}]
    catalog_envelope = {
        "contract_version": 1,
        "revision": 2,
        "methods": validated,
    }
    custom_statuses = (
        ("custom_needs_endpoint", "needs_endpoint", {}),
        ("custom_needs_method", "needs_method", {"endpoint_ready": True}),
        (
            "custom_needs_schema_evidence",
            "needs_schema_evidence",
            {
                "endpoint_ready": True,
                "draft": {
                    "contract_version": 1,
                    "phase": "evidence",
                    "method": "eth_blockNumber",
                },
            },
        ),
        ("custom_needs_weights", "needs_weights", {"validated_methods": validated}),
    )
    for scenario_id, status, extra in custom_statuses:
        runtime_extra = {key: value for key, value in extra.items() if key != "draft"}
        state = new_state(f"catalog-{scenario_id}", language=language, session_purpose="coverage")
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed"}
        state["custom_rpc"] = {
            "status": status,
            "catalog": {
                **deepcopy(catalog_envelope),
                "methods": deepcopy(extra.get("validated_methods") or []),
                **(
                    {"draft": deepcopy(extra["draft"])}
                    if isinstance(extra.get("draft"), dict)
                    else {}
                ),
            },
            **deepcopy(runtime_extra),
        }
        manual_path = ""
        next_ids: tuple[str, ...] = ()
        if scenario_id == "custom_needs_method":
            manual_path = "custom_rpc.catalog.draft.method"
            next_ids = (
                "custom_rpc_parameter_confirm",
                "custom_rpc_schema_evidence",
                "custom_rpc_schema_confirm",
                "custom_rpc_response_confirm",
            )
        elif scenario_id == "custom_needs_schema_evidence":
            manual_path = "custom_rpc.catalog.last_transition.command"
            next_ids = (
                "custom_rpc_parameter_confirm",
                "custom_rpc_schema_confirm",
                "custom_rpc_response_confirm",
            )
        elif scenario_id == "custom_needs_weights":
            manual_path = "custom_rpc.weights"
        elif scenario_id == "custom_needs_endpoint":
            manual_path = "custom_rpc.endpoint"
            next_ids = ("custom_rpc_endpoint", "custom_rpc_method")
        catalog(
            scenario_id,
            state,
            question_for_chain_rpc(state, "endpoint_process"),
            manual_postcondition_path=manual_path,
            manual_next_question_ids=next_ids,
        )

    new_chain_statuses = (
        ("new_chain_existing_family_needs_endpoint", "existing_family_needs_endpoint", {}),
        ("new_chain_existing_family_needs_method", "existing_family_needs_method", {}),
        (
            "new_chain_existing_family_needs_schema_evidence",
            "existing_family_needs_schema_evidence",
            {
                "candidate_method": "eth_blockNumber",
                "draft": {
                    "contract_version": 1,
                    "phase": "evidence",
                    "method": "eth_blockNumber",
                },
            },
        ),
        (
            "new_chain_existing_family_needs_weights",
            "existing_family_needs_weights",
            {"validated_methods": validated},
        ),
    )
    for scenario_id, status, extra in new_chain_statuses:
        runtime_extra = {key: value for key, value in extra.items() if key != "draft"}
        state = new_state(f"catalog-{scenario_id}", language=language, session_purpose="coverage")
        state["chain_identity"] = {
            "canonical": "new-chain",
            "status": status,
            **deepcopy(runtime_extra),
        }
        state["custom_rpc"] = {
            "catalog": {
                **deepcopy(catalog_envelope),
                "methods": deepcopy(extra.get("validated_methods") or []),
                **(
                    {"draft": deepcopy(extra["draft"])}
                    if isinstance(extra.get("draft"), dict)
                    else {}
                ),
            }
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint_ready": status != "existing_family_needs_endpoint"
        }
        manual_path = ""
        next_ids: tuple[str, ...] = ()
        if scenario_id == "new_chain_existing_family_needs_method":
            manual_path = "custom_rpc.catalog.draft.method"
            next_ids = (
                "new_chain_parameter_confirm",
                "new_chain_schema_evidence",
                "new_chain_schema_confirm",
                "new_chain_response_confirm",
            )
        elif scenario_id == "new_chain_existing_family_needs_schema_evidence":
            manual_path = "custom_rpc.catalog.last_transition.command"
            next_ids = (
                "new_chain_parameter_confirm",
                "new_chain_schema_confirm",
                "new_chain_response_confirm",
            )
        elif scenario_id == "new_chain_existing_family_needs_weights":
            manual_path = "chain_identity.weights"
        elif scenario_id == "new_chain_existing_family_needs_endpoint":
            manual_path = "endpoint_evidence.candidate_endpoint"
            next_ids = ("new_chain_endpoint", "new_chain_method")
        catalog(
            scenario_id,
            state,
            question_for_chain_rpc(state, "endpoint_process"),
            manual_postcondition_path=manual_path,
            manual_next_question_ids=next_ids,
        )

    state = new_state("catalog-target-change-scope", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "rpc_mode": "single",
        "workload": {"confirmed": True},
    })
    catalog("target_change_scope", state, _target_change_scope_question(state))

    complete_config = {
        "CLOUD_REGION": "test-region",
        "CLOUD_ZONE": "test-zone",
        "MACHINE_TYPE": "test-machine",
        "LEDGER_DEVICE": "vda",
        "DATA_VOL_TYPE": "test-disk",
        "DATA_VOL_SIZE": "1",
        "DATA_VOL_MAX_IOPS": "1",
        "DATA_VOL_MAX_THROUGHPUT": "1",
        "has_accounts_device": False,
        "NETWORK_INTERFACE": "eth0",
        "NETWORK_MAX_BANDWIDTH_GBPS": "1",
    }
    state = new_state("catalog-execution", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "confirmed_config": complete_config,
        "rpc_mode": "single",
        "workload": {"confirmed": True},
        "qps_profile": {"mode": "quick", "confirmed": True},
        "observability": {"mode": "disabled"},
        "advanced_tuning": {"confirmed": True},
    })
    catalog("execution", state, question_for_execution(state, "preflight_smoke_execution"))
    return output


def _runtime_execution_scenarios(language: str) -> list[QuestionScenario]:
    smoke_state = new_state("coverage-real-smoke", language=language, session_purpose="coverage")
    smoke_state.update({
        "target_mode": "real-node",
        "preflight": {"approved": True, "status": "passed", "execution_request_id": "coverage"},
    })
    final_state = new_state("coverage-real-final", language=language, session_purpose="coverage")
    final_state.update({
        "target_mode": "real-node",
        "plan_file": COVERAGE_REAL_NODE_PLAN_PATH,
        "smoke": {
            "purpose": "real_node_isolated_smoke",
            "status": "completed",
            "job_id": "coverage-smoke-job",
        },
        "final_benchmark": {},
    })
    output: list[QuestionScenario] = []
    for scenario_id, state in (
        ("runtime_real_node_smoke", smoke_state),
        ("runtime_real_node_final", final_state),
    ):
        question = question_for_execution(state, "job_monitoring")
        if question:
            state["active_group"] = "job_monitoring"
            state["pending_question"] = deepcopy(question)
            output.append(QuestionScenario(scenario_id, question, state, "runtime_seed"))
    return output


def _valid_literal(validation: Mapping[str, Any]) -> str | None:
    value_type = str(validation.get("value_type") or "")
    if value_type == "scalar_token":
        return "n2-standard-16"
    if value_type in {"positive_number", "positive_integer"}:
        return "1"
    if value_type == "json":
        return '{"key":"value"}'
    if value_type == "enum":
        values = list(validation.get("values") or [])
        return str(values[0]) if values else None
    return None
COVERAGE_REAL_NODE_PLAN_PATH = "/tmp/anychain-agent-real-cli-coverage-approved-plan.json"
