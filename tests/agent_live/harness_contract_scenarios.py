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


@dataclass(frozen=True)
class QuestionScenario:
    scenario_id: str
    question: Mapping[str, Any]
    seed_state: Mapping[str, Any] | None = None
    source: str = "catalog"
    manual_postcondition_path: str = ""
    option_postcondition_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        return self.seed_state is not None

    @property
    def state_fingerprint(self) -> str:
        if self.seed_state is None:
            return content_hash({"catalog_scenario_id": self.scenario_id})
        stable = deepcopy(dict(self.seed_state))
        session = stable.get("session")
        if isinstance(session, dict):
            session.pop("created_at", None)
            session.pop("updated_at", None)
        return content_hash(stable)


@dataclass(frozen=True)
class ManualInputCase:
    input_class: str
    text: str
    expected_admitted: bool


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


def _compiled_action_question(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    user_text: str,
) -> Mapping[str, Any]:
    """Render a control-owning question through the compiled product graph."""

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
    return question


def manual_input_case(question: Mapping[str, Any], input_class: str) -> ManualInputCase | None:
    """Return a concrete local input only when its expected behavior is known."""

    validation = dict(question.get("validation") or {})
    value_type = str(validation.get("value_type") or "")
    if input_class == "valid_literal":
        value = _valid_literal(validation)
        return ManualInputCase(input_class, value, True) if value is not None else None
    if input_class == "trimmed_whitespace_punctuation" and value_type == "scalar_token":
        return ManualInputCase(input_class, "  n2-standard-16,  ", True)
    if input_class == "empty_whitespace":
        return ManualInputCase(input_class, "   ", False)
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
        option_postcondition_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not question:
            return
        state["active_group"] = str(question.get("group") or state.get("active_group") or "opening")
        state["pending_question"] = deepcopy(dict(question))
        scenarios[scenario_id] = QuestionScenario(
            scenario_id,
            deepcopy(dict(question)),
            deepcopy(state),
            source,
            manual_postcondition_path,
            deepcopy(dict(option_postcondition_overrides or {})),
        )

    opening_state = new_state("coverage-opening", language=language, session_purpose="coverage")
    opening = opening_question(opening_state)
    add("opening", opening_state, opening)

    resume_state = new_state("coverage-resume", language=language, session_purpose="coverage")
    resume_state["target_mode"] = "fake-node"
    from agent.harness.domains.orientation import resume_question
    add(
        "resume",
        resume_state,
        resume_question(resume_state),
        option_postcondition_overrides={
            "1": {
                "target_mode": "fake-node",
                "active_group": "chain_identity",
                "pending_question.id": "chain",
            }
        },
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
                "rpc_mode": "single",
                "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            },
        ),
        (
            "workload_mixed",
            "workload_rpc",
            {
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
    """Construct contracts that are known but not yet safe deterministic turns."""

    from agent.harness.domains.chain_identity import _chain_ambiguity_question
    from agent.harness.domains.chain_rpc_questions import _target_change_scope_question
    from agent.harness.domains.environment import config_proposal_review_question
    from agent.harness.domains.orientation import resume_question

    output: dict[str, QuestionScenario] = {}

    def catalog(scenario_id: str, question: Mapping[str, Any] | None) -> None:
        if question:
            output[scenario_id] = QuestionScenario(
                scenario_id,
                deepcopy(dict(question)),
                source="catalog_only",
            )

    state = new_state("catalog-resume-quarantine", language=language, session_purpose="coverage")
    state["checkpoint_recovery"] = {"status": "quarantined", "reason": "partial"}
    catalog("resume_quarantine", resume_question(state))

    for group in ("provider_deployment", "ledger_disk", "accounts_disk", "network"):
        catalog(
            f"inferred_config_{group}",
            config_proposal_review_question(
                group,
                {"config_values": {"CLOUD_REGION": "test-region"}},
                language=language,
            ),
        )

    state = new_state("catalog-ledger-size", language=language, session_purpose="coverage")
    state["confirmed_config"] = {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "test-disk"}
    state["discovery"] = {
        "disks": {"candidates": [{"name": "vda", "size": "100G", "type": "disk"}]}
    }
    catalog("ledger_size_inference", question_for_environment(state, "ledger_disk"))

    state = new_state("catalog-target-mode-change", language=language, session_purpose="coverage")
    state.update({"target_mode": "fake-node", "active_group": "qps_profile"})
    catalog("target_mode_change", _compiled_action_question(
        state,
        {
            "type": "choose_target_mode",
            "target_mode": "real-node",
            "target_mode_explicit": True,
            "source_evidence": "real-node",
            "confidence": "high",
        },
        user_text="switch to real-node",
    ))

    state = new_state("catalog-unknown-chain", language=language, session_purpose="coverage")
    state["target_mode"] = "fake-node"
    state["workflow_mode"] = "rpc_benchmark"
    catalog("unknown_chain_identity", _compiled_action_question(
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
    ))

    state = new_state("catalog-chain-change", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "fake-node",
        "workflow_mode": "rpc_benchmark",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
    })
    catalog("chain_change", _compiled_action_question(
        state,
        {
            "type": "change_chain",
            "chain_text": "ethereum",
            "source_evidence": "ethereum",
            "confidence": "high",
        },
        user_text="change chain to ethereum",
    ))

    state = new_state("catalog-chain-ambiguity", language=language, session_purpose="coverage")
    catalog(
        "chain_ambiguity",
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
        catalog(scenario_id, question_for_chain_rpc(state, "endpoint_process"))

    catalog(
        "chain_change_input",
        {
            "contract_version": 1,
            "id": "chain_change_input",
            "group": "endpoint_process",
            "kind": "chain",
            "prompt": "Enter replacement chain",
            "field": "chain_change_input",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending", "choose_chain", "change_chain"],
            "queue_barrier": True,
            "validation": {},
        },
    )

    state = new_state("catalog-mainnet-review", language=language, session_purpose="coverage")
    state.update({
        "target_mode": "real-node",
        "chain_identity": {"canonical": "bsc", "status": "confirmed"},
        "endpoint_evidence": {"local_rpc_url_ready": True},
        "confirmed_config": {"BLOCKCHAIN_PROCESS_NAMES": "node"},
    })
    catalog("mainnet_review", question_for_chain_rpc(state, "endpoint_process"))

    state = new_state("catalog-chain-api-key", language=language, session_purpose="coverage")
    state["chain_identity"] = {"canonical": "litecoin", "status": "confirmed"}
    catalog("chain_auxiliary_api_key", question_for_chain_rpc(state, "chain_auxiliary_endpoints"))

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
            {"endpoint_ready": True, "method": "eth_blockNumber"},
        ),
        ("custom_needs_weights", "needs_weights", {"validated_methods": validated}),
    )
    for scenario_id, status, extra in custom_statuses:
        state = new_state(f"catalog-{scenario_id}", language=language, session_purpose="coverage")
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed"}
        state["custom_rpc"] = {
            "status": status,
            "catalog": {
                **deepcopy(catalog_envelope),
                "methods": deepcopy(extra.get("validated_methods") or []),
            },
            **deepcopy(extra),
        }
        catalog(scenario_id, question_for_chain_rpc(state, "endpoint_process"))

    new_chain_statuses = (
        ("new_chain_existing_family_needs_endpoint", "existing_family_needs_endpoint", {}),
        ("new_chain_existing_family_needs_method", "existing_family_needs_method", {}),
        (
            "new_chain_existing_family_needs_schema_evidence",
            "existing_family_needs_schema_evidence",
            {"candidate_method": "eth_blockNumber"},
        ),
        (
            "new_chain_existing_family_needs_weights",
            "existing_family_needs_weights",
            {"validated_methods": validated},
        ),
    )
    for scenario_id, status, extra in new_chain_statuses:
        state = new_state(f"catalog-{scenario_id}", language=language, session_purpose="coverage")
        state["chain_identity"] = {
            "canonical": "new-chain",
            "status": status,
            **deepcopy(extra),
        }
        state["custom_rpc"] = {
            "catalog": {
                **deepcopy(catalog_envelope),
                "methods": deepcopy(extra.get("validated_methods") or []),
            }
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint_ready": status != "existing_family_needs_endpoint"
        }
        catalog(scenario_id, question_for_chain_rpc(state, "endpoint_process"))

    state = new_state("catalog-target-change-scope", language=language, session_purpose="coverage")
    catalog("target_change_scope", _target_change_scope_question(state))

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
    catalog("execution", question_for_execution(state, "preflight_smoke_execution"))
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
