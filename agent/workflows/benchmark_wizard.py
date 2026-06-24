"""Deterministic benchmark setup workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from discovery.environment import discover_environment
from knowledge.entry_contract import OPTIONAL_ACCOUNTS_FIELDS
from knowledge.framework_capabilities import load_framework_capabilities
from onboarding.chain_onboarding import generate_onboarding_package, format_onboarding_package
from onboarding.families import SUPPORTED_FAMILIES
from onboarding.request_answers import answer_onboarding_request
from terminal.language import t
from workflows.requirements import ENVIRONMENT_BLOCKERS, REAL_NODE_BLOCKERS, missing_smoke_blockers
from workflows.state import WorkflowState


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_KEY_ALIASES = {
    "BLOCKCHAIN_NODE": "chain",
    "RPC_MODE": "rpc_mode",
    "LOCAL_RPC_URL": "local_rpc_url",
    "MAINNET_RPC_URL": "mainnet_rpc_url",
    "CLOUD_PROVIDER": "cloud_provider",
    "CLOUD_REGION": "cloud_region",
    "CLOUD_ZONE": "cloud_zone",
    "MACHINE_TYPE": "machine_type",
    "BLOCKCHAIN_PROCESS_NAMES": "blockchain_process_names",
    "LEDGER_DEVICE": "ledger_device",
    "ACCOUNTS_DEVICE": "accounts_device",
    "ACCOUNTS_VOL_TYPE": "accounts_vol_type",
    "ACCOUNTS_VOL_SIZE": "accounts_vol_size",
    "ACCOUNTS_VOL_MAX_IOPS": "accounts_vol_max_iops",
    "ACCOUNTS_VOL_MAX_THROUGHPUT": "accounts_vol_max_throughput",
    "DATA_VOL_TYPE": "data_vol_type",
    "DATA_VOL_SIZE": "data_vol_size",
    "DATA_VOL_MAX_IOPS": "data_vol_max_iops",
    "DATA_VOL_MAX_THROUGHPUT": "data_vol_max_throughput",
    "NETWORK_INTERFACE": "network_interface",
    "NETWORK_MAX_BANDWIDTH_GBPS": "network_max_bandwidth_gbps",
    "CHAIN_REST_URL": "chain_rest_url",
    "CHAIN_INDEXER_URL": "chain_indexer_url",
    "CHAIN_SIDECAR_URL": "chain_sidecar_url",
    "CHAIN_EVM_RPC_URL": "chain_evm_rpc_url",
    "CHAIN_JSON_RPC_URL": "chain_json_rpc_url",
    "CHAIN_MIRROR_URL": "chain_mirror_url",
    "RPC_API_KEY": "rpc_api_key",
    "TARGET_ADDRESS": "target_address",
    "TARGET_TX_HASH": "target_tx_hash",
    "TARGET_TXID": "target_txid",
    "TARGET_BLOCK_HASH": "target_block_hash",
    "TARGET_BLOCK": "target_block",
    "TARGET_HEIGHT": "target_height",
    "TARGET_ROUND": "target_round",
    "TARGET_ASSET_ID": "target_asset_id",
    "TARGET_ASSET": "target_asset",
    "TARGET_EPOCH": "target_epoch",
    "TARGET_VP": "target_vp",
    "TARGET_POOL_ID": "target_pool_id",
    "TARGET_TOKEN_ACCOUNT": "target_token_account",
    "TARGET_TOKEN_MINT": "target_token_mint",
    "TARGET_CONTRACT_ADDRESS": "target_contract_address",
    "TARGET_EVM_ADDRESS": "target_evm_address",
    "TARGET_SIGNER_ID": "target_signer_id",
    "TARGET_STORAGE_SLOT": "target_storage_slot",
}
NUMERIC_FIELDS = {
    "data_vol_size",
    "data_vol_max_iops",
    "data_vol_max_throughput",
    "accounts_vol_size",
    "accounts_vol_max_iops",
    "accounts_vol_max_throughput",
    "network_max_bandwidth_gbps",
}
OPTIONAL_ACCOUNTS_BLOCKERS = tuple(field.key for field in OPTIONAL_ACCOUNTS_FIELDS if field.key != "accounts_device")
_DEFAULT_CONFIRMATIONS = {"y", "yes", "确认", "是", "1", "default", "默认", "use default"}
_CUSTOM_CONFIRMATIONS = {"n", "no", "否", "2", "custom", "自定义", "修改", "change"}


@dataclass
class WorkflowResponse:
    handled: bool
    messages: list[str]


class BenchmarkWizard:
    """Conservative state machine for benchmark setup.

    The wizard decides state transitions and gates. ADK/LLM can explain
    decisions, but cannot bypass this state.
    """

    def __init__(self, state: WorkflowState, discovery: dict[str, Any] | None = None) -> None:
        self.state = state
        self.discovery = discovery

    def handle(self, text: str) -> WorkflowResponse:
        lowered = text.lower().strip()
        if "export path" in lowered:
            return self._one("not_manual_export")

        assigned = self._assign_key_value(text)
        if assigned:
            return self._after_value_assignment(assigned)

        special = self._handle_real_node_special_answer(text)
        if special.handled:
            return special

        if self.state.current_question_id == "rpc_workload":
            workload_response = self._handle_rpc_workload_answer(text)
            if workload_response.handled:
                return workload_response

        if self.state.current_question_id == "chain_choice":
            chain_response = self._handle_chain_choice(text)
            if chain_response.handled:
                return chain_response

        if self.state.current_question_id == "unsupported_chain_family":
            family_response = self._handle_unsupported_chain_family(text)
            if family_response.handled:
                return family_response

        if self.state.current_question_id == "unsupported_chain_methods":
            methods_response = self._handle_unsupported_chain_methods(text)
            if methods_response.handled:
                return methods_response

        if self.state.current_question_id == "target_type":
            target_response = self._handle_target_type_answer(text)
            if target_response.handled:
                return target_response

        if self.state.current_question_id == "advanced_config_review":
            advanced_response = self._handle_advanced_config_review(text)
            if advanced_response.handled:
                return advanced_response

        if self.state.current_question_id in set(REAL_NODE_BLOCKERS) | set(ENVIRONMENT_BLOCKERS) | set(OPTIONAL_ACCOUNTS_BLOCKERS):
            validation_error = self._assign_current_benchmark_value(text)
            if validation_error:
                return WorkflowResponse(True, [validation_error])
            return self._ask_next_benchmark_value()

        chain = extract_chain(text)
        if chain:
            self.state.intent = "benchmark"
            self.state.confirmed_values["chain"] = chain
            if "fake-node" in lowered or "fake node" in lowered:
                return self._select_fake_node()
            if "real-node" in lowered or "real node" in lowered or "真实节点" in text:
                return self._select_real_node()
            self.state.stage = "confirm_chain"
            self.state.current_question_id = "target_type"
            self._refresh_blockers()
            return WorkflowResponse(True, [
                t(self.state.language, "benchmark_chain", chain=chain),
                t(self.state.language, "select_target"),
            ])

        if _looks_like_benchmark_request(text):
            self.state.intent = "benchmark"
            self.state.stage = "select_chain"
            self.state.current_question_id = "chain_choice"
            return WorkflowResponse(True, [
                t(self.state.language, "ask_chain", chains=_format_chain_sample()),
            ])

        target = _parse_target_type(text, allow_numeric=False)
        if target == "fake-node":
            if not self.state.confirmed_values.get("chain"):
                return self._ask_chain_for_target("fake-node")
            return self._select_fake_node()
        if target == "real-node":
            if not self.state.confirmed_values.get("chain"):
                return self._ask_chain_for_target("real-node")
            return self._select_real_node()

        rpc_mode = _parse_rpc_mode(text)
        if rpc_mode and self.state.stage == "select_rpc_mode":
            self.state.confirmed_values["rpc_mode"] = rpc_mode
            self.state.stage = "confirm_rpc_workload"
            self.state.current_question_id = "rpc_workload"
            return self._confirm_workload(rpc_mode)

        if lowered in {"y", "yes", "确认", "是"}:
            return self._confirm_current()

        return WorkflowResponse(False, [])

    def can_run_smoke(self) -> tuple[bool, list[str]]:
        missing = missing_smoke_blockers(self.state.confirmed_values)
        return not missing, missing

    def _select_fake_node(self) -> WorkflowResponse:
        self.state.intent = "benchmark"
        if not self.state.confirmed_values.get("chain"):
            return self._ask_chain_for_target("fake-node")
        self.state.confirmed_values["use_fake_node"] = True
        self._prepare_environment_defaults()
        self._refresh_blockers()
        messages = [t(self.state.language, "fake_node_selected")]
        next_response = self._ask_next_benchmark_value()
        messages.extend(next_response.messages)
        return WorkflowResponse(True, messages)

    def _select_real_node(self) -> WorkflowResponse:
        self.state.intent = "benchmark"
        if not self.state.confirmed_values.get("chain"):
            return self._ask_chain_for_target("real-node")
        self.state.confirmed_values["use_fake_node"] = False
        self._prepare_environment_defaults()
        self.state.stage = "real_node_endpoint"
        self.state.current_question_id = "local_rpc_url"
        self._refresh_blockers()
        messages = [t(self.state.language, "real_node_selected")]
        next_response = self._ask_next_benchmark_value()
        messages.extend(next_response.messages)
        return WorkflowResponse(True, messages)

    def _confirm_workload(self, mode: str) -> WorkflowResponse:
        chain_defaults = self._chain_workload_defaults()
        if mode == "mixed":
            mixed_weights = chain_defaults.get("mixed_weights", {})
            if mixed_weights:
                self.state.defaulted_values["mixed_weights"] = mixed_weights
                self.state.current_question_id = "mixed_weights_confirm"
                self.state.stage = "confirm_mixed_weights"
                return WorkflowResponse(True, [
                    t(self.state.language, "ask_detected_mixed_weights", weights=_format_weights(mixed_weights))
                ])
            self.state.current_question_id = "rpc_workload"
            self.state.stage = "collect_mixed_weights"
            self._refresh_blockers()
            return self._one("confirm_mixed_weights")
        single_method = chain_defaults.get("single_method", "")
        if single_method:
            self.state.defaulted_values["single_method"] = single_method
            self.state.current_question_id = "single_method_confirm"
            self.state.stage = "confirm_single_method"
            return WorkflowResponse(True, [
                t(self.state.language, "ask_detected_single_method", method=single_method)
            ])
        self.state.current_question_id = "rpc_workload"
        self.state.stage = "collect_single_method"
        self._refresh_blockers()
        return self._one("confirm_single_method")

    def _confirm_current(self) -> WorkflowResponse:
        if self.state.stage == "confirm_chain":
            self.state.stage = "select_target_type"
            self.state.current_question_id = "target_type"
            return self._one("select_target")
        if self.state.current_question_id == "rpc_workload":
            self.state.confirmed_values["rpc_workload_confirmed"] = True
            self.state.stage = "confirm_rpc_param_samples"
            self.state.current_question_id = "rpc_param_samples"
            self._refresh_blockers()
            return self._one("confirm_param_samples")
        if self.state.current_question_id == "rpc_param_samples":
            self.state.confirmed_values["rpc_param_samples_confirmed"] = True
            self.state.stage = "ready_for_smoke"
            self.state.current_question_id = "smoke_confirmation"
            self._refresh_blockers()
            return WorkflowResponse(True, [
                t(self.state.language, "gate_ready"),
                t(self.state.language, "prepare_smoke_offer"),
            ])
        ok, missing = self.can_run_smoke()
        if not ok:
            self.state.missing_blockers = missing
            return WorkflowResponse(True, [t(self.state.language, "gate_blocked", missing=", ".join(missing))])
        return WorkflowResponse(True, [
            t(self.state.language, "gate_ready"),
            t(self.state.language, "prepare_smoke_offer"),
        ])

    def _chain_workload_defaults(self) -> dict[str, Any]:
        chain = str(self.state.confirmed_values.get("chain", ""))
        if not chain:
            return {}
        capabilities = load_framework_capabilities()
        chain_data = next(
            (row for row in capabilities.get("chains", []) if row.get("chain") == chain),
            {},
        )
        weighted = {}
        for item in chain_data.get("mixed_weighted", []):
            method = str(item.get("method", "")).strip()
            weight = item.get("weight")
            if method and isinstance(weight, int):
                weighted[method] = weight
        return {
            "single_method": chain_data.get("single", ""),
            "mixed_weights": weighted,
        }

    def _ask_chain_for_target(self, target: str) -> WorkflowResponse:
        self.state.intent = "benchmark"
        self.state.stage = "select_chain"
        self.state.current_question_id = "chain_choice"
        self.state.defaulted_values["pending_target"] = target
        return WorkflowResponse(True, [
            t(self.state.language, "ask_chain_for_target", target=target, chains=_format_chain_sample()),
        ])

    def _handle_chain_choice(self, text: str) -> WorkflowResponse:
        chain = extract_chain(text) or _normalize_chain_choice(text)
        chains = set(known_chains())
        if chain not in chains:
            self.state.confirmed_values["onboarding_chain"] = chain or text.strip()
            self.state.stage = "classify_unsupported_chain"
            self.state.current_question_id = "unsupported_chain_family"
            return WorkflowResponse(True, [
                t(self.state.language, "unsupported_chain_intro", chain=chain or text.strip(), chains=_format_chain_sample()),
                t(self.state.language, "ask_unsupported_chain_family", families=_format_family_choices()),
            ])
        self.state.confirmed_values["chain"] = chain
        pending_target = str(self.state.defaulted_values.pop("pending_target", ""))
        if pending_target == "fake-node":
            selected = self._select_fake_node()
            return WorkflowResponse(True, [
                t(self.state.language, "benchmark_chain", chain=chain),
                *selected.messages,
            ])
        if pending_target == "real-node":
            selected = self._select_real_node()
            return WorkflowResponse(True, [
                t(self.state.language, "benchmark_chain", chain=chain),
                *selected.messages,
            ])
        self.state.stage = "confirm_chain"
        self.state.current_question_id = "target_type"
        self._refresh_blockers()
        return WorkflowResponse(True, [
            t(self.state.language, "benchmark_chain", chain=chain),
            t(self.state.language, "select_target"),
        ])

    def _handle_target_type_answer(self, text: str) -> WorkflowResponse:
        target = _parse_target_type(text, allow_numeric=True)
        if target == "fake-node":
            return self._select_fake_node()
        if target == "real-node":
            return self._select_real_node()
        lowered = text.lower().strip()
        if lowered in {"y", "yes", "确认", "是"}:
            return self._one("select_target")
        return WorkflowResponse(True, [t(self.state.language, "invalid_target_type")])

    def _handle_unsupported_chain_family(self, text: str) -> WorkflowResponse:
        family = _parse_family_choice(text)
        chain = str(self.state.confirmed_values.get("onboarding_chain", "")).strip()
        if family == "new_family":
            self.state.confirmed_values["onboarding_family"] = "new_family"
            self.state.stage = "onboarding_handoff_ready"
            self.state.current_question_id = ""
            return WorkflowResponse(True, [
                t(self.state.language, "unsupported_chain_new_family"),
                answer_onboarding_request(f"Generate a plan to add a new protocol family for {chain or 'new chain'}"),
            ])
        if not family:
            return WorkflowResponse(True, [
                t(self.state.language, "invalid_family_choice"),
                t(self.state.language, "ask_unsupported_chain_family", families=_format_family_choices()),
            ])
        self.state.confirmed_values["onboarding_family"] = family
        self.state.stage = "collect_unsupported_chain_methods"
        self.state.current_question_id = "unsupported_chain_methods"
        return WorkflowResponse(True, [
            t(self.state.language, "unsupported_chain_family_recorded", chain=chain, family=family),
            t(self.state.language, "ask_unsupported_chain_methods"),
        ])

    def _handle_unsupported_chain_methods(self, text: str) -> WorkflowResponse:
        chain = str(self.state.confirmed_values.get("onboarding_chain", "")).strip() or "newchain"
        family = str(self.state.confirmed_values.get("onboarding_family", "")).strip() or "<choose-family>"
        methods = _parse_method_list(text)
        if not methods:
            return WorkflowResponse(True, [t(self.state.language, "ask_unsupported_chain_methods")])
        package = generate_onboarding_package(chain, methods=methods, adapter_family=family)
        self.state.stage = "onboarding_handoff_ready"
        self.state.current_question_id = ""
        return WorkflowResponse(True, [
            t(self.state.language, "unsupported_chain_onboarding_ready", chain=chain),
            format_onboarding_package(package),
        ])

    def _handle_advanced_config_review(self, text: str) -> WorkflowResponse:
        lowered = text.lower().strip()
        self.state.confirmed_values["advanced_config_reviewed"] = True
        if lowered in {"y", "yes", "确认", "是", "需要", "看一下", "show", "review"}:
            message = t(self.state.language, "advanced_config_summary")
        else:
            message = t(self.state.language, "advanced_config_skipped")
        self.state.stage = "select_rpc_mode"
        self.state.current_question_id = "rpc_mode"
        return WorkflowResponse(True, [
            message,
            t(self.state.language, "benchmark_required_done"),
            t(self.state.language, "ask_rpc_mode"),
        ])

    def _handle_rpc_workload_answer(self, text: str) -> WorkflowResponse:
        value = text.strip().strip("\"'")
        if not value:
            return WorkflowResponse(True, [t(self.state.language, "invalid_empty")])
        mode = self.state.confirmed_values.get("rpc_mode")
        if mode == "mixed":
            weights = _parse_mixed_weights(value)
            if not weights:
                return WorkflowResponse(True, [t(self.state.language, "invalid_mixed_weights")])
            total = sum(weights.values())
            if total != 100:
                return WorkflowResponse(True, [t(self.state.language, "invalid_mixed_weight_total", total=total)])
            self.state.confirmed_values["mixed_weights"] = weights
            self.state.confirmed_values["rpc_methods"] = list(weights.keys())
            self.state.confirmed_values["mixed_weights_confirmed"] = True
            self.state.confirmed_values["rpc_workload_confirmed"] = True
            self.state.stage = "confirm_rpc_param_samples"
            self.state.current_question_id = "rpc_param_samples"
            self._refresh_blockers()
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name="mixed_weights"),
                t(self.state.language, "confirm_param_samples"),
            ])
        self.state.confirmed_values["single_method"] = value
        self.state.confirmed_values["rpc_methods"] = [value]
        self.state.confirmed_values["rpc_workload_confirmed"] = True
        self.state.stage = "confirm_rpc_param_samples"
        self.state.current_question_id = "rpc_param_samples"
        self._refresh_blockers()
        return WorkflowResponse(True, [
            t(self.state.language, "recorded_value", name="single_method"),
            t(self.state.language, "confirm_param_samples"),
        ])

    def _refresh_blockers(self) -> None:
        _, missing = self.can_run_smoke()
        self.state.missing_blockers = missing

    def _one(self, key: str) -> WorkflowResponse:
        return WorkflowResponse(True, [t(self.state.language, key)])

    def _assign_key_value(self, text: str) -> str:
        if "=" not in text:
            return ""
        raw_key, raw_value = text.split("=", 1)
        key = raw_key.strip().upper()
        value = raw_value.strip().strip("\"'")
        target = ENV_KEY_ALIASES.get(key)
        if not target or not value:
            return ""
        if self._validate_value(target, value):
            return ""
        if target == "blockchain_process_names":
            self.state.confirmed_values[target] = [item for item in value.replace(",", " ").split() if item]
        else:
            self.state.confirmed_values[target] = value
        return target

    def _assign_current_benchmark_value(self, text: str) -> str:
        key = self.state.current_question_id
        value = text.strip().strip("\"'")
        validation_error = self._validate_value(key, value)
        if validation_error:
            return validation_error
        if key == "blockchain_process_names":
            self.state.confirmed_values[key] = [item for item in value.replace(",", " ").split() if item]
        else:
            self.state.confirmed_values[key] = value
        return ""

    def _after_value_assignment(self, assigned: str) -> WorkflowResponse:
        self._refresh_blockers()
        messages = [t(self.state.language, "recorded_value", name=assigned)]
        if self.state.confirmed_values.get("use_fake_node") in {True, False}:
            next_response = self._ask_next_benchmark_value()
            messages.extend(next_response.messages)
        return WorkflowResponse(True, messages)

    def _ask_next_benchmark_value(self) -> WorkflowResponse:
        self._refresh_blockers()
        if self.state.confirmed_values.get("use_fake_node") is False:
            for key in ("local_rpc_url", "mainnet_rpc_url_reviewed"):
                if key == "mainnet_rpc_url_reviewed" and _is_missing(self.state.confirmed_values.get(key)):
                    self.state.stage = "confirm_mainnet_rpc_url"
                    self.state.current_question_id = "mainnet_rpc_url_reviewed"
                    return WorkflowResponse(True, [t(self.state.language, "ask_mainnet_rpc_url")])
                if _is_missing(self.state.confirmed_values.get(key)):
                    self.state.current_question_id = key
                    self.state.stage = f"collect_{key}"
                    return WorkflowResponse(True, [t(self.state.language, "ask_required_value", name=_display_key(key))])
        for key in ENVIRONMENT_BLOCKERS:
            if _is_missing(self.state.confirmed_values.get(key)):
                if key == "ledger_device":
                    return self._ask_ledger_device()
                if key in {"cloud_region", "cloud_zone", "machine_type"} and self.state.defaulted_values.get(key):
                    self.state.current_question_id = f"{key}_confirm"
                    self.state.stage = f"confirm_{key}"
                    return WorkflowResponse(True, [
                        t(self.state.language, "ask_detected_value_confirm", name=_display_key(key), value=self.state.defaulted_values[key])
                    ])
                if key == "network_interface" and self.state.defaulted_values.get("network_interface"):
                    self.state.current_question_id = "network_interface_confirm"
                    self.state.stage = "confirm_network_interface"
                    return WorkflowResponse(True, [
                        t(self.state.language, "ask_network_interface_confirm", value=self.state.defaulted_values["network_interface"])
                    ])
                self.state.current_question_id = key
                self.state.stage = f"collect_{key}"
                return WorkflowResponse(True, [t(self.state.language, "ask_required_value", name=_display_key(key))])
        accounts_response = self._ask_accounts_device_if_needed()
        if accounts_response.handled:
            return accounts_response
        if self.state.confirmed_values.get("has_accounts_device") is True:
            for key in OPTIONAL_ACCOUNTS_BLOCKERS:
                if _is_missing(self.state.confirmed_values.get(key)):
                    self.state.current_question_id = key
                    self.state.stage = f"collect_{key}"
                    return WorkflowResponse(True, [t(self.state.language, "ask_required_value", name=_display_key(key))])
        if not self.state.confirmed_values.get("advanced_config_reviewed"):
            self.state.stage = "confirm_advanced_config"
            self.state.current_question_id = "advanced_config_review"
            return WorkflowResponse(True, [t(self.state.language, "ask_advanced_config_review")])
        if (
            self.state.confirmed_values.get("rpc_mode")
            and self.state.confirmed_values.get("rpc_workload_confirmed")
            and self.state.confirmed_values.get("rpc_param_samples_confirmed")
        ):
            self.state.stage = "ready_for_smoke"
            self.state.current_question_id = "smoke_confirmation"
            self._refresh_blockers()
            return WorkflowResponse(True, [
                t(self.state.language, "benchmark_required_done"),
                t(self.state.language, "gate_ready"),
                t(self.state.language, "prepare_smoke_offer"),
            ])
        self.state.stage = "select_rpc_mode"
        self.state.current_question_id = "rpc_mode"
        return WorkflowResponse(True, [
            t(self.state.language, "benchmark_required_done"),
            t(self.state.language, "ask_rpc_mode"),
        ])

    def _prepare_environment_defaults(self) -> None:
        discovery = self._discovery()
        cloud = discovery.get("cloud", {})
        deployment = discovery.get("deployment", {})
        network = discovery.get("network", {})
        self.state.defaulted_values["discovery"] = discovery
        self.state.defaulted_values["disk_candidates"] = _disk_candidates(discovery.get("disks", {}))
        self.state.defaulted_values["network_default"] = network.get("default_interface", "")
        if cloud.get("provider") and not self.state.confirmed_values.get("cloud_provider"):
            self.state.confirmed_values["cloud_provider"] = cloud.get("provider")
        platform_name = cloud.get("platform") or deployment.get("type")
        if platform_name and not self.state.confirmed_values.get("deployment_type"):
            self.state.confirmed_values["deployment_type"] = platform_name
        for key in ("region", "zone", "machine_type"):
            value = cloud.get(key)
            target = {
                "region": "cloud_region",
                "zone": "cloud_zone",
                "machine_type": "machine_type",
            }[key]
            if value and not self.state.confirmed_values.get(target):
                self.state.defaulted_values[target] = value
        if network.get("default_interface") and not self.state.confirmed_values.get("network_interface"):
            self.state.defaulted_values["network_interface"] = network.get("default_interface")

    def _discovery(self) -> dict[str, Any]:
        if self.discovery is None:
            self.discovery = discover_environment()
        return self.discovery

    def _ask_ledger_device(self) -> WorkflowResponse:
        candidates = self.state.defaulted_values.get("disk_candidates") or []
        if candidates:
            self.state.stage = "select_ledger_device"
            self.state.current_question_id = "ledger_device_choice"
            return WorkflowResponse(True, [
                t(self.state.language, "disk_candidates", candidates=_format_candidates(candidates)),
                t(self.state.language, "ask_ledger_device_choice"),
            ])
        self.state.current_question_id = "ledger_device"
        self.state.stage = "collect_ledger_device"
        return WorkflowResponse(True, [t(self.state.language, "ask_required_value", name="LEDGER_DEVICE")])

    def _ask_accounts_device_if_needed(self) -> WorkflowResponse:
        if self.state.confirmed_values.get("has_accounts_device") is not None:
            return WorkflowResponse(False, [])
        candidates = self.state.defaulted_values.get("disk_candidates") or []
        self.state.stage = "confirm_accounts_device"
        self.state.current_question_id = "has_accounts_device"
        messages = []
        if candidates:
            messages.append(t(self.state.language, "disk_candidates", candidates=_format_candidates(candidates)))
        messages.append(t(self.state.language, "ask_accounts_device_presence"))
        return WorkflowResponse(True, messages)

    def _handle_real_node_special_answer(self, text: str) -> WorkflowResponse:
        key = self.state.current_question_id
        if key == "ledger_device_choice":
            value = self._resolve_candidate_or_manual(text)
            if not value:
                return WorkflowResponse(True, [t(self.state.language, "invalid_choice")])
            self.state.confirmed_values["ledger_device"] = value
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name="LEDGER_DEVICE"),
                *self._ask_next_benchmark_value().messages,
            ])
        if key == "mainnet_rpc_url_reviewed":
            lowered = text.lower().strip()
            if lowered in {"skip", "default", "n", "no", "否", "默认", "跳过", "不配置"}:
                self.state.confirmed_values["mainnet_rpc_url_reviewed"] = True
                self.state.skipped_values["mainnet_rpc_url"] = "chain_template_default"
            else:
                validation_error = self._validate_value("local_rpc_url", text.strip().strip("\"'"))
                if validation_error:
                    return WorkflowResponse(True, [t(self.state.language, "invalid_url", name="MAINNET_RPC_URL")])
                self.state.confirmed_values["mainnet_rpc_url"] = text.strip().strip("\"'")
                self.state.confirmed_values["mainnet_rpc_url_reviewed"] = True
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name="MAINNET_RPC_URL"),
                *self._ask_next_benchmark_value().messages,
            ])
        if key == "network_interface_confirm":
            lowered = text.lower().strip()
            default_iface = str(self.state.defaulted_values.get("network_interface", ""))
            if lowered in {"y", "yes", "确认", "是"} and default_iface:
                self.state.confirmed_values["network_interface"] = default_iface
            else:
                value = text.strip().strip("\"'")
                validation_error = self._validate_value("network_interface", value)
                if validation_error:
                    return WorkflowResponse(True, [validation_error])
                self.state.confirmed_values["network_interface"] = value
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name="NETWORK_INTERFACE"),
                *self._ask_next_benchmark_value().messages,
            ])
        if key in {"cloud_region_confirm", "cloud_zone_confirm", "machine_type_confirm"}:
            target = key.removesuffix("_confirm")
            lowered = text.lower().strip()
            default_value = str(self.state.defaulted_values.get(target, ""))
            if lowered in {"y", "yes", "确认", "是"} and default_value:
                self.state.confirmed_values[target] = default_value
            else:
                value = text.strip().strip("\"'")
                validation_error = self._validate_value(target, value)
                if validation_error:
                    return WorkflowResponse(True, [validation_error])
                self.state.confirmed_values[target] = value
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name=_display_key(target)),
                *self._ask_next_benchmark_value().messages,
            ])
        if key == "has_accounts_device":
            lowered = text.lower().strip()
            if lowered in {"n", "no", "否", "没有", "不需要"}:
                self.state.confirmed_values["has_accounts_device"] = False
                self.state.skipped_values["accounts_device"] = "not_used"
                return WorkflowResponse(True, [
                    t(self.state.language, "accounts_device_skipped"),
                    *self._ask_next_benchmark_value().messages,
                ])
            if lowered in {"y", "yes", "确认", "是", "有", "需要"}:
                self.state.confirmed_values["has_accounts_device"] = True
                self.state.stage = "select_accounts_device"
                self.state.current_question_id = "accounts_device_choice"
                candidates = self.state.defaulted_values.get("disk_candidates") or []
                messages = []
                if candidates:
                    messages.append(t(self.state.language, "disk_candidates", candidates=_format_candidates(candidates)))
                messages.append(t(self.state.language, "ask_accounts_device_choice"))
                return WorkflowResponse(True, messages)
            return WorkflowResponse(True, [t(self.state.language, "ask_accounts_device_presence")])
        if key == "single_method_confirm":
            lowered = text.lower().strip()
            default_method = str(self.state.defaulted_values.get("single_method", ""))
            if lowered in _DEFAULT_CONFIRMATIONS and default_method:
                self.state.confirmed_values["single_method"] = default_method
                self.state.confirmed_values["rpc_methods"] = [default_method]
                self.state.confirmed_values["rpc_workload_confirmed"] = True
                self.state.stage = "confirm_rpc_param_samples"
                self.state.current_question_id = "rpc_param_samples"
                self._refresh_blockers()
                return WorkflowResponse(True, [
                    t(self.state.language, "recorded_value", name="single_method"),
                    t(self.state.language, "confirm_param_samples"),
                ])
            if lowered and lowered not in _CUSTOM_CONFIRMATIONS:
                return WorkflowResponse(True, [t(self.state.language, "ask_single_default_or_custom")])
            self.state.stage = "collect_single_method"
            self.state.current_question_id = "rpc_workload"
            return WorkflowResponse(True, [t(self.state.language, "ask_custom_single_method")])
        if key == "mixed_weights_confirm":
            lowered = text.lower().strip()
            default_weights = self.state.defaulted_values.get("mixed_weights") or {}
            if lowered in _DEFAULT_CONFIRMATIONS and default_weights:
                self.state.confirmed_values["mixed_weights"] = dict(default_weights)
                self.state.confirmed_values["rpc_methods"] = list(default_weights.keys())
                self.state.confirmed_values["mixed_weights_confirmed"] = True
                self.state.confirmed_values["rpc_workload_confirmed"] = True
                self.state.stage = "confirm_rpc_param_samples"
                self.state.current_question_id = "rpc_param_samples"
                self._refresh_blockers()
                return WorkflowResponse(True, [
                    t(self.state.language, "recorded_value", name="mixed_weights"),
                    t(self.state.language, "confirm_param_samples"),
                ])
            if lowered and lowered not in _CUSTOM_CONFIRMATIONS:
                return WorkflowResponse(True, [t(self.state.language, "ask_mixed_default_or_custom")])
            self.state.stage = "collect_mixed_weights"
            self.state.current_question_id = "rpc_workload"
            return WorkflowResponse(True, [t(self.state.language, "ask_custom_mixed_weights")])
        if key == "accounts_device_choice":
            value = self._resolve_candidate_or_manual(text)
            if not value:
                return WorkflowResponse(True, [t(self.state.language, "invalid_choice")])
            self.state.confirmed_values["accounts_device"] = value
            return WorkflowResponse(True, [
                t(self.state.language, "recorded_value", name="ACCOUNTS_DEVICE"),
                *self._ask_next_benchmark_value().messages,
            ])
        return WorkflowResponse(False, [])

    def _resolve_candidate_or_manual(self, text: str) -> str:
        raw = text.strip().strip("\"'")
        candidates = self.state.defaulted_values.get("disk_candidates") or []
        if raw.isdigit() and candidates:
            index = int(raw) - 1
            if 0 <= index < len(candidates):
                return str(candidates[index].get("name", ""))
            return ""
        return raw

    def _validate_value(self, key: str, value: str) -> str:
        if not value:
            return t(self.state.language, "invalid_empty")
        if key == "local_rpc_url":
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
                return t(self.state.language, "invalid_url", name="LOCAL_RPC_URL")
        if key in NUMERIC_FIELDS:
            try:
                if float(value) <= 0:
                    raise ValueError
            except ValueError:
                return t(self.state.language, "invalid_numeric", name=_display_key(key))
        return ""


def extract_chain(text: str) -> str:
    lowered = text.lower()
    for chain in known_chains():
        if chain in lowered:
            return chain
    return ""


def _normalize_chain_choice(text: str) -> str:
    value = text.lower().strip().strip("\"'")
    if not value:
        return ""
    tokens = [
        token.strip(".,:;()[]{}")
        for token in value.replace("/", " ").split()
        if token.strip(".,:;()[]{}")
    ]
    stop = {
        "i", "want", "to", "benchmark", "test", "chain", "node", "with", "use",
        "我要", "压测", "测试", "节点", "区块链",
    }
    candidates = [token for token in tokens if token not in stop]
    return (candidates[-1] if candidates else value).replace(" ", "-")


def known_chains() -> tuple[str, ...]:
    chain_dir = REPO_ROOT / "config" / "chains"
    chains = sorted(path.stem for path in chain_dir.glob("*.json"))
    return tuple(chains)


def _looks_like_benchmark_request(text: str) -> bool:
    lowered = text.lower()
    keywords = (
        "benchmark", "load test", "stress test", "qps", "vegeta",
        "压测", "测试节点", "性能测试", "极限测试", "跑测试", "测试一个节点",
    )
    return any(keyword in lowered or keyword in text for keyword in keywords)


def _parse_target_type(text: str, allow_numeric: bool = False) -> str:
    normalized = text.lower().strip()
    fake_tokens = {
        "fake", "fake-node", "fake node", "mock", "mock-node", "mock node",
        "closed-loop", "closed loop",
        "闭环", "闭环测试", "模拟", "模拟节点",
    }
    real_tokens = {
        "real", "real-node", "real node", "node", "live", "production",
        "真实", "真实节点", "真节点", "实际节点", "生产节点",
    }
    if allow_numeric:
        if normalized == "1":
            return "fake-node"
        if normalized == "2":
            return "real-node"
    if normalized in fake_tokens or "fake-node" in normalized or "fake node" in normalized:
        return "fake-node"
    if normalized in real_tokens or "real-node" in normalized or "real node" in normalized or "真实节点" in text:
        return "real-node"
    return ""


def _parse_rpc_mode(text: str) -> str:
    normalized = text.lower().strip()
    if normalized in {"1", "single", "单一", "单方法", "单个", "单 rpc", "单rpc"}:
        return "single"
    if normalized in {"2", "mixed", "mix", "混合", "混合模式", "多方法", "多个"}:
        return "mixed"
    return ""


def _format_chain_sample(limit: int = 12) -> str:
    chains = known_chains()
    sample = ", ".join(chains[:limit])
    if len(chains) > limit:
        return f"{sample}, ... ({len(chains)} total)"
    return sample


def _format_family_choices() -> str:
    rows = [f"[{index}] {family}" for index, family in enumerate(SUPPORTED_FAMILIES, start=1)]
    rows.append(f"[{len(SUPPORTED_FAMILIES) + 1}] new/unknown protocol family")
    return "\n".join(rows)


def _parse_family_choice(text: str) -> str:
    value = text.lower().strip().replace("-", "_")
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(SUPPORTED_FAMILIES):
            return SUPPORTED_FAMILIES[index - 1]
        if index == len(SUPPORTED_FAMILIES) + 1:
            return "new_family"
    if value in {"new", "unknown", "new_family", "new protocol", "unknown protocol", "新的协议", "新协议", "不知道"}:
        return "new_family"
    for family in SUPPORTED_FAMILIES:
        if value == family or family in value:
            return family
    return ""


def _parse_method_list(text: str) -> list[str]:
    value = text.strip().strip("\"'")
    if not value:
        return []
    raw = value.replace("，", ",").replace("\n", ",").replace(";", ",")
    methods = []
    for item in raw.split(","):
        method = item.strip()
        if not method:
            continue
        if "=" in method:
            method = method.split("=", 1)[0].strip()
        if method and method not in methods:
            methods.append(method)
    return methods[:20]


def _display_key(key: str) -> str:
    for env_key, internal_key in ENV_KEY_ALIASES.items():
        if internal_key == key:
            return env_key
    return key


def _is_missing(value: object) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


def _disk_candidates(disks: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for item in disks.get("candidates", []):
        name = item.get("name")
        if not name or item.get("mountpoint") in {"/boot", "/boot/efi"}:
            continue
        candidates.append({
            "name": name,
            "type": item.get("type", ""),
            "size": item.get("size", ""),
            "mountpoint": item.get("mountpoint", ""),
            "fstype": item.get("fstype", ""),
            "label": item.get("label", ""),
        })
    return candidates


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(candidates, start=1):
        lines.append(
            "{index}. {name} type={type} size={size} mount={mountpoint} fstype={fstype} label={label}".format(
                index=index,
                name=item.get("name", ""),
                type=item.get("type", ""),
                size=item.get("size", ""),
                mountpoint=item.get("mountpoint", "") or "<none>",
                fstype=item.get("fstype", "") or "<none>",
                label=item.get("label", "") or "<none>",
            )
        )
    return "\n".join(lines)


def _format_weights(weights: dict[str, int]) -> str:
    return ",".join(f"{method}={weight}" for method, weight in weights.items())


def _parse_mixed_weights(text: str) -> dict[str, int]:
    normalized = text.replace("，", ",").replace("；", ",").replace(";", ",")
    weights: dict[str, int] = {}
    for raw_item in normalized.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            method, weight = item.split("=", 1)
        elif ":" in item:
            method, weight = item.split(":", 1)
        else:
            parts = item.replace("%", "").split()
            if len(parts) != 2:
                return {}
            method, weight = parts
        method = method.strip()
        weight_text = weight.strip().rstrip("%")
        if not method or not weight_text.isdigit():
            return {}
        weights[method] = int(weight_text)
    return weights
