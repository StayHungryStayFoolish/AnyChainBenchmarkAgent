"""Deterministic benchmark setup workflow."""

from __future__ import annotations

from dataclasses import dataclass

from terminal.language import t
from workflows.requirements import REAL_NODE_BLOCKERS, missing_smoke_blockers
from workflows.state import WorkflowState


KNOWN_CHAINS = ("solana", "ethereum", "bitcoin", "polygon", "bsc", "base", "sui", "aptos")
ENV_KEY_ALIASES = {
    "LOCAL_RPC_URL": "local_rpc_url",
    "MAINNET_RPC_URL": "mainnet_rpc_url",
    "BLOCKCHAIN_PROCESS_NAMES": "blockchain_process_names",
    "LEDGER_DEVICE": "ledger_device",
    "ACCOUNTS_DEVICE": "accounts_device",
    "DATA_VOL_TYPE": "data_vol_type",
    "DATA_VOL_SIZE": "data_vol_size",
    "DATA_VOL_MAX_IOPS": "data_vol_max_iops",
    "DATA_VOL_MAX_THROUGHPUT": "data_vol_max_throughput",
    "NETWORK_INTERFACE": "network_interface",
    "NETWORK_MAX_BANDWIDTH_GBPS": "network_max_bandwidth_gbps",
}


@dataclass
class WorkflowResponse:
    handled: bool
    messages: list[str]


class BenchmarkWizard:
    """Conservative state machine for benchmark setup.

    The wizard decides state transitions and gates. ADK/LLM can explain
    decisions, but cannot bypass this state.
    """

    def __init__(self, state: WorkflowState) -> None:
        self.state = state

    def handle(self, text: str) -> WorkflowResponse:
        lowered = text.lower().strip()
        if "export path" in lowered:
            return self._one("not_manual_export")

        assigned = self._assign_key_value(text)
        if assigned:
            return self._after_value_assignment(assigned)

        if self.state.confirmed_values.get("use_fake_node") is False and self.state.current_question_id in REAL_NODE_BLOCKERS:
            self._assign_current_real_node_value(text)
            return self._ask_next_real_node_value()

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

        if "fake-node" in lowered or "fake node" in lowered:
            return self._select_fake_node()
        if "real-node" in lowered or "real node" in lowered or "真实节点" in text:
            return self._select_real_node()

        if lowered in {"single", "mixed"} and self.state.stage == "select_rpc_mode":
            self.state.confirmed_values["rpc_mode"] = lowered
            self.state.stage = "confirm_rpc_workload"
            self.state.current_question_id = "rpc_workload"
            return self._confirm_workload(lowered)

        if lowered in {"y", "yes", "确认", "是"}:
            return self._confirm_current()

        return WorkflowResponse(False, [])

    def can_run_smoke(self) -> tuple[bool, list[str]]:
        missing = missing_smoke_blockers(self.state.confirmed_values)
        return not missing, missing

    def _select_fake_node(self) -> WorkflowResponse:
        self.state.intent = "benchmark"
        self.state.confirmed_values["use_fake_node"] = True
        self.state.stage = "select_rpc_mode"
        self.state.current_question_id = "rpc_mode"
        self._refresh_blockers()
        return WorkflowResponse(True, [
            t(self.state.language, "fake_node_selected"),
            t(self.state.language, "ask_rpc_mode"),
        ])

    def _select_real_node(self) -> WorkflowResponse:
        self.state.intent = "benchmark"
        self.state.confirmed_values["use_fake_node"] = False
        self.state.stage = "real_node_endpoint"
        self.state.current_question_id = "local_rpc_url"
        self._refresh_blockers()
        return WorkflowResponse(True, [
            t(self.state.language, "real_node_selected"),
            t(self.state.language, "ask_required_value", name="LOCAL_RPC_URL"),
        ])

    def _confirm_workload(self, mode: str) -> WorkflowResponse:
        if mode == "mixed":
            self.state.pending_confirmations = ["mixed_weights"]
            self._refresh_blockers()
            return self._one("confirm_mixed_weights")
        self.state.confirmed_values["single_method_review"] = "pending"
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
        if target == "blockchain_process_names":
            self.state.confirmed_values[target] = [item for item in value.replace(",", " ").split() if item]
        else:
            self.state.confirmed_values[target] = value
        return target

    def _assign_current_real_node_value(self, text: str) -> None:
        key = self.state.current_question_id
        value = text.strip().strip("\"'")
        if key == "blockchain_process_names":
            self.state.confirmed_values[key] = [item for item in value.replace(",", " ").split() if item]
        else:
            self.state.confirmed_values[key] = value

    def _after_value_assignment(self, assigned: str) -> WorkflowResponse:
        self._refresh_blockers()
        messages = [t(self.state.language, "recorded_value", name=assigned)]
        if self.state.confirmed_values.get("use_fake_node") is False:
            next_response = self._ask_next_real_node_value()
            messages.extend(next_response.messages)
        return WorkflowResponse(True, messages)

    def _ask_next_real_node_value(self) -> WorkflowResponse:
        self._refresh_blockers()
        for key in REAL_NODE_BLOCKERS:
            if _is_missing(self.state.confirmed_values.get(key)):
                self.state.current_question_id = key
                self.state.stage = f"collect_{key}"
                return WorkflowResponse(True, [t(self.state.language, "ask_required_value", name=_display_key(key))])
        self.state.stage = "select_rpc_mode"
        self.state.current_question_id = "rpc_mode"
        return WorkflowResponse(True, [
            t(self.state.language, "real_node_required_done"),
            t(self.state.language, "ask_rpc_mode"),
        ])


def extract_chain(text: str) -> str:
    lowered = text.lower()
    for chain in KNOWN_CHAINS:
        if chain in lowered:
            return chain
    return ""


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
