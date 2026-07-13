"""LLM intent resolver for free-form Harness turns."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

try:
    from llm.providers import provider_from_config
    from llm.types import LLMMessage, LLMRequest
except ModuleNotFoundError:  # package import path used by tests
    from llm.providers import provider_from_config
    from llm.types import LLMMessage, LLMRequest

from .state import DEFAULT_GROUP_ORDER, AgentGraphState
from .state import PendingQuestion

try:
    from agent.onboarding.families import SUPPORTED_FAMILIES
except ModuleNotFoundError:  # package import path used by tests / product script
    from onboarding.families import SUPPORTED_FAMILIES

# Import rather than retype: `state.DEFAULT_GROUP_ORDER` is the single source
# of truth for group membership (architecture audit Finding A). The LLM must
# never be allowed to target a group that does not actually exist.
ALLOWED_GROUPS = list(DEFAULT_GROUP_ORDER)

# Single source of truth for the supported adapter families (audit Finding C3):
# derive prompt/payload copies from `onboarding.families.SUPPORTED_FAMILIES`.
ADAPTER_FAMILIES = list(SUPPORTED_FAMILIES)
# The identity/hint resolvers additionally accept these two non-family answers.
ADAPTER_FAMILY_HINT_ENUM = "|".join(ADAPTER_FAMILIES + ["unsupported", "unknown"])

ALLOWED_ACTION_TYPES = [
    "greeting",
    "choose_target_mode",
    "choose_chain",
    "change_chain",
    "reset_session",
    "change_group",
    "go_back",
    "ask_capabilities",
    "answer_opening_question",
    "analyze_evidence",
    "analyze_report",
    "set_rpc_mode",
    "set_qps_mode",
    "set_qps_override",
    "set_observability",
    "set_sync_observe_source",
    "set_accounts_presence",
    "start_custom_rpc",
    "propose_config_values",
    "answer_pending",
    "unknown",
]


def resolve_action_queue(state: AgentGraphState, text: str) -> dict[str, Any]:
    """Return ordered typed action proposals for a free-form user turn."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_action_queue_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(_action_queue_payload(state, text), ensure_ascii=False, sort_keys=True),
                    ),
                ],
                temperature=0.0,
                max_tokens=1400,
            )
        )
        return _parse_action_queue(response.text)
    except Exception as exc:
        return {
            "actions": [{"type": "unknown", "reason": f"action queue resolver failed: {type(exc).__name__}", "confidence": "low"}],
            "reason": "resolver failed",
        }


def _action_queue_prompt() -> str:
    return (
        "You are the action-queue resolver for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. "
        "You do not mutate state, do not ask workflow questions, and do not invent configuration values. "
        "Map the user turn into an ordered list of typed actions for the LangGraph Harness.\n"
        f"Allowed action types: {', '.join(ALLOWED_ACTION_TYPES)}.\n"
        "Allowed target_mode values: fake-node, real-node, sync-observe.\n"
        "Allowed rpc_mode values: single, mixed.\n"
        "Allowed qps_mode values: quick, standard, intensive.\n"
        "Allowed observability_mode values: disabled, local, exporter.\n"
        "Allowed sync_observe_source values: existing_local_node, endpoint_only, client_setup, demo_only.\n"
        f"Allowed group values: {', '.join(ALLOWED_GROUPS)}.\n"
        "If the user says several things in one turn, return several actions in the same order a helpful product agent should handle them. "
        "If the user asks how to restart, clear previous config, start over, or begin from scratch without directly commanding it, use answer_opening_question with topic=reset_help. "
        "If the user directly asks to reset, clear previous config, start over, discard old config, or restart the Agent configuration, use reset_session. "
        "If the user asks whether startup environment inference still exists after clearing config, what values were inferred at startup (cloud provider, deployment, CPU, memory, disk candidates, network candidates, dependency status), or whether an environment-dependency detection script already ran and what it found, use answer_opening_question with topic=startup_discovery. This is different from topic=current_config, which is about confirmed benchmark configuration, not startup discovery. "
        "If the user asks to switch response language and also gives configuration intent, ignore the language switch as an action and still extract the configuration actions. "
        "If the user asks about capabilities/support AND also expresses a benchmark/run/closed-loop goal or mentions a chain, include ask_capabilities plus the benchmark actions; do not let the capability answer replace the benchmark goal. "
        "If the user asks who you are, where you come from, where you are going, what you are for, or what kind of agent this is, use answer_opening_question with topic=identity. "
        "If the user asks what the Agent can do, use answer_opening_question with topic=agent_capabilities. "
        "If the user asks which chains, adapter families, RPC methods, or templates are supported, use answer_opening_question with topic=supported_chains. "
        "If the user asks what RPC workload/methods a SPECIFIC named chain has (e.g. 'bsc 有哪些 rpc workload', 'what methods does solana support', 'solana 的默认 rpc method'), use answer_opening_question with topic=supported_chains and set subject to that chain name. This is a question, NOT a benchmark request: do not also emit choose_chain/change_chain for it. "
        "If the user asks what 'this' current prompt/menu/pending question is for, use answer_opening_question with topic=current_context. "
        "If the user asks what is currently selected/configured, current chain, current mode, current settings, or current state, use answer_opening_question with topic=current_config. "
        "If the user wants to quickly try/validate/run something but is unsure which mode or chain to choose, use answer_opening_question with topic=recommendation; do not choose a mode unless they explicitly choose one. "
        "If the user asks what they must do/provide/prepare to run a benchmark, use answer_opening_question with topic=requirements. "
        "If the user asks whether THIS host/machine/environment can run a benchmark, is capable/ready, is powerful enough, or can handle the load (e.g. '我这台机器能不能跑', '环境准备好了吗', 'can my machine run this', 'is my env ready', 'is this box enough'), use answer_opening_question with topic=environment_readiness. This is about the current host's readiness/capacity, distinct from topic=requirements (what the user must provide) and topic=startup_discovery (what was inferred at startup). A question about a specific CHAIN being runnable ('bsc 能不能跑') is not this topic — that is topic=supported_chains with the chain in subject. "
        "If the user asks how the workflow runs or what the steps are, use answer_opening_question with topic=workflow. "
        "If the user asks what preflight or smoke is or does (the pre-run checks before a benchmark), use answer_opening_question with topic=execution_preflight_smoke. "
        "This topic is only for a what-is/explain question; a request to RUN, START, or EXECUTE preflight/smoke is not this topic — use change_group group=preflight_smoke_execution for that. "
        "If the user asks the difference between fake-node, real-node, and sync-observe, use answer_opening_question with topic=mode_comparison. "
        "If the user asks how to measure QPS capacity, maximum throughput, maximum requests, saturation, latency under load, or bottlenecks, use answer_opening_question with topic=performance_benchmark_guidance unless they also explicitly selected a target mode. "
        "If that same turn names a chain, include both actions: answer_opening_question topic=performance_benchmark_guidance first, then choose_chain/change_chain. "
        "If the user says they do not understand, asks whether you understood, says you answered the wrong question, or asks what you mean, use answer_opening_question with topic=correction. "
        "If the user asks what a config field/group means, use answer_opening_question with topic=config_explanation and subject when known. "
        "If the user asks about adding chains, adding custom RPC methods, or extension paths without starting the concrete workflow, use answer_opening_question with topic=extension. If the question names a SPECIFIC chain (e.g. 'bsc 的自定义 rpc method 如何添加', 'how do I add a custom RPC method for solana'), set subject to that chain name. This is a how-to question, NOT a benchmark request: do not also emit choose_chain/change_chain/start_custom_rpc for it. "
        "If the user says they are unsure between fake-node and real-node, do not choose a target mode; preserve chain/capability actions and let the Harness ask the target-mode question. "
        "Use choose_target_mode only when the user explicitly says fake-node, simulated/mock node, real-node, real node,真实节点,模拟节点,sync-observe, observe sync, observe block catch-up/import, 观察追块, 追块, or an equivalent explicit mode. "
        "Never infer real-node merely because the user says test/benchmark/测试/压测. "
        "If no target mode is explicit, do not emit choose_target_mode; emit other actions and let the Harness ask the target-mode question. "
        "When you emit choose_target_mode, include target_mode_explicit=true. "
        "If the same turn explicitly mentions both target mode and chain, include both. "
        "If the user names a chain while another chain is confirmed, use change_chain. "
        "If the user says they want to test another/different chain but does not provide the chain name, use change_group group=chain_identity; do not continue the current preflight/smoke path. "
        "If no chain is confirmed, use choose_chain. "
        "For a chain not listed in configured chains, still emit choose_chain/change_chain with the raw chain_text; the Harness will run chain identity gates. "
        "If the user explicitly states a likely protocol/family for an unknown chain, include adapter_family and chain_exists=true while still leaving final confirmation to the Harness. "
        "Map EVM, Ethereum-compatible, JSON-RPC, or eth_* RPC wording to adapter_family=jsonrpc. "
        "If the user changes mode from RPC benchmark to sync observation, include choose_target_mode target_mode=sync-observe and preserve any chain mention. "
        "If the user asks to jump to RPC/QPS/disk/network/observability/report/logs, use change_group with the closest allowed group. "
        "Even if the sentence contains go back/previous/return, when it names a configuration area such as RPC, QPS, disk, network, observability, report, or logs, use change_group instead of go_back. "
        "Use go_back only when the user asks for a generic previous step and does not name an area. "
        "If the current pending question is custom_rpc_endpoint or new_chain_endpoint (validating a custom RPC method's endpoint, or a new/unsupported chain's endpoint) and the user wants to cancel, give up, abandon this method, or stop trying to validate it (e.g. 算了/取消/放弃/不弄了/never mind/forget it/stop this), use go_back if no area is named, or change_group group=workload_rpc when they mention workload/method/RPC; never re-render the same endpoint question as if their words were an attempted endpoint. "
        "Use set_rpc_mode only when the user clearly selects single or mixed. "
        "Use set_qps_mode only when the user explicitly says quick, standard, or intensive. "
        "If the user wants to configure or adjust QPS but does not name quick/standard/intensive, use change_group group=qps_profile instead. "
        "If the user wants to adjust advanced tuning, monitoring, or bottleneck-threshold settings (monitoring interval, disk monitor rate, success-rate/latency thresholds, or CPU/memory/disk/network bottleneck thresholds) and there is no active advanced_tuning pending question, use change_group group=advanced_tuning instead of answering or explaining. "
        "Use set_qps_override when the user gives concrete QPS parameter values, including natural language such as initial qps=5, max qps 100, qps step 10, or duration 30 seconds. "
        "For qps_overrides use keys INITIAL_QPS, MAX_QPS, QPS_STEP, and DURATION. "
        "For any language, when the user asks to configure QPS and also gives a concrete mode such as quick/standard/intensive, emit change_group group=qps_profile followed by set_qps_mode. "
        "If the user is answering a disk/network/provider question but says '先配置 QPS', '先把 QPS 改成 quick', or 'configure QPS first, then return to disk/network/provider', emit change_group group=qps_profile followed by set_qps_mode when quick/standard/intensive is explicit; do not treat it as an answer to the current pending question. "
        "When the user says to configure one group first and then return to the current/pending group, do not emit a second change_group for the return; the Harness fallback path will resume the earliest incomplete group after the first group is confirmed. "
        "Use set_observability only when the user clearly selects, disables, or asks to keep off disabled/local/exporter observability. "
        "For observability, 'local Grafana', 'local Prometheus/Grafana', 'start Grafana', or 'open Grafana' means observability_mode=local. "
        "Use observability_mode=disabled when the user says not to enable observability, do not start/open Grafana, do not start/open Prometheus, keep Grafana off, 暂时不要开 Grafana/Prometheus, 不开启监控, or 禁用可观测性. "
        "Use observability_mode=exporter only when the user explicitly says exporter-only, existing Prometheus will scrape it, or connect to their own Prometheus without starting local Prometheus/Grafana. "
        "If the user asks whether Grafana/Prometheus/observability is currently disabled or wants to confirm that setting while another question is pending, emit change_group group=observability; if the user's wording also says it should be disabled, emit set_observability observability_mode=disabled. "
        "Use set_sync_observe_source when the user clearly chooses an existing local node, endpoint-only observation, Agent-prepared client setup, or demo-only sync-observe plumbing. "
        "Use set_accounts_presence when the user clearly says whether there is a separate accounts/state disk. "
        "Examples: no separate accounts disk, 没有 accounts 盘 => has_accounts_device=false; separate accounts disk, 有独立 accounts 盘 => has_accounts_device=true. "
        "Use start_custom_rpc when the user clearly wants to add, validate, replace, or test a concrete custom RPC method. "
        "If the user only asks whether custom RPC methods can be added, how custom RPC works, or whether it is possible, use answer_opening_question topic=extension instead of start_custom_rpc. "
        "For start_custom_rpc, include rpc_method when the method name is explicit, rpc_endpoint when the user gives a URL for validating it, and rpc_schema_evidence when the user pasted request/response/docs. "
        "If one turn contains both configuration facts and a custom RPC request, emit propose_config_values first, then start_custom_rpc, so the Harness can review config and then continue the custom RPC workflow. "
        "Do not put custom RPC method names into unmapped_values when start_custom_rpc can represent the intent. "
        "Use propose_config_values when the user pastes YAML, JSON, env vars, shell snippets, tables, multi-line notes, or mixed natural language that contains benchmark configuration facts across one or more groups. "
        "For propose_config_values, map only known AnyChain fields into config_values and put anything else into unmapped_values. "
        "Do not use propose_config_values for chain names, target mode, QPS mode, RPC mode, observability, or sync-observe source when a more specific action exists. "
        "If the user only asks whether you can analyze logs/errors but does not include the actual log/error content, use answer_opening_question topic=evidence_help. "
        "If the user pasted actual logs/errors/transcripts, use analyze_evidence. "
        "If the user asks about reports, metrics, or artifacts, use analyze_report. "
        "If the user is only greeting, use greeting. "
        "Examples: 'I want to benchmark BSC max throughput and find bottlenecks, not just test the tool itself' => actions answer_opening_question topic=performance_benchmark_guidance then choose_chain chain_text='BSC'; no choose_target_mode. "
        "Examples: 'I want to quickly validate the framework, not sure fake-node or real-node, maybe BNB, also show supported chains' => actions ask_capabilities then choose_chain chain_text='BNB'; no choose_target_mode. "
        "If unsure, return one unknown action with a concise reason.\n"
        "Schema: {actions:[{type:string,target_mode?:string,target_mode_explicit?:boolean,chain_text?:string,group?:string,rpc_mode?:string,"
        "qps_mode?:string,qps_overrides?:object,observability_mode?:string,sync_observe_source?:string,"
        "has_accounts_device?:boolean,config_values?:object,unmapped_values?:object|array,source_format?:string,reason?:string,"
        "confidence:'low'|'medium'|'high'}], reason?:string}."
    )


def _action_queue_payload(state: AgentGraphState, text: str) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    framework = state.get("framework_summary") or {}
    custom_rpc = state.get("custom_rpc") or {}
    sync = state.get("sync_observe") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    confirmed = state.get("confirmed_config") or {}
    return {
        "user_text": text,
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "",
        "active_subgroup": state.get("active_subgroup") or "",
        "pending_question": state.get("pending_question") or {},
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "confirmed_chain": identity.get("canonical") or "",
        "chain_status": identity.get("status") or "",
        "rpc_mode": state.get("rpc_mode") or "",
        "qps_profile": {
            "mode": qps.get("mode"),
            "confirmed": qps.get("confirmed"),
            "default_decision_made": qps.get("default_decision_made"),
            "overrides": qps.get("overrides") or {},
        },
        "custom_rpc": {
            "status": custom_rpc.get("status"),
            "method": custom_rpc.get("method"),
            "scope": custom_rpc.get("scope"),
        },
        "sync_observe": {
            "source": sync.get("source"),
            "stop_condition": sync.get("stop_condition"),
        },
        "observability": {
            "mode": observability.get("mode"),
        },
        "confirmed_config_keys": sorted(confirmed.keys()),
        "framework_summary": {
            "chain_count": framework.get("chain_count"),
            "family_count": framework.get("family_count"),
            "unique_rpc_method_count": framework.get("unique_rpc_method_count"),
            "chains": framework.get("chains", [])[:120],
            "adapter_families": ADAPTER_FAMILIES,
        },
        "web_research": state.get("web_research") or {},
        "known_config_fields_for_proposals": [
            "CLOUD_REGION",
            "CLOUD_ZONE",
            "MACHINE_TYPE",
            "LEDGER_DEVICE",
            "DATA_VOL_TYPE",
            "DATA_VOL_SIZE",
            "DATA_VOL_MAX_IOPS",
            "DATA_VOL_MAX_THROUGHPUT",
            "ACCOUNTS_DEVICE",
            "ACCOUNTS_VOL_TYPE",
            "ACCOUNTS_VOL_SIZE",
            "ACCOUNTS_VOL_MAX_IOPS",
            "ACCOUNTS_VOL_MAX_THROUGHPUT",
            "NETWORK_INTERFACE",
            "NETWORK_MAX_BANDWIDTH_GBPS",
            "LOCAL_RPC_URL",
            "MAINNET_RPC_URL",
            "BLOCKCHAIN_PROCESS_NAMES",
            "RPC_MODE",
        ],
        "output_schema": {
            "actions": [
                {
                    "type": "one allowed action type",
                    "target_mode": "optional fake-node|real-node|sync-observe",
                    "target_mode_explicit": "true only when target mode is explicit in the user text",
                    "chain_text": "optional raw chain name from user",
                    "adapter_family": f"optional {ADAPTER_FAMILY_HINT_ENUM} for unknown-chain hints",
                    "chain_exists": "optional boolean when the user claims or implies the unknown chain exists",
                    "group": "optional allowed group",
                    "topic": "optional for answer_opening_question: identity|agent_capabilities|supported_chains|current_context|current_config|startup_discovery|requirements|environment_readiness|workflow|mode_comparison|execution_preflight_smoke|performance_benchmark_guidance|recommendation|config_explanation|extension|correction|evidence_help|reset_help",
                    "subject": "optional field, chain, method, or concept being asked about",
                    "rpc_mode": "optional single|mixed",
                    "qps_mode": "optional quick|standard|intensive",
                    "qps_overrides": "optional object",
                    "observability_mode": "optional disabled|local|exporter",
                    "sync_observe_source": "optional existing_local_node|endpoint_only|client_setup|demo_only",
                    "has_accounts_device": "optional boolean for set_accounts_presence",
                    "rpc_method": "optional custom RPC method name for start_custom_rpc",
                    "rpc_endpoint": "optional endpoint URL for start_custom_rpc validation",
                    "rpc_schema_evidence": "optional request/response/docs evidence for start_custom_rpc",
                    "config_values": "optional object of known AnyChain config fields for propose_config_values",
                    "unmapped_values": "optional object/array of extracted facts that do not map to known fields",
                    "source_format": "optional yaml|json|env|shell|table|text|mixed",
                    "reason": "optional short reason",
                    "confidence": "low|medium|high",
                }
            ],
            "reason": "optional short reason",
        },
    }


def resolve_unknown_chain_identity(state: AgentGraphState, chain_text: str) -> dict[str, Any]:
    """Ask the configured model whether an unknown chain appears to exist."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_chain_identity_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(_chain_identity_payload(state, chain_text), ensure_ascii=False, sort_keys=True),
                    ),
                ],
                temperature=0.0,
                max_tokens=900,
            )
        )
        payload = _parse_json_object(response.text)
    except Exception as exc:
        payload = {"chain_exists": None, "reason": f"chain identity resolver failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("chain_text", chain_text)
    return payload


def extract_chain_mention(state: AgentGraphState, text: str) -> dict[str, Any]:
    """Extract a chain/network mention from a turn that was routed elsewhere."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_chain_mention_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(_chain_mention_payload(state, text), ensure_ascii=False, sort_keys=True),
                    ),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        )
        payload = _parse_json_object(response.text)
    except Exception as exc:
        payload = {"found": False, "reason": f"chain mention extraction failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("found", False)
    return payload


def extract_rpc_schema_from_evidence(state: AgentGraphState, evidence: str, *, method_hint: str = "") -> dict[str, Any]:
    """Extract a typed RPC schema draft from user-provided evidence.

    The draft is not trusted by itself. The Harness must still ask the user to
    confirm it and then probe the endpoint before accepting the method.
    """

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_rpc_schema_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            _rpc_schema_payload(state, evidence, method_hint=method_hint),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )
        payload = _parse_json_object(response.text)
    except Exception as exc:
        payload = {"status": "failed", "reason": f"schema extraction failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("status", "draft")
    payload.setdefault("method", method_hint)
    return payload


def resolve_pending_choice(state: AgentGraphState, text: str, question: PendingQuestion) -> dict[str, Any]:
    """Map a free-form answer to the active typed question options."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_pending_choice_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            _pending_choice_payload(state, text, question),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=600,
            )
        )
        payload = _parse_json_object(response.text)
    except Exception as exc:
        payload = {"matched": False, "reason": f"pending choice resolver failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("matched", False)
    return payload


def _chain_identity_prompt() -> str:
    return (
        "You resolve blockchain chain names for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. "
        "The input chain is not one of the configured AnyChain templates. "
        "Decide whether it appears to be a real blockchain/network name, "
        "whether it is just a typo/partial alias for a known chain, and what adapter family it likely uses. "
        "Do not claim certainty if unsure. "
        f"Allowed adapter_family values: {', '.join(ADAPTER_FAMILIES + ['unsupported', 'unknown'])}. "
        "Schema: {chain_exists:boolean|null, canonical_chain_name:string, adapter_family:string, possible_known_chain:string, "
        "evidence_summary:string, confidence:'low'|'medium'|'high'}."
    )


def _chain_identity_payload(state: AgentGraphState, chain_text: str) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    return {
        "unknown_chain_text": chain_text,
        "known_chains": framework.get("chains", [])[:120],
        "supported_adapter_families": ADAPTER_FAMILIES,
        "web_research": state.get("web_research") or {},
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
    }


def _chain_mention_prompt() -> str:
    return (
        "You extract blockchain or network names from one AnyChain user turn. "
        "Return one JSON object only. Do not explain. "
        "Extract only an explicit chain/network/product name that the user wants to benchmark or observe. "
        "Do not treat cloud regions, zones, instance types, disks, URLs, QPS values, or generic words as chain names. "
        "If no explicit chain/network is present, return found=false. "
        "Known examples include Solana, Ethereum, BNB, BSC, Bitcoin, Flow, Monad, Base, Polygon. "
        "Schema: {found:boolean, chain_text:string, reason:string, confidence:'low'|'medium'|'high'}."
    )


def _chain_mention_payload(state: AgentGraphState, text: str) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    return {
        "user_text": text,
        "known_chains": framework.get("chains", [])[:120],
        "current_target_mode": state.get("target_mode") or "",
        "current_workflow_mode": state.get("workflow_mode") or "",
        "current_chain": (state.get("chain_identity") or {}).get("canonical") or "",
    }


def _rpc_schema_prompt() -> str:
    return (
        "You extract RPC method schemas for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. "
        "The user may provide a curl command, JSON-RPC request, response sample, docs excerpt, URL text, or endpoint transcript. "
        "First classify the evidence kind and transport before extracting a method. "
        "Do not treat a URL, REST path, or REST documentation title as a JSON-RPC method name. "
        "Infer only what is supported by the evidence; mark unknown fields as unknown. "
        "Do not invent parameters. "
        "Schema: {status:'draft'|'failed', evidence_kind:'jsonrpc_request'|'rest_endpoint'|'rest_path'|'response_sample'|'docs_excerpt'|'method_name'|'mixed'|'unknown', "
        "transport:'jsonrpc'|'rest'|'unknown', method:string, endpoint_url:string, rest_path:string, http_method:string, "
        "params:[{index:number,name:string,type:string,meaning:string,example:any,required:boolean}], "
        "params_json:any, response_summary:string, response_fields:[{name:string,type:string,meaning:string}], "
        "auth_notes:string, rate_limit_notes:string, conflicts:[string], confidence:'low'|'medium'|'high', reason:string, "
        "evidence_summary:string}."
    )


def _rpc_schema_payload(state: AgentGraphState, evidence: str, *, method_hint: str) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    return {
        "evidence": evidence,
        "method_hint": method_hint,
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "chain": identity.get("canonical") or identity.get("raw") or "",
        "adapter_family": identity.get("adapter_family") or "",
        "web_research": state.get("web_research") or {},
    }


def _pending_choice_prompt() -> str:
    return (
        "You map a user's free-form answer to one option of the current AnyChain typed pending question. "
        "Return one JSON object only. Do not explain. "
        "Only choose an option when the user's intent clearly matches that option. "
        "Use option label, value, and description as the only source of option semantics. "
        "Do not infer configuration values. Do not advance to another workflow group. "
        "If the user text is an assignment or a list of assignments such as key=value, method=weight, "
        "or JSON/config content while the current options are not asking for that kind of input, return matched=false. "
        "If no single option clearly matches, return matched=false. "
        "Schema: {matched:boolean, selected_value:any, selected_label:string, confidence:'low'|'medium'|'high', reason:string}."
    )


def _pending_choice_payload(state: AgentGraphState, text: str, question: PendingQuestion) -> dict[str, Any]:
    return {
        "user_text": text,
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "",
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "pending_question": {
            "id": question.get("id"),
            "group": question.get("group"),
            "kind": question.get("kind"),
            "prompt": question.get("prompt"),
            "options": question.get("options") or [],
            "field": question.get("field"),
        },
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "unknown", "reason": "model did not return valid JSON", "confidence": "low"}
    return payload if isinstance(payload, dict) else {"intent": "unknown", "reason": "model returned non-object JSON", "confidence": "low"}


def _parse_action_queue(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    actions_raw = payload.get("actions")
    if isinstance(actions_raw, dict):
        actions_raw = [actions_raw]
    if not isinstance(actions_raw, list):
        action = dict(payload)
        if "type" not in action and "intent" in action:
            action["type"] = action.get("intent")
        actions_raw = [action]
    actions: list[dict[str, Any]] = []
    for raw in actions_raw:
        if not isinstance(raw, dict):
            continue
        action = dict(raw)
        action_type = str(action.get("type") or action.get("intent") or "unknown").strip()
        if action_type not in ALLOWED_ACTION_TYPES:
            action_type = "unknown"
        action["type"] = action_type
        action.setdefault("confidence", "low" if action_type == "unknown" else "medium")
        actions.append(action)
    if not actions:
        actions = [{"type": "unknown", "reason": "model returned no actions", "confidence": "low"}]
    return {"actions": actions, "reason": payload.get("reason") or payload.get("reasoning") or ""}
