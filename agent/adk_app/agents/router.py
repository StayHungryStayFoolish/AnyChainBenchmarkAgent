"""Intent router contract for the ADK workflow."""

from __future__ import annotations

import re

from adk_app.workflow.schemas import IntentEntities, IntentRoute
from terminal.language import detect_language


SUPPORTED_INTENTS = [
    "START_BENCHMARK",
    "RESUME_JOB",
    "ANALYZE_ARTIFACTS",
    "ONBOARD_CHAIN_RPC",
    "CONFIG_HELP",
    "GENERAL_QA",
    "OUT_OF_SCOPE",
]


def route_user_intent(text: str, default_language: str = "en") -> dict:
    """Return a structured route for one user utterance.

    This is the offline-safe router implementation. In the ADK workflow, an LLM
    router may produce the same schema, but workflow gates must validate it before
    routing. Keeping this deterministic version lets CI test route semantics
    without live credentials.
    """
    language = detect_language(text, default_language)
    lowered = text.lower()
    entities = IntentEntities(
        chain=_extract_chain(lowered),
        rpc_methods=_extract_methods(text),
        rpc_mode=_extract_rpc_mode(lowered),
        target=_extract_target(lowered),
        job_id=_extract_job_id(text),
    )
    intent, confidence, missing = _intent(lowered, entities)
    return IntentRoute(
        intent=intent,
        confidence=confidence,
        language=language,
        entities=entities,
        missing_clarifications=missing,
    ).as_dict()


def _intent(lowered: str, entities: IntentEntities) -> tuple[str, float, list[str]]:
    if any(token in lowered for token in ("resume", "status", "logs", "job", "继续", "恢复", "状态", "日志")):
        return "RESUME_JOB", 0.9 if entities.job_id else 0.75, []
    if any(token in lowered for token in ("analyze", "report", "html", "artifact", "分析", "报告", "图表")):
        return "ANALYZE_ARTIFACTS", 0.85, []
    if any(token in lowered for token in ("onboard", "add chain", "new chain", "custom rpc", "新增", "添加", "新链", "rpc method")):
        return "ONBOARD_CHAIN_RPC", 0.88, []
    if any(token in lowered for token in ("config", "configure", "配置", "变量", "依赖", "install", "安装")):
        return "CONFIG_HELP", 0.78, []
    if any(token in lowered for token in ("benchmark", "test", "qps", "fake-node", "real-node", "压测", "测试")):
        missing = []
        if not entities.chain:
            missing.append("chain")
        if entities.target == "unknown":
            missing.append("target")
        return "START_BENCHMARK", 0.86, missing
    if any(token in lowered for token in ("hello", "hi", "你好", "你是谁", "what can you do", "能做什么")):
        return "GENERAL_QA", 0.75, []
    return "OUT_OF_SCOPE", 0.55, []


def _extract_chain(lowered: str) -> str:
    known = (
        "solana", "ethereum", "bitcoin", "polygon", "base", "bsc", "arbitrum",
        "optimism", "avalanche-c", "avalanche-x", "sui", "aptos", "polkadot",
        "kusama", "cosmos-hub", "osmosis", "near", "tron", "hedera",
    )
    for chain in known:
        if chain in lowered:
            return chain
    match = re.search(r"(?:chain|node|新增|添加|支持)\s+([A-Za-z0-9_-]{3,})", lowered)
    return match.group(1).strip("-_") if match else ""


def _extract_methods(text: str) -> list[str]:
    methods = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_./:-]{2,}\b", text):
        lowered = token.lower()
        if lowered in {"benchmark", "fake-node", "real-node", "chain", "node", "method", "methods"}:
            continue
        if "_" in token or "." in token or token.startswith(("get", "eth", "chain", "system", "wallet")):
            if token not in methods:
                methods.append(token)
    return methods[:12]


def _extract_rpc_mode(lowered: str) -> str:
    if "mixed" in lowered:
        return "mixed"
    if "single" in lowered:
        return "single"
    return ""


def _extract_target(lowered: str) -> str:
    if "fake-node" in lowered or "fake node" in lowered or "模拟节点" in lowered:
        return "fake-node"
    if (
        "real-node" in lowered
        or "real node" in lowered
        or "真实节点" in lowered
        or "真实" in lowered and "节点" in lowered
        or "real" in lowered and "node" in lowered
    ):
        return "real-node"
    return "unknown"


def _extract_job_id(text: str) -> str:
    match = re.search(r"\bjob_[0-9A-Za-z_-]+\b", text)
    return match.group(0) if match else ""
