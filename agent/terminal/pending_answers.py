"""Terminal helpers for typed pending-question answers."""

from __future__ import annotations

import re
from typing import Any

from knowledge.chain_identity import canonicalize_chain_scalar


_SIMPLE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}$")
_DEVICE_RE = re.compile(
    r"^(?:/dev/)?(?:(?:sd|vd|xvd)[a-z][0-9]*|nvme[0-9]+n[0-9]+p?[0-9]*|dm-[0-9]+|md[0-9]+|mapper/[A-Za-z0-9_.:+-]+|disk/by-[A-Za-z0-9_.:+-]+/[A-Za-z0-9_.:@+-]+)$"
)
_CLOUD_LOCATION_RE = re.compile(r"^(?:global|(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*-)[A-Za-z][A-Za-z0-9_-]{1,63})$")
_NUMBER_WITH_OPTIONAL_UNIT_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:\s*(?:gib|gb|tib|tb|mib/s|mb/s|gbps|iops))?$", re.IGNORECASE)
_PROCESS_NAMES_RE = re.compile(r"^[A-Za-z0-9_./:+-]+(?:[ ,]+[A-Za-z0-9_./:+-]+)*$")


def is_structural_pending_answer(text: str, question: dict[str, Any]) -> bool:
    raw = _normalize_structural_candidate(text, question)
    if not raw or "\n" in raw or "\r" in raw:
        return False
    if "?" in raw or "？" in raw:
        return False
    lowered = raw.lower()
    if lowered in {"back", "previous", "undo"}:
        return True
    kind = str(question.get("kind") or question.get("expected_answer") or "")
    if lowered in {"y", "yes", "n", "no"}:
        return _confirmation_token_allowed_for_question(lowered, kind, question)
    if _matches_pending_option(raw, question.get("options") or []):
        return True
    if str(question.get("id") or "") == "target_mode" and _target_mode_option(raw, question.get("options") or []):
        return True
    if kind in {"yes_no", "confirmation"} and raw.isdigit() and len(raw) <= 2:
        return True
    if kind in {"numbered_choice", "multi_select", "device"} and raw.isdigit():
        return True
    if kind == "url" and raw.startswith(("http://", "https://", "ws://", "wss://")):
        return True
    if kind == "chain":
        return _is_known_chain_answer(raw, question)
    if _is_location_question(question) and _is_device_token(raw):
        return True
    if kind == "device" and (_is_device_token(raw) or _is_simple_identifier(raw)):
        return True
    if kind in {"manual_value", "qps_profile", "rpc_weight", "rpc_method"}:
        return _matches_manual_value_shape(raw, question)
    return False


def is_unbound_structural_answer(text: str) -> bool:
    raw = _strip_scalar_wrappers(text or "")
    if not raw or "\n" in raw or "\r" in raw:
        return False
    lowered = raw.lower()
    if lowered in {"y", "yes", "n", "no", "back", "previous", "undo"}:
        return True
    if raw.isdigit() and len(raw) <= 2:
        return True
    return False


def _matches_pending_option(raw: str, options: list[dict[str, Any]]) -> bool:
    lowered = raw.strip().lower()
    if not lowered:
        return False
    for index, option in enumerate(options, start=1):
        candidates = {
            str(index).strip().lower(),
            str(option.get("id", "")).strip().lower(),
            str(option.get("label", "")).strip().lower(),
            str(option.get("value", "")).strip().lower(),
        }
        if lowered in {item for item in candidates if item}:
            return True
    return False


def _target_mode_option(raw: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    mode = _target_mode_from_text(raw)
    if not mode:
        return None
    for option in options:
        if str(option.get("value") or "").strip().lower() == mode:
            return option
    return None


def target_mode_from_text(raw: str) -> str:
    """Return an explicit target mode from user text, if one is present."""
    return _target_mode_from_text(raw)


def _target_mode_from_text(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    compact = text.replace("_", "-")
    if _looks_like_composite_target_request(compact):
        return ""
    fake_mentioned = any(token in compact for token in ("fake-node", "fakenode", "fake node", "mock"))
    real_mentioned = any(token in compact for token in ("real-node", "realnode", "real node", "真实节点", "真实"))
    fake_negated = any(
        marker in compact
        for marker in (
            "不使用 fake",
            "不用 fake",
            "不要 fake",
            "not fake",
            "without fake",
            "no fake",
        )
    )
    if fake_negated:
        return "real-node"
    if real_mentioned and not fake_mentioned:
        return "real-node"
    if fake_mentioned and not real_mentioned:
        return "fake-node"
    return ""


def _looks_like_composite_target_request(text: str) -> bool:
    if not text:
        return False
    if any(marker in text for marker in ("切换", "改成", "换成", "change to", "switch to")):
        return False
    if any(marker in text for marker in ("smoke", "假设", "快速验证", "确认 agent", "确认框架", "can run")):
        return True
    return len(text.split()) > 8


def _is_known_chain_answer(raw: str, question: dict[str, Any]) -> bool:
    known_values = question.get("known_chains")
    if not isinstance(known_values, list) or not known_values:
        return False
    return bool(
        canonicalize_chain_scalar(
            raw,
            known_chains=[str(item).strip().lower() for item in known_values if str(item).strip()],
            aliases=question.get("chain_aliases") if isinstance(question.get("chain_aliases"), dict) else None,
        )
    )


def _confirmation_token_allowed_for_question(lowered: str, kind: str, question: dict[str, Any]) -> bool:
    _ = lowered, question
    if kind in {"yes_no", "confirmation"}:
        return True
    if kind in {"numbered_choice", "multi_select", "device"}:
        return True
    if kind in {"manual_value", "qps_profile", "rpc_weight", "chain", "rpc_method", "evidence_apply", "free_text"}:
        return True
    return False


def _matches_manual_value_shape(raw: str, question: dict[str, Any]) -> bool:
    field = str(question.get("field") or "").strip()
    question_id = str(question.get("id") or "").strip()
    key = field or question_id
    if not raw or len(raw) > 240:
        return False
    if key in {"CLOUD_REGION", "cloud_region"}:
        return bool(_CLOUD_LOCATION_RE.fullmatch(raw))
    if key in {"CLOUD_ZONE", "cloud_zone"}:
        return bool(_CLOUD_LOCATION_RE.fullmatch(raw))
    if key in {"MACHINE_TYPE", "machine_type"}:
        return _is_simple_identifier(raw)
    if key in {"LEDGER_DEVICE", "ACCOUNTS_DEVICE", "ledger_device", "accounts_device"}:
        return _is_device_token(raw)
    if key in {
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_SIZE",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
        "data_vol_size",
        "data_vol_max_iops",
        "data_vol_max_throughput",
        "accounts_vol_size",
        "accounts_vol_max_iops",
        "accounts_vol_max_throughput",
        "network_max_bandwidth_gbps",
    }:
        return bool(_NUMBER_WITH_OPTIONAL_UNIT_RE.fullmatch(raw))
    if key in {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE", "data_vol_type", "accounts_vol_type", "network_interface", "NETWORK_INTERFACE"}:
        return _is_simple_identifier(raw)
    if key in {"BLOCKCHAIN_PROCESS_NAMES", "BLOCKCHAIN_PROCESS_NAMES_STR", "blockchain_process_names"}:
        return bool(_PROCESS_NAMES_RE.fullmatch(raw))
    if question_id in {"benchmark_profile_adjust_item"}:
        return bool(_NUMBER_WITH_OPTIONAL_UNIT_RE.fullmatch(raw))
    if str(question.get("kind") or "") == "rpc_weight":
        return bool(re.fullmatch(r"[A-Za-z0-9_.:/-]+\s*=\s*[0-9]{1,3}(?:\s*,\s*[A-Za-z0-9_.:/-]+\s*=\s*[0-9]{1,3})*", raw))
    if str(question.get("kind") or "") == "rpc_method":
        return _is_simple_identifier(raw)
    return _is_simple_identifier(raw)


def _normalize_structural_candidate(text: str, question: dict[str, Any]) -> str:
    raw = (text or "").strip()
    kind = str(question.get("kind") or question.get("expected_answer") or "")
    if kind in {"free_text", "rpc_weight", "evidence_apply"}:
        return raw
    return _strip_scalar_wrappers(raw)


def _strip_scalar_wrappers(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"', "“", "‘"} and text[-1] in {"`", "'", '"', "”", "’"}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；"}:
        text = text[:-1].strip()
    return text


def _is_simple_identifier(raw: str) -> bool:
    return bool(_SIMPLE_TOKEN_RE.fullmatch(raw))


def _is_device_token(raw: str) -> bool:
    return bool(_DEVICE_RE.fullmatch(raw))


def _is_location_question(question: dict[str, Any]) -> bool:
    key = str(question.get("field") or question.get("id") or "").strip()
    return key in {"CLOUD_REGION", "CLOUD_ZONE", "cloud_region", "cloud_zone"}


def compose_pending_answer_followup(answer: str, result: dict[str, Any]) -> str:
    selected = result.get("selected")
    state = result.get("state") or {}
    return (
        "The user answered the active typed pending_question. The terminal has "
        "already applied the answer through answer_pending_question; do not "
        "reinterpret the raw short answer. Continue from the updated workflow "
        "state. Ask exactly one next blocking question, run an allowed next "
        "tool, or provide the next concise status.\n\n"
        f"Raw user answer: {answer}\n"
        f"Applied question_id: {result.get('question_id', '')}\n"
        f"Answer kind: {result.get('answer_kind', '')}\n"
        f"Selected: {selected}\n"
        f"Allowed next actions: {state.get('allowed_next_actions') or result.get('next_actions') or []}\n"
    )
