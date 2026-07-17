"""Turn-clause and semantic-unit contracts for model action plans.

This module does not classify intent.  It gives each independent input block a
stable turn-local id.  The model then partitions prose clauses into exact
source spans, and this module verifies that every character and semantic unit
is accounted for by a typed action or an explicit unresolved decision before
the transaction can be admitted.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？；])\s*|(?<=[!?;])\s+|(?<=\.)\s+(?=[A-Z])")
_URL_RE = re.compile(r"https?://[^\s'\"`<>]+", re.IGNORECASE)
_WIRE_METHOD_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:[._:][A-Za-z0-9]+)+"
    r"|[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*"
    r")(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class TurnClause:
    clause_id: str
    text: str
    input_shape: str = "prose"

    def as_dict(self) -> dict[str, str]:
        return {
            "clause_id": self.clause_id,
            "text": self.text,
            "input_shape": self.input_shape,
        }


@dataclass(frozen=True)
class PlanCoverageResult:
    valid: bool
    errors: tuple[str, ...]
    unresolved_clauses: tuple[str, ...]
    rejected_action_indexes: tuple[int, ...] = ()
    incomplete_unit_ids: tuple[str, ...] = ()


def segment_user_turn(text: str) -> tuple[TurnClause, ...]:
    """Split one complete turn without interpreting product intent.

    Fenced code, JSON, curl, YAML-like, and env-like multiline blocks remain
    atomic so their syntax and request/response relationship are not damaged.
    Ordinary prose is split only on explicit sentence boundaries.
    """

    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ()
    parts: list[tuple[str, str]] = []
    regions = _split_input_shapes(raw)
    index = 0
    while index < len(regions):
        value, shape = regions[index]
        if shape == "structured":
            parts.append((value, shape))
            index += 1
            continue
        prose_parts: list[str] = []
        for line in (item.strip() for item in value.splitlines()):
            if not line:
                continue
            split = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(line) if item.strip()]
            prose_parts.extend(split)
        if (
            prose_parts
            and index + 1 < len(regions)
            and regions[index + 1][1] == "structured"
            and prose_parts[-1].endswith((":", "："))
        ):
            parts.extend((item, "prose") for item in prose_parts[:-1])
            parts.append((f"{prose_parts[-1]}\n{regions[index + 1][0]}", "structured"))
            index += 2
            continue
        parts.extend((item, "prose") for item in prose_parts)
        index += 1
    if not parts:
        parts = [(raw, "prose")]
    return tuple(
        TurnClause(f"clause-{index}", value, shape)
        for index, (value, shape) in enumerate(parts, start=1)
    )


def validate_plan_coverage(
    payload: Mapping[str, Any],
    clauses: Sequence[TurnClause],
) -> PlanCoverageResult:
    """Validate complete clause-to-action accounting for one model plan."""

    actions = payload.get("actions")
    action_list = list(actions) if isinstance(actions, list) else []
    expected = {clause.clause_id: clause for clause in clauses}
    raw_units = payload.get("semantic_units")
    if not isinstance(raw_units, list):
        raw_units = []
    raw_units, span_errors = _canonicalize_semantic_units(raw_units, expected)
    units_by_clause: dict[str, list[Mapping[str, Any]]] = {
        clause_id: [] for clause_id in expected
    }
    seen_unit_ids: set[str] = set()
    errors: list[str] = list(span_errors)
    unresolved: list[str] = []
    referenced_actions: set[int] = set()

    for raw in raw_units:
        if not isinstance(raw, Mapping):
            errors.append("semantic_units contains a non-object row")
            continue
        unit_id = str(raw.get("unit_id") or "").strip()
        if not unit_id:
            errors.append("semantic unit has no unit_id")
            continue
        if unit_id in seen_unit_ids:
            errors.append(f"duplicate semantic unit id: {unit_id}")
            continue
        seen_unit_ids.add(unit_id)
        clause_id = str(raw.get("clause_id") or "").strip()
        if clause_id not in expected:
            errors.append(f"unknown clause id for {unit_id}: {clause_id or '<missing>'}")
            continue
        start = raw.get("start")
        end = raw.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(expected[clause_id].text)
        ):
            errors.append(f"invalid source span for {unit_id}: {start!r}:{end!r}")
            continue
        source_text = str(raw.get("source_text") or "")
        if source_text != expected[clause_id].text[start:end]:
            errors.append(f"source_text does not match its exact span for {unit_id}")
            continue
        units_by_clause[clause_id].append(raw)
        disposition = str(raw.get("disposition") or "").strip()
        indexes = raw.get("action_indexes")
        indexes = list(indexes) if isinstance(indexes, list) else []
        scope_constraint = str(raw.get("scope_constraint") or "").strip()
        if disposition == "unresolved":
            if not str(raw.get("reason") or "").strip():
                errors.append(f"unresolved semantic unit has no reason: {unit_id}")
            if indexes:
                errors.append(f"unresolved semantic unit has action indexes: {unit_id}")
            unresolved.append(source_text)
            continue
        if disposition == "context":
            if expected[clause_id].input_shape != "prose":
                errors.append(f"context semantic unit is not prose: {unit_id}")
            if not str(raw.get("reason") or "").strip():
                errors.append(f"context semantic unit has no reason: {unit_id}")
            if indexes:
                errors.append(f"context semantic unit has action indexes: {unit_id}")
            if scope_constraint:
                errors.append(f"context semantic unit has a scope constraint: {unit_id}")
            continue
        if disposition != "action":
            errors.append(f"invalid semantic unit disposition for {unit_id}: {disposition or '<missing>'}")
            continue
        if scope_constraint == "consultation_only" and not indexes:
            indexes = _derive_consultation_scope_indexes(unit_id, action_list, errors)
        if not indexes:
            errors.append(f"action semantic unit has no action index: {unit_id}")
            continue
        mapped_actions: list[Mapping[str, Any]] = []
        for value in indexes:
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(action_list):
                errors.append(f"invalid action index for {unit_id}: {value!r}")
                continue
            action = action_list[value]
            if not isinstance(action, Mapping):
                errors.append(f"mapped action is not an object for {unit_id}: {value}")
                continue
            referenced_actions.add(value)
            mapped_actions.append(action)
        if scope_constraint:
            if scope_constraint != "consultation_only":
                errors.append(f"invalid scope constraint for {unit_id}: {scope_constraint}")
            else:
                _validate_consultation_scope(unit_id, mapped_actions, errors)
        _validate_literal_anchors(unit_id, source_text, mapped_actions, errors)

    for clause_id, clause in expected.items():
        units = sorted(units_by_clause[clause_id], key=lambda item: int(item["start"]))
        if not units:
            errors.append(f"missing semantic units for {clause_id}")
            unresolved.append(clause.text)
            continue
        cursor = 0
        for unit in units:
            start = int(unit["start"])
            end = int(unit["end"])
            if start != cursor:
                relation = "overlap" if start < cursor else "gap"
                errors.append(
                    f"semantic unit partition has a {relation} in {clause_id} at {cursor}:{start}"
                )
            cursor = max(cursor, end)
        if cursor != len(clause.text):
            errors.append(
                f"semantic unit partition does not reach the end of {clause_id}: {cursor}:{len(clause.text)}"
            )

    orphaned = [
        index for index, action in enumerate(action_list)
        if isinstance(action, Mapping)
        and str(action.get("type") or "") != "unknown"
        and index not in referenced_actions
    ]
    errors.extend(f"unreferenced action index: {index}" for index in orphaned)
    return PlanCoverageResult(
        valid=not errors and not unresolved,
        errors=tuple(errors),
        unresolved_clauses=tuple(dict.fromkeys(unresolved)),
    )


def _canonicalize_semantic_units(
    raw_units: Sequence[Any],
    expected: Mapping[str, TurnClause],
) -> tuple[list[Any], list[str]]:
    """Derive exact source spans from ordered model-authored text anchors.

    Character offsets are control metadata, not semantic model output. The
    model identifies units with exact source excerpts; the Harness resolves a
    unique ordered placement and attaches only punctuation/whitespace between
    anchors. Any omitted prose or ambiguous placement remains invalid.
    """

    canonical = list(raw_units)
    dropped_indexes: set[int] = set()
    errors: list[str] = []
    indexes_by_clause: dict[str, list[int]] = {clause_id: [] for clause_id in expected}
    for index, raw in enumerate(raw_units):
        if isinstance(raw, Mapping):
            clause_id = str(raw.get("clause_id") or "").strip()
            if clause_id in indexes_by_clause:
                indexes_by_clause[clause_id].append(index)

    for clause_id, indexes in indexes_by_clause.items():
        if not indexes:
            continue
        clause = expected[clause_id]
        units = [raw_units[index] for index in indexes]
        if not all(isinstance(unit, Mapping) for unit in units):
            continue
        anchors = [str(unit.get("source_text") or "") for unit in units]
        if any(not anchor for anchor in anchors):
            errors.append(f"semantic unit has an empty source anchor in {clause_id}")
            continue
        if clause.input_shape == "structured":
            if all(anchor == clause.text for anchor in anchors):
                dispositions = {str(unit.get("disposition") or "").strip() for unit in units}
                scopes = {str(unit.get("scope_constraint") or "").strip() for unit in units}
                if dispositions != {"action"} or len(scopes) != 1:
                    errors.append(f"structured clause has incompatible semantic units: {clause_id}")
                    continue
                merged_indexes: list[int] = []
                for unit in units:
                    indexes_value = unit.get("action_indexes")
                    if isinstance(indexes_value, list):
                        for action_index in indexes_value:
                            if action_index not in merged_indexes:
                                merged_indexes.append(action_index)
                normalized = dict(units[0])
                normalized.update({
                    "start": 0,
                    "end": len(clause.text),
                    "source_text": clause.text,
                    "action_indexes": merged_indexes,
                })
                canonical[indexes[0]] = normalized
                dropped_indexes.update(indexes[1:])
                continue
            placement_sets = _unique_anchor_placements(clause.text, anchors)
            if not placement_sets:
                errors.append(f"structured source anchors omit content in {clause_id}")
                continue
            if len(placement_sets) != 1:
                errors.append(f"structured source anchors are ambiguous in {clause_id}")
                continue
            placements = placement_sets[0]
        else:
            placement_sets = _unique_anchor_placements(clause.text, anchors)
            if not placement_sets:
                errors.append(f"source anchors do not cover {clause_id} without omitted prose")
                continue
            if len(placement_sets) != 1:
                errors.append(f"source anchors are ambiguous in {clause_id}")
                continue
            placements = placement_sets[0]

        for unit_offset, raw_index in enumerate(indexes):
            start = 0 if unit_offset == 0 else placements[unit_offset][0]
            end = (
                placements[unit_offset + 1][0]
                if unit_offset + 1 < len(placements)
                else len(clause.text)
            )
            normalized = dict(raw_units[raw_index])
            normalized.update({
                "start": start,
                "end": end,
                "source_text": clause.text[start:end],
            })
            canonical[raw_index] = normalized
    return [item for index, item in enumerate(canonical) if index not in dropped_indexes], errors


def _unique_anchor_placements(
    text: str,
    anchors: Sequence[str],
) -> list[list[tuple[int, int]]]:
    solutions: list[list[tuple[int, int]]] = []

    def visit(index: int, cursor: int, placements: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if index == len(anchors):
            if _separator_only(text[cursor:]):
                solutions.append(list(placements))
            return
        anchor = anchors[index]
        position = text.find(anchor, cursor)
        while position >= 0:
            if _separator_only(text[cursor:position]):
                end = position + len(anchor)
                placements.append((position, end))
                visit(index + 1, end, placements)
                placements.pop()
            position = text.find(anchor, position + 1)

    visit(0, 0, [])
    return solutions[:2]


def _separator_only(value: str) -> bool:
    return all(character.isspace() or unicodedata.category(character)[0] in {"P", "S"} for character in value)


def _validate_consultation_scope(
    unit_id: str,
    mapped_actions: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    from .action_registry import action_is_turn_local

    if not mapped_actions:
        errors.append(f"consultation-only scope has no mapped actions: {unit_id}")
        return
    for action in mapped_actions:
        if not action_is_turn_local(dict(action)):
            errors.append(
                f"consultation-only scope maps to a durable action for {unit_id}: "
                f"{action.get('type') or '<missing>'}"
            )


def _derive_consultation_scope_indexes(
    unit_id: str,
    actions: Sequence[Any],
    errors: list[str],
) -> list[int]:
    from .action_registry import action_is_turn_local

    if not actions:
        errors.append(f"consultation-only scope has no actions to bind: {unit_id}")
        return []
    indexes: list[int] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or not action_is_turn_local(dict(action)):
            errors.append(
                f"consultation-only scope cannot bind a durable action for {unit_id}: "
                f"{action.get('type') if isinstance(action, Mapping) else '<invalid>'}"
            )
            return []
        indexes.append(index)
    return indexes


def _validate_literal_anchors(
    unit_id: str,
    source_text: str,
    actions: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    serialized = json.dumps(list(actions), ensure_ascii=False, sort_keys=True, default=str)
    for raw_url in _URL_RE.findall(source_text):
        url = raw_url.rstrip(".,;，；。)")
        if url and url not in serialized:
            errors.append(f"URL from {unit_id} is absent from its mapped actions")
    rpc_owner_actions = {
        "rpc_catalog_command",
        "rpc_workload_command",
        "secondary_handoff_command",
    }
    source_lower = source_text.casefold()
    declares_rpc_context = bool(
        re.search(r"(?:\brpc\b|\bmethod\b|[\"']method[\"']\s*:)", source_lower)
    )
    if not declares_rpc_context and not any(
        str(action.get("type") or "") in rpc_owner_actions
        for action in actions
    ):
        return
    source_without_urls = _URL_RE.sub(" ", source_text)
    for method in _WIRE_METHOD_CANDIDATE_RE.findall(source_without_urls):
        if method == method.upper():
            continue
        if not _mapped_action_preserves_wire_method(method, actions):
            errors.append(
                f"wire method {method!r} from {unit_id} is absent from its mapped actions"
            )


def _mapped_action_preserves_wire_method(
    method: str,
    actions: Sequence[Mapping[str, Any]],
) -> bool:
    """Require an exact method fact in its owning action or evidence payload."""

    for action in actions:
        if str(action.get("rpc_method") or "").strip() == method:
            return True
        for key in ("rpc_schema_evidence", "handoff_evidence", "evidence", "subject"):
            value = action.get(key)
            if isinstance(value, str) and method in value:
                return True
    return False


def _is_structured_block(text: str) -> bool:
    value = str(text or "").strip()
    if value.startswith("```") or value.startswith(("{", "[", "curl ", "curl\\")):
        return True
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) <= 1:
        return False
    structured = sum(
        bool(
            re.match(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*=", line)
            or re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*:", line)
            or line.startswith(("- ", "--", "'", '"'))
        )
        for line in lines
    )
    return structured >= max(2, len(lines) // 2)


def _split_input_shapes(text: str) -> list[tuple[str, str]]:
    """Separate explicit data blocks from surrounding prose without reading intent."""

    explicit: list[tuple[int, int]] = []
    cursor = 0
    decoder = json.JSONDecoder()
    while cursor < len(text):
        line_start = cursor == 0 or text[cursor - 1] == "\n"
        if line_start:
            content_start = cursor + len(text[cursor:]) - len(text[cursor:].lstrip(" \t"))
            if text.startswith("```", content_start):
                closing = text.find("```", content_start + 3)
                end = len(text) if closing < 0 else closing + 3
                explicit.append((content_start, end))
                cursor = end
                continue
            if content_start < len(text) and text[content_start] in "{[":
                try:
                    value, consumed = decoder.raw_decode(text[content_start:])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, (dict, list)):
                        explicit.append((content_start, content_start + consumed))
                        cursor = content_start + consumed
                        continue
        cursor = text.find("\n", cursor) + 1
        if cursor == 0:
            break

    regions: list[tuple[str, str]] = []
    cursor = 0
    for start, end in explicit:
        _append_unstructured_regions(regions, text[cursor:start])
        value = text[start:end].strip()
        if value:
            regions.append((value, "structured"))
        cursor = end
    _append_unstructured_regions(regions, text[cursor:])
    return regions or [(text.strip(), "prose")]


def _append_unstructured_regions(regions: list[tuple[str, str]], text: str) -> None:
    """Preserve multiline curl/env/YAML runs while leaving prose independently addressable."""

    lines = text.strip().splitlines()
    index = 0
    prose: list[str] = []

    def flush_prose() -> None:
        value = "\n".join(prose).strip()
        if value:
            regions.append((value, "prose"))
        prose.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_prose()
            index += 1
            continue
        if stripped.startswith("curl ") or stripped.startswith("curl\\"):
            flush_prose()
            block = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                candidate_value = candidate.strip()
                if not candidate_value:
                    break
                if block[-1].rstrip().endswith("\\") or candidate_value.startswith(("--", "-H ", "-d ")):
                    block.append(candidate)
                    index += 1
                    continue
                break
            regions.append(("\n".join(block).strip(), "structured"))
            continue
        if _line_looks_structured(stripped):
            run = [line]
            lookahead = index + 1
            while lookahead < len(lines) and _line_looks_structured(lines[lookahead].strip()):
                run.append(lines[lookahead])
                lookahead += 1
            if len(run) >= 2:
                flush_prose()
                regions.append(("\n".join(run).strip(), "structured"))
                index = lookahead
                continue
        prose.append(line)
        index += 1
    flush_prose()


def _line_looks_structured(line: str) -> bool:
    return bool(
        re.match(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*=", line)
        or re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*:", line)
        or line.startswith(("- ", "--", "'", '"'))
    )
