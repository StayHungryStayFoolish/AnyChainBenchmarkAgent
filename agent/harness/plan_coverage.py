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
    unresolved_units: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SemanticPartitionResult:
    valid: bool
    units: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]


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


def validate_semantic_partition(
    raw_units: Sequence[Any],
    clauses: Sequence[TurnClause],
) -> SemanticPartitionResult:
    """Validate and canonicalize a lossless Stage A source partition."""

    expected = {clause.clause_id: clause for clause in clauses}
    canonical, errors = _canonicalize_semantic_units(raw_units, expected)
    seen_ids: set[str] = set()
    seen_clauses: set[str] = set()
    last_clause_index = -1
    clause_indexes = {
        clause.clause_id: index for index, clause in enumerate(clauses)
    }
    output: list[dict[str, Any]] = []
    for raw in canonical:
        if not isinstance(raw, Mapping):
            errors.append("semantic partition contains a non-object unit")
            continue
        unit = dict(raw)
        unit_id = str(unit.get("unit_id") or "").strip()
        clause_id = str(unit.get("clause_id") or "").strip()
        if not unit_id:
            errors.append("semantic partition unit has no unit_id")
        elif unit_id in seen_ids:
            errors.append(f"duplicate semantic partition unit id: {unit_id}")
        else:
            seen_ids.add(unit_id)
        if clause_id not in expected:
            errors.append(f"semantic partition references unknown clause: {clause_id}")
        else:
            seen_clauses.add(clause_id)
            clause_index = clause_indexes[clause_id]
            if clause_index < last_clause_index:
                errors.append(
                    f"semantic partition reorders clauses at unit: {unit_id}"
                )
            last_clause_index = max(last_clause_index, clause_index)
        if not str(unit.get("source_text") or ""):
            errors.append(f"semantic partition unit has empty source_text: {unit_id}")
        output.append(unit)
    for clause_id in expected:
        if clause_id not in seen_clauses:
            errors.append(f"semantic partition omits clause: {clause_id}")
    return SemanticPartitionResult(
        valid=not errors,
        units=tuple(output),
        errors=tuple(dict.fromkeys(errors)),
    )


def validate_plan_coverage(
    payload: Mapping[str, Any],
    clauses: Sequence[TurnClause],
    *,
    pending_answer_action_indexes: Sequence[int] = (),
) -> PlanCoverageResult:
    """Validate complete clause-to-action accounting for one model plan."""

    actions = payload.get("actions")
    action_list = list(actions) if isinstance(actions, list) else []
    expected = {clause.clause_id: clause for clause in clauses}
    raw_units = payload.get("semantic_units")
    if not isinstance(raw_units, list):
        raw_units = []
    raw_units, span_errors = _canonicalize_semantic_units(raw_units, expected)
    pending_support_unit_ids = {
        str(unit_id)
        for unit_id in payload.get("pending_support_unit_ids", [])
        if str(unit_id)
    }
    units_by_clause: dict[str, list[Mapping[str, Any]]] = {
        clause_id: [] for clause_id in expected
    }
    seen_unit_ids: set[str] = set()
    errors: list[str] = list(span_errors)
    errors.extend(_entry_intake_exclusivity_errors(action_list))
    unresolved: list[str] = []
    unresolved_units: list[dict[str, Any]] = []
    referenced_actions: set[int] = set()
    incomplete_unit_ids: set[str] = set()

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
            unresolved_units.append({
                "unit_id": unit_id,
                "parent_unit_id": str(raw.get("parent_unit_id") or ""),
                "clause_id": clause_id,
                "start": start,
                "end": end,
                "source_text": source_text,
                "source_path": str(raw.get("source_path") or ""),
                "owner_routes": [
                    dict(route)
                    for route in raw.get("owner_routes") or ()
                    if isinstance(route, Mapping)
                ],
                "reason": str(raw.get("reason") or ""),
            })
            continue
        if disposition == "context":
            if (
                expected[clause_id].input_shape != "prose"
                and unit_id not in pending_support_unit_ids
            ):
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
        if scope_constraint and not indexes:
            indexes = _derive_scope_indexes(unit_id, scope_constraint, action_list, errors)
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
            scope_error_count = len(errors)
            _validate_semantic_scope(unit_id, scope_constraint, mapped_actions, errors)
            if len(errors) != scope_error_count:
                incomplete_unit_ids.add(unit_id)
        _validate_literal_anchors(
            unit_id,
            _structured_atom_source(raw, source_text),
            mapped_actions,
            errors,
            input_shape=expected[clause_id].input_shape,
            allow_pending_answer=any(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in pending_answer_action_indexes
                for value in indexes
            ),
            source_path=str(raw.get("source_path") or ""),
        )

    for clause_id, clause in expected.items():
        units = sorted(units_by_clause[clause_id], key=lambda item: int(item["start"]))
        if not units:
            errors.append(f"missing semantic units for {clause_id}")
            unresolved.append(clause.text)
            continue
        coverage_units: list[Mapping[str, Any]] = []
        coverage_by_identity: dict[str, Mapping[str, Any]] = {}
        for unit in units:
            identity = str(
                unit.get("parent_unit_id")
                or unit.get("unit_id")
                or ""
            )
            prior = coverage_by_identity.get(identity)
            if prior is None:
                coverage_by_identity[identity] = unit
                coverage_units.append(unit)
                continue
            if (
                int(prior["start"]) != int(unit["start"])
                or int(prior["end"]) != int(unit["end"])
                or str(prior.get("source_text") or "")
                != str(unit.get("source_text") or "")
            ):
                errors.append(
                    "semantic DemandAtoms sharing a parent have different "
                    f"source spans: {identity}"
                )
        coverage_units.sort(key=lambda item: int(item["start"]))
        structured_paths = [
            str(unit.get("source_path") or "").strip()
            for unit in coverage_units
        ]
        if (
            clause.input_shape == "structured"
            and all(structured_paths)
            and len(set(structured_paths)) == len(structured_paths)
            and all(
                int(unit["start"]) == 0
                and int(unit["end"]) == len(clause.text)
                for unit in coverage_units
            )
        ):
            continue
        cursor = 0
        for unit in coverage_units:
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
    errors.extend(
        f"pending support receipt references an unknown semantic unit: {unit_id}"
        for unit_id in sorted(pending_support_unit_ids - seen_unit_ids)
    )
    return PlanCoverageResult(
        valid=not errors and not unresolved,
        errors=tuple(errors),
        unresolved_clauses=tuple(dict.fromkeys(unresolved)),
        incomplete_unit_ids=tuple(sorted(incomplete_unit_ids)),
        unresolved_units=tuple(unresolved_units),
    )


def _entry_intake_exclusivity_errors(actions: Sequence[Any]) -> list[str]:
    """Reject generic navigation that competes with a typed group entry.

    Entry ownership is registry metadata. Once a transaction contains the
    registered typed intake for a group, ``change_group`` cannot also claim
    that destination; the typed intake owns interruption and group activation.
    """

    from .action_registry import ACTION_BY_TYPE, resolve_action_target_group

    typed_entry_groups: set[str] = set()
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.entry_intake:
            continue
        if any(
            action.get(key) != value
            for key, value in spec.entry_intake_fixed_arguments
        ):
            continue
        if (
            spec.entry_intake_value_arguments
            and not any(bool(action.get(key)) for key in spec.entry_intake_value_arguments)
        ):
            continue
        target = resolve_action_target_group(action)
        if target:
            typed_entry_groups.add(target)

    return [
        (
            f"generic navigation competes with registered typed entry: {target}; "
            "remove change_group and retain the registered typed entry action"
        )
        for action in actions
        if isinstance(action, Mapping)
        and str(action.get("type") or "") == "change_group"
        and (target := str(action.get("group") or "").strip()) in typed_entry_groups
    ]


def _canonicalize_semantic_units(
    raw_units: Sequence[Any],
    expected: Mapping[str, TurnClause],
) -> tuple[list[Any], list[str]]:
    """Derive exact source spans from ordered model-authored text anchors.

    Character offsets are control metadata, not semantic model output. The
    model identifies units with exact source excerpts; the Harness resolves a
    unique ordered placement and expands prose units into a lossless partition.
    This ensures conjunctions and other unclaimed prose remain visible to the
    independent whole-plan reviewer instead of forcing a second compiler call.
    Structured input remains atomic, and ambiguous placement remains invalid.
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
            source_paths = [
                str(unit.get("source_path") or "").strip()
                for unit in units
            ]
            if (
                all(anchor == clause.text for anchor in anchors)
                and all(source_paths)
                and len(set(source_paths)) == len(source_paths)
            ):
                for raw_index in indexes:
                    normalized = dict(raw_units[raw_index])
                    normalized.update({
                        "start": 0,
                        "end": len(clause.text),
                        "source_text": clause.text,
                    })
                    canonical[raw_index] = normalized
                continue
            if len(units) > 1 and all(anchor == clause.text for anchor in anchors):
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
            unit_spans = [
                (
                    0 if unit_offset == 0 else placements[unit_offset][0],
                    (
                        placements[unit_offset + 1][0]
                        if unit_offset + 1 < len(placements)
                        else len(clause.text)
                    ),
                )
                for unit_offset in range(len(units))
            ]
        else:
            unique_anchors: list[str] = []
            anchor_group_indexes: list[int] = []
            for anchor in anchors:
                if not unique_anchors or anchor != unique_anchors[-1]:
                    unique_anchors.append(anchor)
                anchor_group_indexes.append(len(unique_anchors) - 1)
            placement_sets = _unique_anchor_placements(
                clause.text,
                unique_anchors,
                allow_unclaimed_intervals=True,
            )
            if not placement_sets:
                errors.append(f"source anchors do not cover {clause_id} without omitted prose")
                continue
            if len(placement_sets) != 1:
                errors.append(f"source anchors are ambiguous in {clause_id}")
                continue
            placements = placement_sets[0]
            group_spans = [
                (
                    0 if group_index == 0 else placements[group_index][0],
                    (
                        placements[group_index + 1][0]
                        if group_index + 1 < len(placements)
                        else len(clause.text)
                    ),
                )
                for group_index in range(len(unique_anchors))
            ]
            unit_spans = [
                group_spans[group_index]
                for group_index in anchor_group_indexes
            ]

        for unit_offset, raw_index in enumerate(indexes):
            start, end = unit_spans[unit_offset]
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
    *,
    allow_unclaimed_intervals: bool = False,
) -> list[list[tuple[int, int]]]:
    solutions: list[list[tuple[int, int]]] = []

    def visit(index: int, cursor: int, placements: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if index == len(anchors):
            if allow_unclaimed_intervals or _separator_only(text[cursor:]):
                solutions.append(list(placements))
            return
        anchor = anchors[index]
        position = text.find(anchor, cursor)
        while position >= 0:
            if allow_unclaimed_intervals or _separator_only(text[cursor:position]):
                end = position + len(anchor)
                placements.append((position, end))
                visit(index + 1, end, placements)
                placements.pop()
            position = text.find(anchor, position + 1)

    visit(0, 0, [])
    return solutions[:2]


def _separator_only(value: str) -> bool:
    return all(character.isspace() or unicodedata.category(character)[0] in {"P", "S"} for character in value)


def _validate_semantic_scope(
    unit_id: str,
    scope_constraint: str,
    mapped_actions: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    from .action_registry import (
        SEMANTIC_SCOPE_POLICIES,
        action_effect,
        semantic_scope_accepts_action,
    )

    if scope_constraint not in SEMANTIC_SCOPE_POLICIES:
        errors.append(f"invalid scope constraint for {unit_id}: {scope_constraint}")
        return
    if not mapped_actions:
        errors.append(f"{scope_constraint} scope has no mapped actions: {unit_id}")
        return
    for action in mapped_actions:
        if not semantic_scope_accepts_action(scope_constraint, action):
            errors.append(
                f"{scope_constraint} scope rejects {action_effect(action)} action for {unit_id}: "
                f"{action.get('type') or '<missing>'}"
            )


def _derive_scope_indexes(
    unit_id: str,
    scope_constraint: str,
    actions: Sequence[Any],
    errors: list[str],
) -> list[int]:
    from .action_registry import SEMANTIC_SCOPE_POLICIES, semantic_scope_accepts_action

    if scope_constraint not in SEMANTIC_SCOPE_POLICIES:
        errors.append(f"invalid scope constraint for {unit_id}: {scope_constraint}")
        return []
    if not actions:
        errors.append(f"{scope_constraint} scope has no actions to bind: {unit_id}")
        return []
    indexes: list[int] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or not semantic_scope_accepts_action(scope_constraint, action):
            errors.append(
                f"{scope_constraint} scope cannot bind action for {unit_id}: "
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
    *,
    input_shape: str,
    allow_pending_answer: bool,
    source_path: str = "",
) -> None:
    """Protect exact facts in atomic data without interpreting prose roles.

    Whether a literal in prose is selected, rejected, illustrative, or merely
    contextual is a semantic admission decision. Structured request/evidence
    blocks have no such conversational role ambiguity and remain fail-closed
    here before any action can be admitted.
    """

    if input_shape != "structured":
        return
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
        if _mapped_action_preserves_structured_intake(
            method,
            source_path,
            actions,
        ):
            continue
        if not _mapped_action_preserves_wire_method(
            method,
            actions,
            allow_pending_answer=allow_pending_answer,
        ):
            errors.append(
                f"wire method {method!r} from {unit_id} is absent from its mapped actions"
            )


def _structured_atom_source(
    unit: Mapping[str, Any],
    source_text: str,
) -> str:
    """Project one structured DemandAtom without losing its SourceClause."""

    source_path = str(unit.get("source_path") or "").strip()
    if not source_path:
        return source_text
    from .domains.environment import extract_structured_input_candidates

    candidates = extract_structured_input_candidates(source_text) or {}
    matching = [
        candidate
        for candidate in candidates.get("field_candidates") or []
        if isinstance(candidate, Mapping)
        and str(candidate.get("source_path") or "") == source_path
    ]
    if len(matching) != 1:
        return source_text
    return json.dumps(
        {source_path: matching[0].get("raw_value")},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _mapped_action_preserves_structured_intake(
    token: str,
    source_path: str,
    actions: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize a registry-declared structured action alias as control syntax."""

    leaf = str(source_path or "").rsplit(".", 1)[-1].casefold()
    if token.casefold() != leaf:
        return False
    from .action_registry import ACTION_BY_TYPE

    for action in actions:
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None:
            continue
        for intake in spec.structured_intake:
            if intake.alias.casefold() != leaf:
                continue
            if all(action.get(key) == value for key, value in intake.fixed_arguments):
                return True
    return False


def _mapped_action_preserves_wire_method(
    method: str,
    actions: Sequence[Mapping[str, Any]],
    *,
    allow_pending_answer: bool,
) -> bool:
    """Require an exact method fact in its owning action or evidence payload."""

    for action in actions:
        if str(action.get("rpc_method") or "").strip() == method:
            return True
        for key in (
            "rpc_schema_evidence",
            "handoff_evidence",
            "evidence",
            "subject",
        ):
            value = action.get(key)
            if isinstance(value, str) and method in value:
                return True
        if (
            allow_pending_answer
            and str(action.get("type") or "") == "answer_pending"
            and isinstance(action.get("answer"), str)
            and method in str(action.get("answer"))
        ):
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
            regions.append(("\n".join(block), "structured"))
            continue
        if _line_looks_structured(stripped):
            run = [line]
            lookahead = index + 1
            while lookahead < len(lines) and _line_looks_structured(lines[lookahead].strip()):
                run.append(lines[lookahead])
                lookahead += 1
            # Input shape describes transport syntax, not user intent. A
            # single YAML/env assignment is still structured input and must
            # reach the same proposal/review boundary as a larger pasted
            # block. The semantic planner remains responsible for deciding
            # whether an error label, example, or documentation fragment is
            # configuration at all.
            flush_prose()
            regions.append(("\n".join(run), "structured"))
            index = lookahead
            continue
        prose.append(line)
        index += 1
    flush_prose()


def _line_looks_structured(line: str) -> bool:
    # A standalone endpoint is prose input, not a YAML ``scheme: value`` pair.
    # URL semantics are decided by the planner; this boundary only identifies
    # transport syntax.
    if _URL_RE.fullmatch(line):
        return False
    return bool(
        re.match(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*=", line)
        or re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*:", line)
        or line.startswith(("- ", "--", "'", '"'))
    )
