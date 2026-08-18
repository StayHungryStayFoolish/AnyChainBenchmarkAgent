"""Fail-closed catalog and renderer for Harness-owned product responses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from string import Formatter
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .contracts import (
    FailureDescriptor,
    ResponseArgument,
    ResponseFragment,
    ResponseFragmentKind,
    TextRef,
)
from .response_messages import MESSAGE_SETS


Language = Literal["en", "zh"]
ArgumentType = Literal["string", "integer", "number", "boolean"]
PayloadKind = Literal["none", "analysis_document", "failure_record"]
_ALLOWED_RENDER_ROLES = frozenset(
    {
        "message",
        "status",
        "evidence",
        "warning",
        "error",
        "question_prompt",
        "option_label",
        "option_description",
        "question_help",
        "completion_effect",
        "question_instruction",
    }
)


class ResponseCatalogError(ValueError):
    """Raised when a response cannot be validated and rendered safely."""


@dataclass(frozen=True)
class MessageDefinition:
    """One centrally owned localized message contract."""

    templates: Mapping[Language, str]
    arguments: Mapping[str, ArgumentType]
    kinds: frozenset[str]
    payload_kind: PayloadKind = "none"


@dataclass(frozen=True)
class RenderedFragment:
    """Rendered text plus independent semantic and presentation identities."""

    text: str
    semantic_hash: str
    render_hash: str
    message_id: str
    kind: ResponseFragmentKind


def _load_catalog() -> dict[str, MessageDefinition]:
    catalog: dict[str, MessageDefinition] = {}
    for messages in MESSAGE_SETS:
        for message_id, raw in messages.items():
            if message_id in catalog:
                raise RuntimeError(f"duplicate response message_id: {message_id}")
            kinds = frozenset(raw.get("kinds") or ())
            unknown_kinds = sorted(kinds - _ALLOWED_RENDER_ROLES)
            if not kinds or unknown_kinds:
                raise RuntimeError(
                    f"invalid render roles for {message_id}: {unknown_kinds or '<empty>'}"
                )
            catalog[message_id] = MessageDefinition(
                templates=MappingProxyType({
                    "en": str(raw.get("en") or ""),
                    "zh": str(raw.get("zh") or ""),
                }),
                arguments=MappingProxyType(dict(raw.get("arguments") or {})),
                kinds=kinds,
                payload_kind=str(raw.get("payload_kind") or "none"),  # type: ignore[arg-type]
            )
    return catalog


_MESSAGE_CATALOG = _load_catalog()
MESSAGE_CATALOG: Mapping[str, MessageDefinition] = MappingProxyType(_MESSAGE_CATALOG)


def render_text_ref(
    reference: TextRef,
    language: str,
    *,
    kind: str,
) -> str:
    """Validate and render one reference; unknown or malformed input fails closed."""

    definition = _validate_reference_contract(reference, language, kind=kind)
    template = definition.templates.get(language)  # type: ignore[arg-type]
    if not template:
        raise ResponseCatalogError(
            f"message_id {reference.message_id!r} has no {language!r} template"
        )
    _validate_template(reference.message_id, template, definition.arguments)
    try:
        rendered = template.format(**dict(reference.arguments)).strip()
    except (KeyError, ValueError) as exc:
        raise ResponseCatalogError(
            f"failed to render message_id {reference.message_id!r}"
        ) from exc
    if not rendered:
        raise ResponseCatalogError(
            f"message_id {reference.message_id!r} rendered empty text"
        )
    return rendered


def render_fragment(fragment: ResponseFragment, language: str) -> RenderedFragment:
    """Render one typed fragment and calculate its two independent hashes."""

    _validate_json_value(fragment.payload, path="payload")
    definition = _validate_reference_contract(
        fragment.text_ref,
        language,
        kind=fragment.kind,
    )
    _validate_payload(fragment.message_id, fragment.payload, definition.payload_kind)
    if definition.payload_kind == "analysis_document":
        text = str(fragment.payload["text"]).strip()
    else:
        text = render_text_ref(fragment.text_ref, language, kind=fragment.kind)
    return RenderedFragment(
        text=text,
        semantic_hash=semantic_hash(
            {
                "kind": fragment.kind,
                "message_id": fragment.message_id,
                "arguments": dict(fragment.arguments),
                "payload": dict(fragment.payload),
            }
        ),
        render_hash=render_hash(text),
        message_id=fragment.message_id,
        kind=fragment.kind,
    )


def _validate_reference_contract(
    reference: TextRef,
    language: str,
    *,
    kind: str,
) -> MessageDefinition:
    definition = MESSAGE_CATALOG.get(reference.message_id)
    if definition is None:
        raise ResponseCatalogError(
            f"unknown response message_id: {reference.message_id}"
        )
    if language not in {"en", "zh"}:
        raise ResponseCatalogError(f"unsupported response language: {language}")
    if kind not in definition.kinds:
        raise ResponseCatalogError(
            f"message_id {reference.message_id!r} cannot be rendered as {kind!r}"
        )
    _validate_arguments(
        reference.message_id,
        reference.arguments,
        definition.arguments,
    )
    return definition


def render_failure(failure: FailureDescriptor, language: str) -> RenderedFragment:
    """Render a typed failure through the same catalog and identity rules."""

    fragment = ResponseFragment(
        kind="error",
        message_id=failure.code,
        arguments=failure.arguments,
        payload={
            **dict(failure.payload),
            "retryable": failure.retryable,
            "severity": failure.severity,
        },
        source=failure.source,
    )
    return render_fragment(fragment, language)


def semantic_hash(value: Any) -> str:
    """Hash canonical semantic data independently from localized rendering."""

    _validate_json_value(value, path="semantic")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_hash(text: str) -> str:
    """Hash exact rendered text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_arguments(
    message_id: str,
    actual: Mapping[str, ResponseArgument],
    expected: Mapping[str, ArgumentType],
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ResponseCatalogError(
            f"invalid arguments for {message_id!r}: missing={missing}, extra={extra}"
        )
    for name, argument_type in expected.items():
        value = actual[name]
        if not _matches_argument_type(value, argument_type):
            raise ResponseCatalogError(
                f"argument {name!r} for {message_id!r} must be {argument_type}"
            )


def _matches_argument_type(value: ResponseArgument, expected: ArgumentType) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    return False


def _validate_template(
    message_id: str,
    template: str,
    expected: Mapping[str, ArgumentType],
) -> None:
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    if fields != set(expected):
        raise ResponseCatalogError(
            f"template fields for {message_id!r} do not match its argument schema"
        )


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResponseCatalogError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResponseCatalogError(f"{path} contains a non-string key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise ResponseCatalogError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _validate_payload(
    message_id: str,
    payload: Mapping[str, Any],
    payload_kind: PayloadKind,
) -> None:
    if payload_kind == "none":
        return
    if payload_kind == "analysis_document":
        expected = {
            "text",
            "source_kind",
            "language",
            "evidence_hash",
            "evidence_paths",
        }
        if set(payload) != expected:
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires exactly "
                "text, source_kind, language, evidence_hash, and evidence_paths"
            )
        if not isinstance(payload.get("text"), str) or not str(payload["text"]).strip():
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires non-empty text"
            )
        if not isinstance(payload.get("source_kind"), str) or not str(
            payload["source_kind"]
        ).strip():
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires source_kind"
            )
        if payload.get("language") not in {"en", "zh"}:
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires supported language"
            )
        evidence_hash = payload.get("evidence_hash")
        if not isinstance(evidence_hash, str) or len(evidence_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in evidence_hash
        ):
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires evidence_hash"
            )
        paths = payload.get("evidence_paths")
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ResponseCatalogError(
                f"analysis document {message_id!r} requires string evidence_paths"
            )
        return
    if payload_kind == "failure_record":
        if not isinstance(payload.get("record"), Mapping):
            raise ResponseCatalogError(
                f"failure record {message_id!r} requires record mapping"
            )
        return
    raise ResponseCatalogError(
        f"unsupported payload kind for {message_id!r}: {payload_kind}"
    )
