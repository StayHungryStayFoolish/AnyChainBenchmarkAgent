"""Typed Chain/RPC response values returned through ``HandlerResult`` only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from ..contracts import (
    FailureDescriptor,
    ResponseArgument,
    ResponseFragment,
    ResponseFragmentKind,
)


ResponseCollector: TypeAlias = list[ResponseFragment]


def emit(
    collector: ResponseCollector,
    message_id: str,
    *,
    arguments: Mapping[str, ResponseArgument] | None = None,
    payload: Mapping[str, Any] | None = None,
    source: str,
    kind: ResponseFragmentKind = "message",
) -> None:
    """Append one language-independent response fact to the current handler."""

    collector.append(
        ResponseFragment(
            kind=kind,
            message_id=message_id,
            arguments=dict(arguments or {}),
            payload=dict(payload or {}),
            source=source,
        )
    )


def failure(
    code: str,
    *,
    arguments: Mapping[str, ResponseArgument] | None = None,
    payload: Mapping[str, Any] | None = None,
    source: str,
    retryable: bool = False,
    severity: str = "blocking",
) -> FailureDescriptor:
    """Build one typed Chain/RPC failure without embedding product prose."""

    return FailureDescriptor(
        code=code,
        arguments=dict(arguments or {}),
        payload=dict(payload or {}),
        source=source,
        retryable=retryable,
        severity=severity,  # type: ignore[arg-type]
    )
