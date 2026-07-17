"""Single ownership map for all product workflow groups."""

from __future__ import annotations

from typing import Protocol

from ..contracts import ActionProposal, HandlerResult
from ..state import AgentGraphState, DEFAULT_GROUP_ORDER

from agent.workflows.group_registry import GROUPS, GROUP_OWNER
DOMAIN_GROUPS: dict[str, tuple[str, ...]] = {}
for spec in GROUPS:
    DOMAIN_GROUPS.setdefault(spec.owner, ())
    DOMAIN_GROUPS[spec.owner] = (*DOMAIN_GROUPS[spec.owner], spec.name)


class DomainHandler(Protocol):
    name: str
    groups: tuple[str, ...]

    def apply(self, state: AgentGraphState, action: ActionProposal) -> HandlerResult:
        """Validate and apply one action owned by this domain."""


def validate_domain_ownership() -> None:
    declared = [group for groups in DOMAIN_GROUPS.values() for group in groups]
    duplicates = sorted({group for group in declared if declared.count(group) > 1})
    expected = set(DEFAULT_GROUP_ORDER)
    actual = set(declared)
    if duplicates or actual != expected:
        raise RuntimeError(
            "invalid Harness group ownership: "
            f"duplicates={duplicates}, missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


validate_domain_ownership()
