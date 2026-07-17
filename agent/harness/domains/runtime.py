"""Runtime capabilities for each workflow-domain owner.

The pure ``GroupSpec`` registry declares ownership.  This module binds each
owner to its runtime functions once, so the coordinator never maintains
parallel action, answer, question, and cancellation maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..contracts import ActionProposal, HandlerResult
from ..state import AgentGraphState, PendingQuestion
from .analysis import apply_analysis_action
from .chain_rpc import (
    apply_chain_rpc_action,
    apply_chain_rpc_answer,
    cancel_chain_rpc_question,
    question_for_chain_rpc,
)
from .environment import apply_environment_action, apply_environment_answer, question_for_environment
from .execution import apply_execution_action, apply_execution_answer, question_for_execution
from .orientation import apply_orientation_action, apply_orientation_answer, opening_question
from .performance import apply_performance_action, apply_performance_answer, question_for_performance
from .recovery import apply_recovery_action, question_for_recovery
from .sync_observe import apply_sync_observe_action, apply_sync_observe_answer, question_for_sync_observe


ActionHandler = Callable[[AgentGraphState, ActionProposal], HandlerResult]
AnswerHandler = Callable[[AgentGraphState, PendingQuestion, Any, str], HandlerResult]
QuestionFactory = Callable[[AgentGraphState, str], PendingQuestion | None]
QuestionCanceller = Callable[[AgentGraphState, PendingQuestion], HandlerResult]


@dataclass(frozen=True)
class DomainRuntime:
    apply_action: ActionHandler
    question_factory: QuestionFactory | None = None
    apply_answer: AnswerHandler | None = None
    cancel_question: QuestionCanceller | None = None


DOMAIN_RUNTIME: dict[str, DomainRuntime] = {
    "analysis": DomainRuntime(apply_action=apply_analysis_action),
    "chain_rpc": DomainRuntime(
        apply_action=apply_chain_rpc_action,
        question_factory=question_for_chain_rpc,
        apply_answer=apply_chain_rpc_answer,
        cancel_question=cancel_chain_rpc_question,
    ),
    "environment": DomainRuntime(
        apply_action=apply_environment_action,
        question_factory=question_for_environment,
        apply_answer=lambda state, question, value, _text: apply_environment_answer(state, question, value),
    ),
    "execution": DomainRuntime(
        apply_action=apply_execution_action,
        question_factory=question_for_execution,
        apply_answer=lambda state, question, value, _text: apply_execution_answer(state, value, question),
    ),
    "orientation": DomainRuntime(
        apply_action=apply_orientation_action,
        question_factory=lambda state, _group: opening_question(state),
        apply_answer=apply_orientation_answer,
    ),
    "performance": DomainRuntime(
        apply_action=apply_performance_action,
        question_factory=question_for_performance,
        apply_answer=lambda state, question, value, _text: apply_performance_answer(state, question, value),
    ),
    "recovery": DomainRuntime(
        apply_action=apply_recovery_action,
        question_factory=question_for_recovery,
    ),
    "sync_observe": DomainRuntime(
        apply_action=apply_sync_observe_action,
        question_factory=lambda state, _group: question_for_sync_observe(state),
        apply_answer=lambda state, question, value, _text: apply_sync_observe_answer(state, value, question),
    ),
}
