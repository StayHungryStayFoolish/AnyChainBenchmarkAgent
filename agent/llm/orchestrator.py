"""Prompt bundle orchestration for Agent workflows."""

from __future__ import annotations

from dataclasses import dataclass

from llm.prompt_loader import compose_prompt
from llm.types import LLMMessage, LLMProvider, LLMRequest


PROMPT_BUNDLES: dict[str, tuple[str, ...]] = {
    "intent": ("system", "safety_guardrails", "intent_router", "kb_grounding"),
    "request": ("system", "safety_guardrails", "config_interviewer", "request_drafter", "workload_designer"),
    "artifact_analysis": ("system", "safety_guardrails", "result_analyzer", "chart_diagnostics"),
    "onboarding": ("system", "safety_guardrails", "chain_onboarding", "workload_designer", "kb_grounding"),
    "kb_answer": ("system", "safety_guardrails", "kb_grounding"),
}


@dataclass(frozen=True)
class PromptOrchestrator:
    provider: LLMProvider | None = None

    def system_prompt(self, workflow: str) -> str:
        return compose_prompt(*bundle_for_workflow(workflow))

    def synthesize(self, workflow: str, user_content: str, *, max_tokens: int = 1200) -> str:
        if self.provider is None:
            return ""
        response = self.provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=self.system_prompt(workflow)),
                    LLMMessage(role="user", content=user_content),
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
        )
        return response.text.strip()


def bundle_for_workflow(workflow: str) -> tuple[str, ...]:
    if workflow not in PROMPT_BUNDLES:
        raise KeyError(f"unknown prompt workflow: {workflow}")
    return PROMPT_BUNDLES[workflow]


def prompt_coverage() -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for workflow, prompts in PROMPT_BUNDLES.items():
        for prompt in prompts:
            coverage.setdefault(prompt, []).append(workflow)
    return coverage


def synthesize_with_fallback(
    provider: LLMProvider | None,
    workflow: str,
    user_content: str,
    fallback: str,
    *,
    max_tokens: int = 1200,
) -> str:
    if provider is None:
        return fallback
    try:
        text = PromptOrchestrator(provider).synthesize(workflow, user_content, max_tokens=max_tokens)
    except Exception:
        return fallback
    return text or fallback
