"""LLM-backed conversational responder for product terminal turns."""

from __future__ import annotations

from adk_app.instructions import ROOT_INSTRUCTION
from adk_app.runner_bridge import run_text_once
from knowledge.framework_context import render_framework_context_for_prompt
from llm.config import LLMConfig
from llm.providers import provider_from_config
from llm.types import LLMMessage, LLMRequest
from workflows.state import WorkflowState


def should_answer_as_conversation(text: str) -> bool:
    """Return true when text is a user question, not a workflow value."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    greetings = {"你好", "您好", "hello", "hi", "hey", "嗨"}
    if normalized in greetings:
        return True
    question_markers = ("?", "？", "吗", "么", "什么", "如何", "怎么", "why", "what", "how", "who")
    identity_markers = ("你是 ai", "你是ai", "你是什么", "are you ai", "what are you")
    capability_markers = ("支持多少", "支持哪些", "什么链", "rpc method", "capability", "能做什么")
    return (
        any(marker in normalized for marker in identity_markers)
        or any(marker in normalized for marker in capability_markers)
        or any(marker in normalized for marker in question_markers)
    )


def answer_conversation(text: str, state: WorkflowState, config: LLMConfig) -> str:
    """Answer a conversational turn with LLM when available, else fallback."""
    validation_errors = config.validate()
    if validation_errors:
        return _fallback_answer(text, state, "LLM configuration is incomplete: " + "; ".join(validation_errors))
    adk_prompt = "\n\n".join([
        render_framework_context_for_prompt(language=state.language),
        _state_context(state),
        text,
    ])
    try:
        response = run_text_once(adk_prompt, state_delta=_adk_state_delta(state))
        if response.strip():
            return response.strip()
    except Exception as exc:
        adk_error = f"ADK Runner call failed: {type(exc).__name__}: {exc}"
        if config.provider in {"gemini", "claude"}:
            return _fallback_answer(text, state, adk_error)
    # Non-ADK direct adapters remain a compatibility bridge for providers that
    # are not currently executable through the installed ADK runtime.
    try:
        provider = provider_from_config(config)
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_terminal_instruction()),
                    LLMMessage(role="user", content=render_framework_context_for_prompt(language=state.language)),
                    LLMMessage(role="user", content=_state_context(state)),
                    LLMMessage(role="user", content=text),
                ],
                temperature=0.2,
                max_tokens=900,
            )
        )
        if response.text.strip():
            return response.text.strip()
    except Exception as exc:
        return _fallback_answer(text, state, f"LLM call failed: {exc}")
    return _fallback_answer(text, state, "LLM returned an empty response")


def _terminal_instruction() -> str:
    return (
        ROOT_INSTRUCTION
        + "\n\nYou are answering inside the AnyChain product terminal. "
        + "If the user asks a general or identity question, answer directly. "
        + "Do not treat general questions as configuration values. "
        + "If a benchmark workflow is in progress, briefly mention that it can continue after the answer."
    )


def _state_context(state: WorkflowState) -> str:
    return (
        "Current terminal workflow state:\n"
        f"- language={state.language}\n"
        f"- intent={state.intent or '<none>'}\n"
        f"- stage={state.stage}\n"
        f"- current_question_id={state.current_question_id or '<none>'}\n"
        f"- confirmed_keys={sorted(state.confirmed_values.keys())}\n"
    )


def _adk_state_delta(state: WorkflowState) -> dict[str, object]:
    return {
        "language": state.language,
        "intent": state.intent,
        "stage": state.stage,
        "current_question_id": state.current_question_id,
        "confirmed_values": dict(state.confirmed_values),
        "job_id": state.job_id,
    }


def _fallback_answer(text: str, state: WorkflowState, reason: str) -> str:
    is_zh = state.language == "zh"
    if is_zh:
        base = (
            "我是 AnyChain Benchmark Agent，用来帮助你检查环境、生成压测计划、执行 preflight、"
            "运行 fake-node/真实节点 benchmark，并基于产物分析结果。"
        )
        if state.current_question_id:
            base += f" 当前还有一个未完成的配置项：{state.current_question_id}。"
        return base + f" 但这次没有调用到底层 LLM：{reason}"
    base = (
        "I am AnyChain Benchmark Agent. I help inspect the host, generate benchmark plans, "
        "run preflight, execute fake-node or real-node benchmarks, and analyze artifacts."
    )
    if state.current_question_id:
        base += f" There is an unfinished configuration question: {state.current_question_id}."
    return base + f" I did not use the underlying LLM for this turn: {reason}"
