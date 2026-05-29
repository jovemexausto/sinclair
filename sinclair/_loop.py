"""
_loop.py — LangGraph graph builder.

Three invariants enforced here:

  1. final_answer is the ONLY exit when response_schema is set.
     If the model replies in plain text without calling final_answer,
     the loop sends feedback and continues.

  2. Schema reaches the LLM as a native tool definition (args_schema),
     not as a prompt instruction. tool_choice forces the call.

  3. Validation feedback is surgical — short, field-specific, model-readable.
     No stack traces. No raw JSON.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from ._config import AgentConfig
from ._errors import FinalAnswerError
from ._metrics import IterMetrics, _cost
from ._obs import emit
from ._output import (
    _format_validation_error,
    make_final_answer_tool,
    make_simple_reply_tool,
)
from ._state import AgentState


_FINALIZATION_TAIL_TURNS = 8


# ── graph builder ─────────────────────────────────────────────────────────────


def build_graph(
    tools: Sequence[BaseTool],
    config: AgentConfig,
    result_holder: dict[str, Any],
) -> Any:
    """
    Compile and return a LangGraph StateGraph.

    result_holder is a mutable dict shared with make_final_answer_tool.
    When final_answer succeeds, result_holder["reply"] is set and the
    loop terminates.
    """
    from langchain_openai import ChatOpenAI

    observers = config.observers

    # ── LLM setup ─────────────────────────────────────────────────────────────
    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "use_responses_api": True,
    }
    if config.reasoning_level is not None:
        llm_kwargs["reasoning"] = {"effort": config.reasoning_level}

    llm: BaseChatModel = config.llm or ChatOpenAI(**llm_kwargs)

    # ── tools setup ───────────────────────────────────────────────────────────
    all_tools = list(tools)
    completion_tool_names: set[str] = set()

    if config.response_schema is not None:
        fa_tool, _ = make_final_answer_tool(
            config.response_schema,
            config.response_validator,
            result_holder=result_holder,
        )
        all_tools.append(fa_tool)
        if config.allow_simple_reply:
            simple_reply_tool, _ = make_simple_reply_tool(
                result_holder=result_holder,
            )
            all_tools.append(simple_reply_tool)
            completion_tool_names.add(simple_reply_tool.name)

    tool_map = {t.name: t for t in all_tools}
    analysis_tools = [
        t
        for t in all_tools
        if t.name
        not in {
            "final_answer",
            *completion_tool_names,
        }
    ]
    drafting_tools = [t for t in all_tools if t.name != "final_answer"]
    publication_tools = list(all_tools)

    # LLM bound to all tools
    llm_with_analysis_tools = (
        llm.bind_tools(analysis_tools) if analysis_tools else llm
    )
    llm_with_drafting_tools = (
        llm.bind_tools(drafting_tools) if drafting_tools else llm
    )
    llm_with_publication_tools = (
        llm.bind_tools(publication_tools) if publication_tools else llm
    )

    model_name = (
        config.model
        if config.llm is None
        else getattr(config.llm, "model_name", "")
    )

    def _preview(value: Any, limit: int = 240) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _observer_result_payload(content: str) -> Any:
        try:
            return json.loads(content)
        except Exception:
            return content

    def _extract_tool_intent(args: Any) -> str | None:
        if isinstance(args, dict):
            intent = args.get("intent")
            if isinstance(intent, str) and intent.strip():
                return intent.strip()
        return None

    # ── nodes ──────────────────────────────────────────────────────────────────

    def call_llm(state: AgentState) -> dict:
        iteration = state["iteration"] + 1
        messages = state["messages"]
        finalization_started = state.get("finalization_started", False)
        finalization_mode = (
            config.response_schema is not None
            and iteration >= config.max_iterations
        )
        publication_ready = state.get(
            "publishable_data_prepared", False
        ) or not state.get("metadata", {}).get("require_publishable_data")
        can_publish = publication_ready
        in_warning_window = (
            config.response_schema is not None
            and not finalization_mode
            and config.finalization_window > 0
            and iteration
            >= max(1, config.max_iterations - config.finalization_window)
        )

        active_messages = list(messages)
        if in_warning_window:
            emit(
                observers,
                "budget_warning",
                {
                    "iteration": iteration,
                    "max_iterations": config.max_iterations,
                },
            )
            active_messages.append(
                HumanMessage(
                    content=(
                        "You are approaching the safety limit. Before calling `final_answer`, make sure the hypothesis is settled, chart values are final, visible percentages already have `[ct:<citation_id>]` markers, and citations point to the right target. If something is still missing, resolve only that gap."
                        if publication_ready
                        else "You are approaching the safety limit. Get the final chart numbers first with `get_final_chart_numbers`, then submit."
                    )
                )
            )

        if finalization_mode:
            if not finalization_started:
                emit(
                    observers,
                    "finalization_started",
                    {
                        "iteration": iteration,
                        "max_iterations": config.max_iterations,
                    },
                )
            active_messages.append(
                HumanMessage(
                    content=(
                        "You are now past the safety limit. A valid `final_answer` is required. Fix only the missing field or validation error, keep correct work unchanged, and resubmit once markdown, charts, and citations agree."
                        if publication_ready
                        else "You are now past the safety limit. Get the final chart numbers with `get_final_chart_numbers`, then submit."
                    )
                )
            )

        if can_publish:
            active_llm = llm_with_publication_tools
        elif publication_ready:
            active_llm = llm_with_drafting_tools
        else:
            active_llm = llm_with_analysis_tools

        t0 = time.perf_counter()
        response: AIMessage = active_llm.invoke(active_messages)
        latency = time.perf_counter() - t0

        # Extract token usage from response metadata
        usage = getattr(response, "usage_metadata", None) or {}
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)
        cost = _cost(model_name, input_tok, output_tok)

        tool_calls = [tc["name"] for tc in (response.tool_calls or [])]

        metrics = IterMetrics(
            iteration=iteration,
            latency_s=latency,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            tool_calls=tool_calls,
        )

        emit(
            observers,
            "llm_call",
            {
                "iteration": iteration,
                "mode": "finalization" if finalization_mode else "normal",
                "latency_s": latency,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cost_usd": cost,
                "tool_calls": tool_calls,
            },
        )

        return {
            "messages": [response],
            "iteration": iteration,
            "iter_metrics": state["iter_metrics"] + [metrics],
            "finalization_started": finalization_started or finalization_mode,
            "finalization_turns": state.get("finalization_turns", 0)
            + (1 if finalization_mode else 0),
        }

    def execute_tools(state: AgentState) -> dict:
        last_message = cast(AIMessage, state["messages"][-1])
        tool_calls = last_message.tool_calls or []
        iteration = state["iteration"]

        results: list[ToolMessage] = []
        final_accepted = False
        last_validation_error = state.get("last_validation_error")
        publishable_data_prepared = state.get(
            "publishable_data_prepared", False
        )

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            emit(
                observers,
                "tool_start",
                {
                    "tool": tool_name,
                    "tools": [tool_name],
                    "args": tool_args,
                    "intent": _extract_tool_intent(tool_args),
                    "iteration": iteration,
                },
            )

            tool = tool_map.get(tool_name)
            if tool is None:
                content = f"Tool '{tool_name}' not found."
                emit(
                    observers,
                    "tool_error",
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "intent": _extract_tool_intent(tool_args),
                        "error": content,
                        "attempt": 1,
                        "iteration": iteration,
                    },
                )
                results.append(
                    ToolMessage(content=content, tool_call_id=tool_id)
                )
                continue

            if (
                tool_name == "final_answer"
                and state.get("metadata", {}).get("require_publishable_data")
                and not publishable_data_prepared
            ):
                content = "final_answer rejected — freeze publishable chart data with get_final_chart_numbers before publishing this report."
                last_validation_error = content
                emit(
                    observers,
                    "final_answer_rejected",
                    {"error": content, "iteration": iteration},
                )
                emit(
                    observers,
                    "tool_end",
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "intent": _extract_tool_intent(tool_args),
                        "result": _observer_result_payload(content),
                        "result_preview": _preview(content),
                        "status": "rejected",
                        "iteration": iteration,
                    },
                )
                results.append(
                    ToolMessage(content=content, tool_call_id=tool_id)
                )
                continue

            # Retry logic
            last_exc: Exception | None = None
            wait = config.tool_retry_backoff
            content = ""
            for attempt in range(config.tool_retries + 1):
                try:
                    content = str(tool.invoke(tool_args))
                    last_exc = None
                    break
                except ValidationError as exc:
                    if tool_name == "final_answer":
                        content = _format_validation_error(exc)
                        last_exc = None
                        break
                    last_exc = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < config.tool_retries:
                        emit(
                            observers,
                            "tool_retry",
                            {
                                "tool": tool_name,
                                "args": tool_args,
                                "intent": _extract_tool_intent(tool_args),
                                "attempt": attempt + 1,
                                "wait_s": wait,
                                "error": str(exc),
                                "iteration": iteration,
                            },
                        )
                        time.sleep(wait)
                        wait *= 2

            if last_exc is not None:
                content = f"Tool '{tool_name}' failed after {config.tool_retries + 1} attempts: {last_exc}"
                emit(
                    observers,
                    "tool_error",
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "intent": _extract_tool_intent(tool_args),
                        "error": str(last_exc),
                        "attempt": config.tool_retries + 1,
                        "iteration": iteration,
                    },
                )

            if (
                tool_name == "final_answer"
                and content == "__final_answer_accepted__"
            ):
                final_accepted = True
                last_validation_error = None
                content = "Final answer accepted."
                emit(
                    observers,
                    "final_answer_accepted",
                    {"iteration": iteration},
                )
            elif (
                tool_name == "final_answer"
                and content != "__final_answer_accepted__"
            ):
                last_validation_error = content
                emit(
                    observers,
                    "final_answer_rejected",
                    {"error": content, "iteration": iteration},
                )
            elif (
                tool_name in completion_tool_names
                and content == "__completion_accepted__"
            ):
                final_accepted = True
                last_validation_error = None
                content = "Completion accepted."
            elif tool_name == "get_final_chart_numbers" and last_exc is None:
                publishable_data_prepared = True

            emit(
                observers,
                "tool_end",
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "intent": _extract_tool_intent(tool_args),
                    "result": _observer_result_payload(content),
                    "result_preview": _preview(content),
                    "status": "accepted"
                    if final_accepted
                    and tool_name in {"final_answer", *completion_tool_names}
                    else ("rejected" if tool_name == "final_answer" else "ok"),
                    "iteration": iteration,
                },
            )

            results.append(ToolMessage(content=content, tool_call_id=tool_id))

        return {
            "messages": results,
            "final_answer_accepted": final_accepted,
            "last_validation_error": last_validation_error,
            "publishable_data_prepared": publishable_data_prepared,
        }

    def after_tools(state: AgentState) -> str:
        if state.get("final_answer_accepted"):
            return "end"
        return "llm"

    def should_continue(state: AgentState) -> str:
        if state.get("final_answer_accepted"):
            return "end"

        last_message = cast(AIMessage, state["messages"][-1])
        iteration = state["iteration"]

        if iteration >= config.max_iterations:
            if config.response_schema is None:
                emit(
                    observers,
                    "max_iterations_reached",
                    {"iteration": iteration},
                )
                return "end"

            if state.get("finalization_turns", 0) >= _FINALIZATION_TAIL_TURNS:
                error = (
                    state.get("last_validation_error")
                    or "no accepted final_answer"
                )
                emit(
                    observers,
                    "finalization_failed",
                    {"iteration": iteration, "error": error},
                )
                raise FinalAnswerError(
                    f"structured run failed to produce an accepted final_answer: {error}"
                )

        # Model responded in plain text without calling any tool
        if not getattr(last_message, "tool_calls", None):
            if config.response_schema is not None:
                # Invariant 1: inject feedback, continue loop
                emit(
                    observers,
                    "plain_text_blocked",
                    {
                        "iteration": iteration,
                        "content": _preview(last_message.content),
                    },
                )
                return "force_final_feedback"
            return "end"

        return "tools"

    def inject_final_feedback(state: AgentState) -> dict:
        """
        Invariant 1 enforcement: model escaped the loop in plain text.
        Inject a HumanMessage telling it to call final_answer.
        """
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "You must keep working through tools. Freeze publishable chart data before publication, and only call `final_answer` after markdown, chart anchors, and citation markers are ready."
                        if state.get("metadata", {}).get(
                            "require_publishable_data"
                        )
                        and not state.get("publishable_data_prepared", False)
                        else (
                            "You must complete this task through tools. In chat you may call `simple_reply` for a direct markdown answer, or call `final_answer` when a full structured report is needed. Do not respond in plain text."
                            if completion_tool_names
                            else "You must call the `final_answer` tool to complete this task. Do not respond in plain text. If something failed, fix only that issue, then call `final_answer` again."
                        )
                    )
                )
            ]
        }

    # ── graph assembly ─────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", execute_tools)
    graph.add_node("force_final_feedback", inject_final_feedback)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "force_final_feedback": "force_final_feedback",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "tools",
        after_tools,
        {
            "llm": "llm",
            "end": END,
        },
    )
    graph.add_edge("force_final_feedback", "llm")

    return graph.compile()
