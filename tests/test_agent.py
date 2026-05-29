"""
Tests for Agent.run() and Conversation using FakeListChatModel.

No real LLM calls. FakeListChatModel returns pre-scripted responses,
letting us test loop invariants, history accumulation, and result shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from sinclair import (
    Agent,
    AgentConfig,
    Conversation,
    FinalAnswerError,
    PythonKernel,
)


class Answer(BaseModel):
    summary: str
    score: float


# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_llm_text(text: str):
    """LLM that returns plain text (no tool call)."""
    m = MagicMock()
    ai_msg = AIMessage(content=text)
    ai_msg.tool_calls = []
    m.bind_tools.return_value = m
    m.invoke.return_value = ai_msg
    return m


def _fake_llm_tool_call(tool_name: str, args: dict, then_text: str = ""):
    """
    LLM that first returns a tool call, then plain text.
    Simulates: model calls tool → gets result → concludes.
    """
    call_count = {"n": 0}

    tool_call_msg = AIMessage(content="")
    tool_call_msg.tool_calls = [
        {"name": tool_name, "args": args, "id": "tc_001"}
    ]
    tool_call_msg.usage_metadata = {"input_tokens": 50, "output_tokens": 10}

    text_msg = AIMessage(content=then_text)
    text_msg.tool_calls = []
    text_msg.usage_metadata = {"input_tokens": 60, "output_tokens": 15}

    m = MagicMock()

    def _invoke(messages):
        call_count["n"] += 1
        return tool_call_msg if call_count["n"] == 1 else text_msg

    m.bind_tools.return_value = m
    m.invoke.side_effect = _invoke
    return m


def _fake_llm_sequence(*messages: AIMessage):
    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.side_effect = list(messages)
    return m


def _fake_llm_final_answer(schema_cls: type[BaseModel], data: dict):
    """LLM that immediately calls final_answer with valid data."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "final_answer", "args": data, "id": "tc_fa"}]
    msg.usage_metadata = {"input_tokens": 100, "output_tokens": 30}

    m = MagicMock()
    m.bind_tools.return_value = m
    m.invoke.return_value = msg
    return m


# ── tests: text-free mode ─────────────────────────────────────────────────────


def test_run_text_free_returns_string():
    llm = _fake_llm_text("The answer is 42.")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    result = agent.run("what is the answer?")
    assert isinstance(result.reply, str)
    assert "42" in result.reply


def test_run_result_has_messages():
    llm = _fake_llm_text("hello")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    result = agent.run("hi")
    assert len(result.messages) > 0


def test_agent_config_reasoning_level_passed_to_openai(monkeypatch):
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools, tool_choice=None):
            return self

        def invoke(self, messages):
            msg = AIMessage(content="ok")
            msg.tool_calls = []
            msg.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
            return msg

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    agent = Agent(
        tools=[],
        config=AgentConfig(model="gpt-5.4-mini", reasoning_level="high"),
    )

    result = agent.run("hi")

    assert result.reply == "ok"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["temperature"] == 1.0
    assert captured["use_responses_api"] is True
    assert captured["reasoning"] == {"effort": "high"}


# ── tests: structured output ──────────────────────────────────────────────────


def test_run_structured_output():
    data = {"summary": "things went well", "score": 0.9}
    llm = _fake_llm_final_answer(Answer, data)
    agent = Agent(
        tools=[],
        config=AgentConfig(llm=llm, response_schema=Answer),
    )
    result = agent.run("analyze this")
    assert isinstance(result.reply, Answer)
    assert result.reply.summary == "things went well"
    assert result.reply.score == 0.9
    assert result.stopped_reason == "final_answer"


def test_run_structured_stopped_reason():
    data = {"summary": "ok", "score": 0.5}
    llm = _fake_llm_final_answer(Answer, data)
    agent = Agent(
        tools=[], config=AgentConfig(llm=llm, response_schema=Answer)
    )
    result = agent.run("go")
    assert result.stopped_reason == "final_answer"


def test_run_structured_plain_text_gets_feedback_then_final_answer():
    first = AIMessage(content="I think this looks good.")
    first.tool_calls = []
    first.usage_metadata = {"input_tokens": 40, "output_tokens": 10}

    second = AIMessage(content="")
    second.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "done", "score": 0.8},
            "id": "tc_final",
        }
    ]
    second.usage_metadata = {"input_tokens": 50, "output_tokens": 20}

    llm = _fake_llm_sequence(first, second)
    agent = Agent(
        tools=[],
        config=AgentConfig(llm=llm, response_schema=Answer, max_iterations=3),
    )

    result = agent.run("analyze this")

    assert isinstance(result.reply, Answer)
    assert llm.invoke.call_count == 2

    second_call_messages = llm.invoke.call_args_list[1][0][0]
    assert any(
        isinstance(message, HumanMessage)
        and "You must call the `final_answer` tool" in message.content
        for message in second_call_messages
    )


def test_run_structured_validation_error_returns_feedback_then_recovers():
    invalid = AIMessage(content="")
    invalid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "missing score"},
            "id": "tc_invalid",
        }
    ]
    invalid.usage_metadata = {"input_tokens": 30, "output_tokens": 8}

    valid = AIMessage(content="")
    valid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "fixed", "score": 0.7},
            "id": "tc_valid",
        }
    ]
    valid.usage_metadata = {"input_tokens": 35, "output_tokens": 12}

    llm = _fake_llm_sequence(invalid, valid)
    agent = Agent(
        tools=[],
        config=AgentConfig(llm=llm, response_schema=Answer, max_iterations=3),
    )

    result = agent.run("analyze this")

    assert isinstance(result.reply, Answer)
    second_call_messages = llm.invoke.call_args_list[1][0][0]
    assert any(
        isinstance(message, ToolMessage)
        and "final_answer rejected" in message.content
        for message in second_call_messages
    )


def test_run_requires_publishable_data_before_final_answer():
    def get_final_chart_numbers() -> str:
        return '{"ok":true}'

    invalid = AIMessage(content="")
    invalid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "too soon", "score": 0.5},
            "id": "tc_invalid",
        }
    ]
    invalid.usage_metadata = {"input_tokens": 30, "output_tokens": 8}

    prep = AIMessage(content="")
    prep.tool_calls = [
        {"name": "get_final_chart_numbers", "args": {}, "id": "tc_prep"}
    ]
    prep.usage_metadata = {"input_tokens": 35, "output_tokens": 10}

    valid = AIMessage(content="")
    valid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "done", "score": 0.7},
            "id": "tc_valid",
        }
    ]
    valid.usage_metadata = {"input_tokens": 35, "output_tokens": 12}

    llm = _fake_llm_sequence(invalid, prep, valid)
    agent = Agent(
        tools=[
            StructuredTool.from_function(
                get_final_chart_numbers,
                description="Freeze publishable chart data.",
            ),
        ],
        config=AgentConfig(llm=llm, response_schema=Answer, max_iterations=5),
    )

    result = agent.run(
        "analyze this",
        metadata={"require_publishable_data": True},
    )

    assert isinstance(result.reply, Answer)
    second_call_messages = llm.invoke.call_args_list[1][0][0]
    assert any(
        isinstance(message, ToolMessage)
        and "freeze publishable chart data" in message.content
        for message in second_call_messages
    )


def test_run_with_tool_call_then_plain_text_reply():
    def add(x: int, y: int) -> int:
        return x + y

    tool = StructuredTool.from_function(
        add, name="add", description="Add numbers"
    )
    llm = _fake_llm_tool_call(
        "add", {"x": 20, "y": 22}, then_text="The answer is 42."
    )
    agent = Agent(tools=[tool], config=AgentConfig(llm=llm))

    result = agent.run("what is 20+22?")

    assert result.reply == "The answer is 42."
    assert llm.invoke.call_count == 2


def test_tool_retry_emits_observer_and_recovers():
    calls = {"n": 0}
    events: list[tuple[str, dict]] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    tool = StructuredTool.from_function(
        flaky, name="flaky", description="Sometimes fails"
    )
    llm = _fake_llm_tool_call("flaky", {}, then_text="done")
    agent = Agent(
        tools=[tool],
        config=AgentConfig(
            llm=llm,
            tool_retries=1,
            tool_retry_backoff=0.0,
            observers=[lambda event, payload: events.append((event, payload))],
        ),
    )

    result = agent.run("run flaky")

    assert result.reply == "done"
    assert any(event == "tool_retry" for event, _ in events)
    assert calls["n"] == 2


def test_structured_run_raises_when_finalization_cannot_converge():
    events: list[tuple[str, dict]] = []
    llm = _fake_llm_text("still thinking")
    agent = Agent(
        tools=[],
        config=AgentConfig(
            llm=llm,
            response_schema=Answer,
            max_iterations=1,
            observers=[lambda event, payload: events.append((event, payload))],
        ),
    )

    with pytest.raises(FinalAnswerError):
        agent.run("analyze this")

    assert any(event == "finalization_started" for event, _ in events)
    assert any(event == "finalization_failed" for event, _ in events)


def test_structured_finalization_keeps_tools_available_after_rejection():
    def add(x: int, y: int) -> int:
        return x + y

    tool = StructuredTool.from_function(
        add, name="add", description="Add numbers"
    )

    invalid = AIMessage(content="")
    invalid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "missing score"},
            "id": "tc_invalid",
        }
    ]
    invalid.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    tool_call = AIMessage(content="")
    tool_call.tool_calls = [
        {"name": "add", "args": {"x": 1, "y": 2}, "id": "tc_tool"}
    ]
    tool_call.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    valid = AIMessage(content="")
    valid.tool_calls = [
        {
            "name": "final_answer",
            "args": {"summary": "fixed after tool", "score": 0.8},
            "id": "tc_valid",
        }
    ]
    valid.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    llm = _fake_llm_sequence(invalid, tool_call, valid)
    agent = Agent(
        tools=[tool],
        config=AgentConfig(llm=llm, response_schema=Answer, max_iterations=1),
    )

    result = agent.run("analyze this")

    assert isinstance(result.reply, Answer)
    assert result.reply.summary == "fixed after tool"
    assert llm.invoke.call_count == 3


# ── tests: history ────────────────────────────────────────────────────────────


def test_run_with_history():
    llm = _fake_llm_text("continuing from before")
    prior = [HumanMessage(content="context message")]
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    _ = agent.run("follow up", history=prior)
    # Check history was passed — llm.invoke received messages including prior
    call_args = llm.invoke.call_args[0][0]
    contents = [m.content for m in call_args]
    assert "context message" in contents


# ── tests: Conversation ───────────────────────────────────────────────────────


def test_conversation_accumulates_history():
    llm = _fake_llm_text("reply 1")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    conv = agent.chat()
    conv.send("first message")
    assert len(conv.history) > 0


def test_conversation_reset_clears_history():
    llm = _fake_llm_text("reply")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    conv = agent.chat()
    conv.send("hello")
    conv.reset()
    assert len(conv.history) == 0
    assert len(conv.results) == 0


def test_conversation_results_accumulate():
    llm = _fake_llm_text("reply")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    conv = agent.chat()
    conv.send("msg 1")
    conv.send("msg 2")
    assert len(conv.results) == 2


def test_conversation_dump_load():
    llm = _fake_llm_text("reply")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    conv = agent.chat()
    conv.send("remember this")

    snap = conv.dump(session_id="test-session")
    assert snap.session_id == "test-session"
    assert len(snap.messages) > 0

    conv2 = Conversation.load(agent, snap)
    assert len(conv2.history) == len(conv.history)


def test_conversation_dump_load_restores_kernel_snapshot():
    llm = _fake_llm_text("reply")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    kernel = PythonKernel(env={"x": 41})
    kernel.execute("x = x + 1")

    conv = agent.chat(kernel=kernel)
    snap = conv.dump(session_id="with-kernel")

    conv2 = Conversation.load(agent, snap)

    assert conv2.dump().namespace_snapshot["x"] == 42


def test_conversation_load_reuses_existing_kernel_config_with_snapshot_state():
    llm = _fake_llm_text("reply")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))

    source_kernel = PythonKernel(env={"x": 1})
    source_kernel.execute("x = 7")
    snap = agent.chat(kernel=source_kernel).dump()

    existing_kernel = PythonKernel(
        timeout=5.0, restricted=False, allowed_modules=["math"]
    )
    conv = Conversation.load(agent, snap, kernel=existing_kernel)

    assert conv.dump().namespace_snapshot["x"] == 7
    assert conv._kernel.timeout == 5.0
    assert conv._kernel.restricted is False
    assert conv._kernel.allowed_modules == ["math"]


# ── tests: run is stateless / thread-safe ────────────────────────────────────


def test_run_is_stateless():
    """Two consecutive run() calls don't share state."""
    llm = _fake_llm_text("independent")
    agent = Agent(tools=[], config=AgentConfig(llm=llm))
    r1 = agent.run("first")
    r2 = agent.run("second")
    # Each result has its own messages
    assert r1.messages is not r2.messages
