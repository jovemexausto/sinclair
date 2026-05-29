"""
_agent.py — Agent and Conversation.

Agent compiles the graph once and reuses it. run() is stateless and
thread-safe — each call creates independent state. chat() returns a
Conversation that manages history across multiple send() calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

from pydantic import BaseModel

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from ._config import AgentConfig
from ._errors import FinalAnswerError
from ._loop import build_graph
from ._metrics import AgentResult
from ._state import AgentState


def _coerce_free_reply(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(
            part.strip() for part in parts if part.strip()
        ).strip()
    return str(value or "")


@dataclass
class ConversationSnapshot:
    session_id: str
    messages: list[dict]  # serialized via langchain message_to_dict
    namespace_snapshot: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        tools: Sequence[Any],
        config: AgentConfig | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._tools = list(tools)
        # Graph is compiled once; result_holder is per-invocation (see run())

    def run(
        self,
        prompt: str,
        *,
        history: list[BaseMessage] | None = None,
        kernel: Any | None = None,
        metadata: dict | None = None,
    ) -> AgentResult:
        """
        Stateless pipeline execution. Thread-safe — each call is independent.
        Pass kernel= to make PythonKernel state available for this run.
        """
        tools = list(self._tools)
        if kernel is not None:
            tools = [
                t for t in tools if getattr(t, "name", "") != "run_python"
            ]
            tools.append(kernel.as_tool())

        result_holder: dict[str, Any] = {}
        graph = build_graph(tools, self._config, result_holder)

        messages: list[BaseMessage] = [
            SystemMessage(content=self._config.system_prompt)
        ]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=prompt))

        initial_state: AgentState = {
            "messages": messages,
            "iteration": 0,
            "iter_metrics": [],
            "final_answer_accepted": False,
            "finalization_started": False,
            "finalization_turns": 0,
            "last_validation_error": None,
            "publishable_data_prepared": False,
            "metadata": metadata or {},
        }

        final_state: AgentState = graph.invoke(initial_state)

        return _build_result(final_state, result_holder, self._config)

    def chat(self, *, kernel: Any | None = None) -> "Conversation":
        """Return a stateful Conversation backed by this agent."""
        return Conversation(agent=self, kernel=kernel)


class Conversation:
    def __init__(self, agent: Agent, kernel: Any | None = None) -> None:
        self._agent = agent
        self._kernel = kernel
        self._history: list[BaseMessage] = []
        self._results: list[AgentResult] = []

    def send(
        self,
        message: str,
        metadata: dict | None = None,
    ) -> AgentResult:
        result = self._agent.run(
            message,
            history=self._history,
            kernel=self._kernel,
            metadata=metadata,
        )
        # Accumulate history — append only the new turn messages (skip system)
        self._history.extend(
            m for m in result.messages if not isinstance(m, SystemMessage)
        )
        self._results.append(result)
        return result

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    @property
    def results(self) -> list[AgentResult]:
        return list(self._results)

    def reset(self) -> None:
        self._history.clear()
        self._results.clear()
        if self._kernel is not None:
            self._kernel.reset()

    def dump(self, session_id: str = "") -> ConversationSnapshot:
        from langchain_core.messages import messages_to_dict

        ns = self._kernel.snapshot() if self._kernel is not None else {}
        return ConversationSnapshot(
            session_id=session_id,
            messages=messages_to_dict(self._history),
            namespace_snapshot=ns,
        )

    @classmethod
    def load(
        cls,
        agent: Agent,
        snapshot: ConversationSnapshot,
        kernel: Any | None = None,
    ) -> "Conversation":
        from .python import PythonKernel
        from langchain_core.messages import messages_from_dict

        if snapshot.namespace_snapshot:
            if kernel is None:
                kernel = PythonKernel(env=snapshot.namespace_snapshot)
            elif isinstance(kernel, PythonKernel):
                kernel = PythonKernel(
                    env=snapshot.namespace_snapshot,
                    timeout=kernel.timeout,
                    restricted=kernel.restricted,
                    allowed_modules=list(kernel.allowed_modules),
                )

        conv = cls(agent=agent, kernel=kernel)
        conv._history = messages_from_dict(snapshot.messages)
        return conv


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_result(
    state: AgentState,
    result_holder: dict[str, Any],
    config: AgentConfig,
) -> AgentResult:
    # Determine reply
    if "reply" in result_holder:
        reply = cast(str | BaseModel, result_holder["reply"])
        stopped_reason = result_holder.get("stopped_reason") or "final_answer"
    else:
        if config.response_schema is not None:
            raise FinalAnswerError(
                "structured run ended without an accepted final_answer"
            )

        # Text free mode or max_iterations reached
        last_ai = next(
            (
                m
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage)
            ),
            None,
        )
        reply = _coerce_free_reply(last_ai.content) if last_ai else ""
        stopped_reason = (
            "max_iterations"
            if state["iteration"] >= config.max_iterations
            else "complete"
        )

    # messages: everything after the system prompt
    messages = [
        m for m in state["messages"] if not isinstance(m, SystemMessage)
    ]

    return AgentResult(
        reply=reply,
        messages=messages,
        iter_metrics=state.get("iter_metrics", []),
        stopped_reason=stopped_reason,
        metadata=state.get("metadata", {}),
    )
