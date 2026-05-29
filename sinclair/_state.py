"""
_state.py — internal LangGraph state.

Nothing here is public API. The loop uses this TypedDict as its node I/O.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    iter_metrics: list[Any]  # list[IterMetrics] — avoid circular import
    final_answer_accepted: bool
    finalization_started: bool
    finalization_turns: int
    last_validation_error: str | None
    publishable_data_prepared: bool
    metadata: dict[str, Any]
