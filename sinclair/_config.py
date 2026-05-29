"""
_config.py — AgentConfig dataclass.

llm takes precedence over model/temperature. Pass a pre-configured
BaseChatModel when you need custom retry, proxy, or credentials setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from ._obs import ObserverFn


@dataclass
class AgentConfig:
    system_prompt: str = "You are a helpful assistant."
    llm: BaseChatModel | None = None  # if set, model/temperature are ignored
    model: str = "gpt-5.4-mini"
    temperature: float = 1.0
    reasoning_level: str | None = (
        "low"  # OpenAI reasoning effort: low | medium | high
    )
    max_iterations: int = 10
    finalization_window: int = 2
    tool_retries: int = 2
    tool_retry_backoff: float = 0.5  # seconds; doubles each attempt
    response_schema: Type[BaseModel] | None = None
    response_validator: Callable[[BaseModel], None] | None = None
    allow_simple_reply: bool = False
    observers: list[ObserverFn] = field(default_factory=list)
