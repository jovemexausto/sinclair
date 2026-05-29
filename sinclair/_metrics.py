"""
_metrics.py — AgentResult and per-iteration metrics.

Cost table covers the models most likely used in prod. Returns None for
unknowns rather than raising — caller can decide how to handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
}


def _cost(model: str, input_tok: int, output_tok: int) -> float | None:
    prices = _PRICE_PER_1M.get(model)
    if prices is None:
        # Try prefix match
        for key, p in _PRICE_PER_1M.items():
            if model.startswith(key):
                prices = p
                break
    if prices is None:
        return None
    in_price, out_price = prices
    return (input_tok * in_price + output_tok * out_price) / 1_000_000


@dataclass
class IterMetrics:
    iteration: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    reply: str | BaseModel
    messages: list[BaseMessage]
    iter_metrics: list[IterMetrics] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # "final_answer" | "max_iterations" | "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── aggregates ────────────────────────────────────────────────────────────

    @property
    def iterations(self) -> int:
        return len(self.iter_metrics)

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.iter_metrics)

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.iter_metrics)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def latency_s(self) -> float:
        return sum(m.latency_s for m in self.iter_metrics)

    @property
    def cost_usd(self) -> float | None:
        costs = [m.cost_usd for m in self.iter_metrics if m.cost_usd is not None]
        return sum(costs) if costs else None

    def summary(self) -> str:
        cost_part = f"  cost=${self.cost_usd:.6f}" if self.cost_usd is not None else ""
        return (
            f"iter={self.iterations}"
            f"  tok={self.total_tokens}({self.input_tokens}+{self.output_tokens})"
            f"  lat={self.latency_s:.2f}s"
            f"{cost_part}"
        )
