"""
_obs.py — observer protocol and built-in implementations.

ObserverFn is a simple callable: (event_name, payload) -> None.
The SDK never decides where data goes — callers inject observers.

Events emitted:
  llm_call          iteration, mode, latency_s, input_tokens, output_tokens, cost_usd, tool_calls
  tool_start        tool, args, intent, iteration
  tool_end          tool, args, intent, result_preview, status, iteration
  tool_retry        tool, args, intent, attempt, wait_s, error, iteration
  tool_error        tool, args, intent, error, attempt, iteration
  final_answer_accepted iteration
  final_answer_rejected iteration, error
  budget_warning iteration, max_iterations
  finalization_started iteration, max_iterations
  finalization_failed iteration, error
  plain_text_blocked iteration, content
  max_iterations_reached iteration
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

ObserverFn = Callable[[str, dict[str, Any]], None]

_logger = logging.getLogger("sinclair")


def emit(
    observers: list[ObserverFn], event: str, payload: dict[str, Any]
) -> None:
    for obs in observers:
        try:
            obs(event, payload)
        except Exception:
            # observers must never crash the loop
            pass


def log_observer(event: str, payload: dict[str, Any]) -> None:
    """Structured logging observer — uses stdlib logging at DEBUG level."""
    _logger.debug("sinclair.%s %s", event, payload)


def print_observer(event: str, payload: dict[str, Any]) -> None:
    """Simple stdout observer for local development."""
    ts = time.strftime("%H:%M:%S")
    print(f"[sinclair {ts}] {event} | {payload}")
