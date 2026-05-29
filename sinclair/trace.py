"""Built-in trace observer helpers for CLI debugging."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

ObserverFn = Callable[[str, dict[str, Any]], None]


def make_trace_observer(*, stream: TextIO | None = None) -> ObserverFn:
    """Return a human-readable observer for local CLI traces."""
    stream = stream or sys.stdout

    def write(line: str) -> None:
        print(line, file=stream)

    def write_json_block(label: str, payload: Any) -> None:
        write(label)
        write(json.dumps(payload, ensure_ascii=False, indent=2))

    def observe(event: str, payload: dict[str, Any]) -> None:
        if event == "llm_call":
            write(
                f"[llm:{payload.get('iteration')}] mode={payload.get('mode')} "
                f"tokens={payload.get('input_tokens')}+{payload.get('output_tokens')} "
                f"lat={payload.get('latency_s'):.2f}s"
            )
            return

        if event == "budget_warning":
            write(
                f"[llm:{payload.get('iteration')}] approaching safety limit "
                f"max={payload.get('max_iterations')}"
            )
            return

        if event == "finalization_started":
            write(
                f"[llm:{payload.get('iteration')}] finalization mode started "
                f"max={payload.get('max_iterations')}"
            )
            return

        if event == "finalization_failed":
            write(f"[finalization] failed: {payload.get('error')}")
            return

        if event == "plain_text_blocked":
            write(
                f"[llm:{payload.get('iteration')}] blocked plain text: "
                f"{payload.get('content')}"
            )
            return

        if event == "tool_start":
            tool = payload.get("tool")
            args = payload.get("args", {})
            intent = payload.get("intent")
            write(
                f"[tool:{tool}] start"
                + (f" intent={intent}" if intent else "")
            )
            if tool == "run_python":
                code = args.get("code", "")
                write("[run_python code]")
                write(code.rstrip())
            else:
                write_json_block("[args]", args)
            return

        if event == "tool_retry":
            write(
                f"[tool:{payload.get('tool')}] retry={payload.get('attempt')} "
                f"wait={payload.get('wait_s'):.2f}s error={payload.get('error')}"
            )
            return

        if event == "tool_error":
            write(
                f"[tool:{payload.get('tool')}] error attempt={payload.get('attempt')} "
                f"error={payload.get('error')}"
            )
            return

        if event == "tool_end":
            write(
                f"[tool:{payload.get('tool')}] end status={payload.get('status')} "
                f"preview={payload.get('result_preview')}"
            )
            result = payload.get("result")
            if result not in (None, ""):
                if isinstance(result, (dict, list)):
                    write_json_block("[result]", result)
                else:
                    write("[result]")
                    write(str(result))
            return

        if event == "final_answer_rejected":
            write(f"[final_answer] rejected: {payload.get('error')}")
            return

        if event == "final_answer_accepted":
            write(
                f"[final_answer] accepted at iteration {payload.get('iteration')}"
            )
            return

    return observe
