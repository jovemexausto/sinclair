"""
_python.py — PythonKernel and ExecutionResult.

PythonKernel is a stateful Python execution environment. It persists
its namespace between execute() calls, making it suitable for multi-step
data analysis where the model builds state incrementally.
"""

from __future__ import annotations

import io
import builtins as py_builtins
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import copy
from dataclasses import dataclass, field
from threading import current_thread, main_thread
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, field_validator

from ._intent import validate_intent_text


class _ExecutionTimedOut(Exception):
    pass


def _make_restricted_builtins(allowed_modules: list[str]) -> dict:
    _ = allowed_modules
    return dict(py_builtins.__dict__)


# ── result ─────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    error: str | None  # None if no exception
    variables_added: list[str]
    variables_modified: list[str]
    execution_time_s: float

    @staticmethod
    def _compact_stream(
        text: str, *, max_lines: int = 40, max_chars: int = 4000
    ) -> str:
        if not text:
            return text

        lines = text.splitlines()
        if len(lines) > max_lines:
            head = lines[: max_lines // 2]
            tail = lines[-(max_lines - len(head)) :]
            text = (
                "\n".join(head)
                + f"\n...[truncated {len(lines) - max_lines} lines]...\n"
                + "\n".join(tail)
            )

        if len(text) > max_chars:
            head_chars = max_chars // 2
            tail_chars = max_chars - head_chars
            text = (
                text[:head_chars]
                + f"\n...[truncated {len(text) - max_chars} chars]...\n"
                + text[-tail_chars:]
            )

        return text

    def __str__(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self._compact_stream(self.stdout).rstrip())
        if self.stderr:
            parts.append(
                f"[stderr]\n{self._compact_stream(self.stderr).rstrip()}"
            )
        if self.error:
            parts.append(f"[error]\n{self.error}")
        if self.variables_added:
            parts.append(f"[new vars] {', '.join(self.variables_added)}")
        if self.variables_modified:
            parts.append(
                f"[modified vars] {', '.join(self.variables_modified)}"
            )
        if not parts:
            parts.append("[ok — no output]")
        return "\n".join(parts)


# ── kernel ─────────────────────────────────────────────────────────────────────


@dataclass
class PythonKernel:
    env: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    restricted: bool = True
    allowed_modules: list[str] = field(
        default_factory=lambda: [
            "pandas",
            "numpy",
            "matplotlib",
            "scipy",
            "sklearn",
            "json",
            "math",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "statistics",
            "re",
        ]
    )

    def __post_init__(self) -> None:
        # _initial_env is the clean copy for reset()
        self._initial_env: dict[str, Any] = copy(self.env)
        self._namespace: dict[str, Any] = copy(self.env)

    def execute(self, code: str) -> ExecutionResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        ns_before = set(self._namespace.keys())
        ids_before = {k: id(v) for k, v in self._namespace.items()}

        builtins = (
            _make_restricted_builtins(self.allowed_modules)
            if self.restricted
            else __builtins__
        )
        exec_globals = {**self._namespace, "__builtins__": builtins}

        error: str | None = None
        t0 = time.perf_counter()

        def _handle_timeout(signum: int, frame: Any) -> None:
            raise _ExecutionTimedOut()

        can_enforce_timeout = (
            self.timeout > 0
            and current_thread() is main_thread()
            and hasattr(signal, "setitimer")
        )
        previous_handler = None

        try:
            if can_enforce_timeout:
                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.setitimer(signal.ITIMER_REAL, self.timeout)

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<agent>", "exec"), exec_globals)  # noqa: S102
        except _ExecutionTimedOut:
            error = f"TimeoutError: execution exceeded {self.timeout:.2f}s"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if can_enforce_timeout:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
            elapsed = time.perf_counter() - t0

        # Sync namespace — exclude dunder keys and __builtins__
        self._namespace = {
            k: v for k, v in exec_globals.items() if not k.startswith("__")
        }

        ns_after = set(self._namespace.keys())
        added = sorted(ns_after - ns_before)
        modified = sorted(
            k
            for k in ns_before & ns_after
            if id(self._namespace[k]) is not ids_before.get(k)
        )

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
            variables_added=added,
            variables_modified=modified,
            execution_time_s=elapsed,
        )

    def reset(self) -> None:
        self._namespace = copy(self._initial_env)

    def snapshot(self) -> dict[str, Any]:
        """
        Return a copy of the current namespace with only JSON-serializable values.
        Non-serializable objects (DataFrames, numpy arrays, etc.) are silently skipped.
        The caller decides what to include before passing env to a new kernel.
        """
        import json

        result = {}
        for k, v in self._namespace.items():
            try:
                json.dumps(v)
                result[k] = v
            except (TypeError, ValueError):
                pass
        return result

    def as_tool(self) -> BaseTool:
        """Return a BaseTool that exposes this kernel to the agent."""
        kernel = self

        class RunPythonInput(BaseModel):
            code: str
            intent: str

            @field_validator("intent")
            @classmethod
            def _validate_intent(cls, value: str) -> str:
                return validate_intent_text(value)

        def _run(code: str, intent: str) -> str:
            _ = intent
            result = kernel.execute(code)
            return str(result)

        return StructuredTool.from_function(
            func=_run,
            name="run_python",
            description=(
                "Execute Python code in a persistent sandbox. "
                "Variables persist between calls. "
                "Use print() to inspect values. "
                "Always provide `intent` as one short first-person product-facing status in progress. "
                "Do not import modules outside the allowed list."
            ),
            args_schema=RunPythonInput,
        )
