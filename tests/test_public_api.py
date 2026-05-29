"""Public API smoke tests."""

from __future__ import annotations

from importlib import import_module

import sinclair


def test_top_level_exports():
    assert sinclair.Agent is not None
    assert sinclair.AgentConfig is not None
    assert sinclair.AgentResult is not None
    assert sinclair.Conversation is not None
    assert sinclair.ConversationSnapshot is not None
    assert sinclair.FinalAnswerError is not None
    assert sinclair.PythonKernel is not None
    assert sinclair.ExecutionResult is not None
    assert sinclair.make_trace_observer is not None


def test_python_module_exports():
    mod = import_module("sinclair.python")
    assert mod.PythonKernel is sinclair.PythonKernel
    assert mod.ExecutionResult is sinclair.ExecutionResult


def test_trace_module_exports():
    mod = import_module("sinclair.trace")
    assert mod.make_trace_observer is sinclair.make_trace_observer
