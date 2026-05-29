"""Tests for _obs.py — observer protocol."""

from __future__ import annotations

from io import StringIO

from sinclair._obs import emit, log_observer, print_observer
from sinclair.trace import make_trace_observer


def testemit_calls_observer():
    events = []

    def obs(event, payload):
        events.append((event, payload))

    emit([obs], "llm_call", {"iteration": 1})
    assert len(events) == 1
    assert events[0][0] == "llm_call"
    assert events[0][1]["iteration"] == 1


def testemit_multiple_observers():
    log = []
    emit(
        [
            lambda e, p: log.append(("a", e)),
            lambda e, p: log.append(("b", e)),
        ],
        "tool_start",
        {},
    )
    assert len(log) == 2


def testemit_swallows_observer_exception():
    """A crashing observer must never crash the loop."""

    def bad_obs(event, payload):
        raise RuntimeError("observer exploded")

    good_log = []

    def good_obs(event, payload):
        good_log.append(event)

    # Should not raise
    emit([bad_obs, good_obs], "test_event", {})
    assert "test_event" in good_log


def testemit_empty_observers():
    emit([], "any_event", {"x": 1})  # should not raise


def test_log_observer_does_not_raise():
    log_observer("llm_call", {"iteration": 1, "latency_s": 0.5})


def test_print_observer_does_not_raise(capsys):
    print_observer("tool_start", {"tools": ["run_python"], "iteration": 2})
    captured = capsys.readouterr()
    assert "tool_start" in captured.out


def test_trace_observer_pretty_prints_structured_tool_result():
    stream = StringIO()
    observer = make_trace_observer(stream=stream)

    observer(
        "tool_end",
        {
            "tool": "search_findings",
            "status": "ok",
            "result_preview": '[{"finding_id":"fd_1"}]',
            "result": [{"finding_id": "fd_1", "claim": "hello"}],
        },
    )

    output = stream.getvalue()
    assert "[result]" in output
    assert '\n  {\n    "finding_id": "fd_1"' in output


def test_trace_observer_prints_tool_intent():
    stream = StringIO()
    observer = make_trace_observer(stream=stream)

    observer(
        "tool_start",
        {
            "tool": "run_python",
            "args": {
                "code": "print('ok')",
                "intent": "Estou conferindo a regra.",
            },
            "intent": "Estou conferindo a regra.",
        },
    )

    output = stream.getvalue()
    assert "intent=Estou conferindo a regra." in output
