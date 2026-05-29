"""Tests for _metrics.py — AgentResult aggregates and cost calculation."""

from __future__ import annotations


from sinclair._metrics import AgentResult, IterMetrics, _cost


def _make_metrics(iteration, latency, in_tok, out_tok, model="gpt-5.4-mini"):
    cost = _cost(model, in_tok, out_tok)
    return IterMetrics(
        iteration=iteration,
        latency_s=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        tool_calls=[],
    )


def test_aggregates():
    m1 = _make_metrics(1, 1.0, 100, 20)
    m2 = _make_metrics(2, 2.0, 200, 40)
    result = AgentResult(reply="hello", messages=[], iter_metrics=[m1, m2])

    assert result.iterations == 2
    assert result.input_tokens == 300
    assert result.output_tokens == 60
    assert result.total_tokens == 360
    assert abs(result.latency_s - 3.0) < 1e-6


def test_cost_known_model():
    cost = _cost("gpt-5.4-mini", 1_000_000, 0)
    assert abs(cost - 0.75) < 1e-6

    cost = _cost("gpt-5.4-mini", 0, 1_000_000)
    assert abs(cost - 4.50) < 1e-6


def test_cost_unknown_model():
    assert _cost("some-unknown-model-xyz", 1000, 100) is None


def test_cost_prefix_match():
    # "gpt-5.4-mini-2026-01-01" should match "gpt-5.4-mini"
    cost = _cost("gpt-5.4-mini-2026-01-01", 1_000_000, 0)
    assert cost is not None
    assert abs(cost - 0.75) < 1e-6


def test_summary_format():
    m = _make_metrics(1, 1.23, 340, 41)
    result = AgentResult(reply="x", messages=[], iter_metrics=[m])
    s = result.summary()
    assert "iter=1" in s
    assert "tok=381" in s
    assert "lat=1.23s" in s
    assert "cost=$" in s


def test_summary_no_cost():
    m = IterMetrics(
        iteration=1, latency_s=1.0, input_tokens=100, output_tokens=10, cost_usd=None
    )
    result = AgentResult(reply="x", messages=[], iter_metrics=[m])
    assert "cost" not in result.summary()


def test_stopped_reason_default():
    result = AgentResult(reply="x", messages=[])
    assert result.stopped_reason == "final_answer"
