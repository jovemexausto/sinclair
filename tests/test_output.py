"""Tests for _output.py — final_answer tool and validation error formatting."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from sinclair._output import _format_validation_error, make_final_answer_tool


class MySchema(BaseModel):
    trend: str
    value: float


def test_final_answer_accepted():
    tool, holder = make_final_answer_tool(MySchema)
    result = tool.invoke({"trend": "up", "value": 1.5})
    assert result == "__final_answer_accepted__"
    assert isinstance(holder["reply"], MySchema)
    assert holder["reply"].trend == "up"


def test_final_answer_invalid_type_raises():
    """LangChain validates args_schema before calling _run — ValidationError propagates."""
    tool, holder = make_final_answer_tool(MySchema)
    with pytest.raises(ValidationError):
        tool.invoke({"trend": "up", "value": "not-a-float"})
    assert "reply" not in holder


def test_final_answer_missing_field_raises():
    """Missing required field: ValidationError raised before _run is called."""
    tool, holder = make_final_answer_tool(MySchema)
    with pytest.raises(ValidationError):
        tool.invoke({"trend": "up"})
    assert "reply" not in holder


def test_semantic_validator_called():
    def validate(obj: MySchema) -> None:
        if obj.trend not in ("up", "down", "stable"):
            raise ValueError("trend must be up, down, or stable")

    tool, holder = make_final_answer_tool(MySchema, validator=validate)
    result = tool.invoke({"trend": "sideways", "value": 1.0})
    assert "final_answer rejected" in result
    assert "reply" not in holder


def test_semantic_validator_passes():
    def validate(obj: MySchema) -> None:
        pass

    tool, holder = make_final_answer_tool(MySchema, validator=validate)
    result = tool.invoke({"trend": "up", "value": 2.0})
    assert result == "__final_answer_accepted__"
    assert "reply" in holder


def test_format_validation_error_pydantic():
    try:
        MySchema(trend="up", value="bad")
    except ValidationError as exc:
        msg = _format_validation_error(exc)
        assert "final_answer rejected" in msg
        assert "value" in msg


def test_format_validation_error_value_error():
    msg = _format_validation_error(ValueError("too low"))
    assert "final_answer rejected" in msg
    assert "too low" in msg
