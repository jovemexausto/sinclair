"""
_output.py — final_answer tool factory + validation error formatting.

The final_answer tool is the ONLY exit from the agent loop when response_schema
is set. Its args_schema IS the user's Pydantic model, so the JSON Schema reaches
the LLM as a native tool definition — not a prompt instruction.
"""

from __future__ import annotations

from typing import Any, Type

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError


def _format_validation_error(exc: ValidationError | ValueError) -> str:
    """
    Turn a validation error into a short, model-readable message.
    No stack traces. No JSON blobs. Just what's wrong and where.
    """
    if isinstance(exc, ValidationError):
        lines = [
            "final_answer rejected — fix only these fields and keep the rest unchanged:"
        ]
        for err in exc.errors():
            loc = (
                " → ".join(str(p) for p in err["loc"])
                if err["loc"]
                else "root"
            )
            lines.append(f"  - {loc}: {err['msg']}")
        return "\n".join(lines)

    # ValueError from response_validator (semantic check)
    return (
        "final_answer rejected. Keep correct sections unchanged and repair only this issue:\n"
        f"  {exc}"
    )


def make_final_answer_tool(
    schema: Type[BaseModel],
    validator: Any | None = None,
    result_holder: dict[str, Any] | None = None,
) -> tuple[BaseTool, dict]:
    """
    Build a final_answer tool whose args_schema is the user's Pydantic model.

    Returns (tool, result_holder) where result_holder is a dict that will be
    populated with {"reply": <BaseModel instance>} when the tool is invoked
    successfully. The loop inspects this dict to know when to stop.

    Using a mutable dict as a side-channel avoids coupling the tool's return
    value to LangGraph's message routing — the tool always returns a string
    (for the ToolMessage), and the validated object is captured separately.
    """
    result_holder = result_holder if result_holder is not None else {}

    def _run(**kwargs: Any) -> str:
        try:
            instance = schema(**kwargs)
        except ValidationError as exc:
            return _format_validation_error(exc)

        if validator is not None:
            try:
                validator(instance)
            except ValueError as exc:
                return _format_validation_error(exc)

        result_holder["reply"] = instance
        return "__final_answer_accepted__"

    tool = StructuredTool.from_function(
        func=_run,
        name="final_answer",
        description=(
            "When you are ready to finish, submit your final answer with this tool. "
            "You must use it to end the task and should not reply in plain text."
        ),
        args_schema=schema,
    )

    return tool, result_holder


class SimpleMarkdownReplyInput(BaseModel):
    markdown: str = Field(
        description="Direct markdown reply that ends the current chat turn without the full report schema."
    )


def make_simple_reply_tool(
    result_holder: dict[str, Any] | None = None,
) -> tuple[BaseTool, dict[str, Any]]:
    result_holder = result_holder if result_holder is not None else {}

    def _run(markdown: str) -> str:
        text = str(markdown or "").strip()
        if not text:
            return "simple_reply rejected — markdown cannot be empty"
        result_holder["reply"] = text
        result_holder["stopped_reason"] = "simple_reply"
        return "__completion_accepted__"

    tool = StructuredTool.from_function(
        func=_run,
        name="simple_reply",
        description=(
            "Use this only in chat when a direct markdown answer is enough and a full report with findings, citations, and charts is unnecessary."
        ),
        args_schema=SimpleMarkdownReplyInput,
    )
    return tool, result_holder
