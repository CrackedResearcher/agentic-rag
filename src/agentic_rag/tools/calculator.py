"""Arithmetic on retrieved numbers, so the model never does mental math."""

from __future__ import annotations

from simpleeval import simple_eval

from agentic_rag.tools.base import Tool, ToolContext, string_params


def calculate(ctx: ToolContext, expression: str) -> str:
    """Evaluate a math expression in a sandboxed evaluator."""
    try:
        return f"{expression} = {simple_eval(expression)}"
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


TOOL = Tool(
    name="calculator_tool",
    description=(
        "Evaluate a mathematical expression, e.g. '(12260 - 11058) / 11058 * 100'. "
        "Use this for every calculation on retrieved numbers — growth rates, "
        "totals, margins — instead of computing them yourself."
    ),
    parameters=string_params(
        expression="A math expression using numbers and + - * / ( ) operators."
    ),
    handler=calculate,
)
