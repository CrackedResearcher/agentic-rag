"""The contract every tool implements.

A tool is a plain function plus a JSON-schema description of its arguments.
The schema is handed to the model so it knows when and how to call the tool;
the function is invoked locally with whatever arguments the model produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from google import genai

from agentic_rag.config import Settings


@dataclass(frozen=True)
class ToolContext:
    """Shared resources handed to every tool invocation.

    Tools receive this instead of reaching for module-level globals, which
    keeps them importable and testable in isolation.
    """

    client: genai.Client
    settings: Settings
    table: Any = None  # lancedb.table.Table, opened lazily by the CLI


class ToolHandler(Protocol):
    def __call__(self, ctx: ToolContext, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Tool:
    """A single capability exposed to the model.

    Attributes:
        name: Identifier the model uses to call the tool.
        description: What the tool does, written for the model to read.
        parameters: JSON schema for the tool's arguments.
        handler: Callable invoked as ``handler(ctx, **args)``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def declaration(self) -> dict[str, Any]:
        """Return the function declaration in the shape the Gemini API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        return self.handler(ctx, **kwargs)


def string_params(**fields: str) -> dict[str, Any]:
    """Build a JSON schema where every field is a required string.

    Every tool here happens to take string arguments only, so this removes a
    lot of repeated boilerplate::

        string_params(search_query="what to look for")
    """
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in fields.items()
        },
        "required": list(fields),
    }
