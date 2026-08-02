"""Tool registry.

Each tool module exposes a single ``TOOL`` object. Adding a capability to the
agent means writing one module and adding it to ``_MODULES`` below.
"""

from __future__ import annotations

from agentic_rag.tools import (
    calculator,
    query_metrics,
    read_document,
    search_documents,
    web_search,
)
from agentic_rag.tools.base import Tool, ToolContext, string_params

_MODULES = (search_documents, read_document, query_metrics, web_search, calculator)

#: Every tool the agent can call, keyed by the name the model uses.
REGISTRY: dict[str, Tool] = {module.TOOL.name: module.TOOL for module in _MODULES}

__all__ = ["REGISTRY", "Tool", "ToolContext", "string_params"]
