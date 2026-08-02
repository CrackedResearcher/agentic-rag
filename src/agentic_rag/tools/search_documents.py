"""Semantic search across the indexed document corpus."""

from __future__ import annotations

from typing import Any

from agentic_rag.retrieval import search_chunks
from agentic_rag.tools.base import Tool, ToolContext, string_params


def search_documents(ctx: ToolContext, search_query: str) -> list[dict[str, Any]] | str:
    """Find the passages most relevant to ``search_query``."""
    if ctx.table is None:
        return "The document index is not available. Run `agentic-rag ingest` to build it."
    try:
        return search_chunks(search_query, ctx.client, ctx.table)
    except Exception as exc:  # surfaced to the model so it can recover
        return f"Document search failed: {exc}"


TOOL = Tool(
    name="search_document_tool",
    description=(
        "Search the internal document corpus (company handbook, policies, product "
        "docs, SEC filings) for passages relevant to a question. Returns several "
        "matching excerpts along with the file each came from. Use this first for "
        "any question about the company or its documents."
    ),
    parameters=string_params(
        search_query="A natural-language description of the information you need."
    ),
    handler=search_documents,
)
