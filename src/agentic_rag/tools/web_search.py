"""Web search via Tavily, for facts the internal corpus cannot cover."""

from __future__ import annotations

import logging
from typing import Any

from tavily import TavilyClient

from agentic_rag.tools.base import Tool, ToolContext, string_params

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


def web_search(ctx: ToolContext, search_query: str) -> Any:
    """Search the web and return the top results."""
    if not ctx.settings.tavily_api_key:
        return "Web search is unavailable: TV_API_KEY is not configured."

    try:
        client = TavilyClient(api_key=ctx.settings.tavily_api_key)
        response = client.search(search_query, max_results=MAX_RESULTS)
    except Exception as exc:
        logger.warning("web search failed: %s", exc)
        return f"Web search failed: {exc}"

    # Trim the payload to what the model actually needs to cite an answer.
    return [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
        }
        for result in response.get("results", [])
    ]


TOOL = Tool(
    name="web_search_tool",
    description=(
        "Search the public internet for up-to-date facts. Use this only when the "
        "question cannot be answered from the internal documents or metrics, or "
        "when the internal answer is likely out of date."
    ),
    parameters=string_params(search_query="The internet search query."),
    handler=web_search,
)
