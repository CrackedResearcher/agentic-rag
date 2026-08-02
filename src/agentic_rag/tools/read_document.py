"""Read a full document from the corpus by file name."""

from __future__ import annotations

from pathlib import Path

from agentic_rag.ingest import iter_documents
from agentic_rag.loaders import MissingDependency, load
from agentic_rag.tools.base import Tool, ToolContext, string_params

# Cap what a single read can put into the conversation. Beyond this the model
# should be searching, not reading whole filings.
MAX_CHARS = 60_000


def _resolve(docs_dir: Path, file_name: str) -> Path | None:
    """Find the indexed document matching ``file_name``.

    Accepts either the relative path as indexed ("reports/q1.md") or just the
    base name ("q1.md"). Only files the ingest step would pick up are
    candidates, so a model-supplied path can never escape the corpus.
    """
    wanted = Path(file_name)
    for path in iter_documents(docs_dir):
        relative = path.relative_to(docs_dir)
        if relative == wanted or relative.name == wanted.name:
            return path
    return None


def read_document(ctx: ToolContext, file_name: str) -> str:
    """Return the full text of ``file_name`` from the corpus."""
    docs_dir = ctx.settings.docs_dir
    target = _resolve(docs_dir, file_name)

    if target is None:
        available = [str(p.relative_to(docs_dir)) for p in iter_documents(docs_dir)]
        return (
            f"No document named '{file_name}' in the corpus. "
            f"Available documents: {', '.join(available) or 'none'}"
        )

    try:
        text = load(target)
    except (MissingDependency, OSError, ValueError) as exc:
        return f"Could not read '{file_name}': {exc}"

    if len(text) > MAX_CHARS:
        return (
            text[:MAX_CHARS] + f"\n\n[truncated at {MAX_CHARS} characters — "
            "use search_document_tool to find specific passages in the rest]"
        )
    return text


TOOL = Tool(
    name="read_document_tool",
    description=(
        "Read one document from the corpus in full, by file name. Use this after "
        "search_document_tool when an excerpt is not enough and you need the "
        "surrounding context."
    ),
    parameters=string_params(
        file_name="Exact file name as reported by search_document_tool, e.g. 'refund_policy.md'."
    ),
    handler=read_document,
)
