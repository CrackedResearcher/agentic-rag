"""Embeddings and vector search over the document index."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import lancedb
from google import genai

from agentic_rag.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, Settings

logger = logging.getLogger(__name__)

# Gemini embeds documents and queries with different task types; using the
# matching pair on both sides measurably improves retrieval quality.
DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


def embed(text: str, client: genai.Client, task_type: str) -> Sequence[float]:
    """Embed a single piece of text for indexing or querying."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai.types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_document(text: str, client: genai.Client) -> Sequence[float]:
    return embed(text, client, DOCUMENT_TASK)


def embed_query(text: str, client: genai.Client) -> Sequence[float]:
    return embed(text, client, QUERY_TASK)


def open_table(settings: Settings):
    """Open the vector table, with a clear error if it hasn't been built yet."""
    db = lancedb.connect(settings.data_dir)
    if settings.vector_table not in db.table_names():
        raise FileNotFoundError(
            f"Vector table '{settings.vector_table}' not found in {settings.data_dir}. "
            "Run `agentic-rag ingest` first."
        )
    return db.open_table(settings.vector_table)


def search_chunks(
    query: str,
    client: genai.Client,
    table,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the ``top_k`` document chunks most similar to ``query``.

    ``_distance`` is selected explicitly (LanceDB warns when it isn't) and used
    for debug logging, but is not passed to the model — the file and text are
    what it needs to answer and cite.
    """
    vector = embed_query(query, client)
    rows = table.search(vector).select(["text", "file", "id", "_distance"]).limit(top_k).to_list()
    logger.debug(
        "vector search for %r matched %s",
        query,
        [f"{row['file']}@{row['_distance']:.3f}" for row in rows],
    )
    return [{"text": row["text"], "file": row["file"]} for row in rows]
