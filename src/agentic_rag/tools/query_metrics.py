"""Text-to-SQL over the structured financial metrics table.

The model writes a SQL query from the user's question; we execute it read-only
against the SQLite database and hand the rows back.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from agentic_rag.config import CHAT_MODEL
from agentic_rag.tools.base import Tool, ToolContext, string_params

logger = logging.getLogger(__name__)

# A column with at most this many distinct values is listed in full for the
# model, so it filters on values that actually exist ('Alphabet', not 'Google').
MAX_DISTINCT_VALUES = 25

SQL_SYSTEM_PROMPT = """\
You are a SQL expert. Translate the user's question into a single SQLite SELECT
statement against the schema below. Return only data that answers the question.

Rules:
- SELECT statements only. Never write, update or delete.
- Use exact column names from the schema and exact values from the value lists.
- If the user names an entity that is not in a value list, map it to the closest
  listed value rather than inventing one.
- Prefer aggregates (SUM, AVG) when the question asks for totals or averages.
- If the question spans several rows, return the rows rather than guessing a total.

Schema:
{schema}

Known column values:
{values}
"""


class GeneratedQuery(BaseModel):
    """Structured response we ask the model for."""

    sql_query: str = Field(description="A single, raw, executable SQLite SELECT statement")
    explanation: str = Field(description="One sentence on why this query answers the question")


def _read_schema(db_path: Path, table: str) -> str:
    """Return the CREATE TABLE statement so the model sees the real schema."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
    if not row:
        raise LookupError(f"Table '{table}' not found in {db_path}")
    return row[0]


def _column_values(db_path: Path, table: str) -> str:
    """Describe the distinct values of every low-cardinality column.

    Without this the model guesses labels from the question ("Google") instead of
    using the ones in the data ("Alphabet"), and every query comes back empty.
    """
    lines: list[str] = []
    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
        for column in columns:
            values = [
                row[0]
                for row in conn.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f"ORDER BY 1 LIMIT {MAX_DISTINCT_VALUES + 1}"
                )
            ]
            if len(values) <= MAX_DISTINCT_VALUES:
                lines.append(f"- {column}: {', '.join(repr(v) for v in values)}")
            else:
                lines.append(f"- {column}: many distinct values")
    return "\n".join(lines)


def _is_read_only(sql: str) -> bool:
    """Allow a single SELECT (or CTE) statement and nothing else."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:  # no statement chaining
        return False
    return stripped.lower().startswith(("select", "with"))


def _execute(db_path: Path, sql: str) -> list[dict[str, Any]]:
    """Run ``sql`` against a read-only connection and return rows as dicts."""
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def query_metrics(ctx: ToolContext, search_query: str) -> Any:
    """Answer a quantitative question by generating and running SQL."""
    db_path = ctx.settings.metrics_db
    if not db_path.exists():
        return (
            f"The metrics database is missing at {db_path}. Run `agentic-rag ingest` to build it."
        )

    try:
        schema = _read_schema(db_path, ctx.settings.metrics_table)
        values = _column_values(db_path, ctx.settings.metrics_table)
    except (LookupError, sqlite3.Error) as exc:
        return f"Could not read the metrics schema: {exc}"

    try:
        response = ctx.client.models.generate_content(
            model=CHAT_MODEL,
            contents=search_query,
            config=genai.types.GenerateContentConfig(
                system_instruction=SQL_SYSTEM_PROMPT.format(schema=schema, values=values),
                response_mime_type="application/json",
                response_json_schema=GeneratedQuery.model_json_schema(),
            ),
        )
        generated = GeneratedQuery.model_validate_json(response.text)
    except Exception as exc:
        logger.warning("SQL generation failed: %s", exc)
        return f"Could not generate a SQL query: {exc}"

    logger.info("generated SQL: %s", generated.sql_query)

    if not _is_read_only(generated.sql_query):
        return "Refused to run a non-SELECT query against the metrics database."

    try:
        rows = _execute(db_path, generated.sql_query)
    except sqlite3.Error as exc:
        return f"SQL error: {exc} (query was: {generated.sql_query})"

    result: dict[str, Any] = {
        "sql_query": generated.sql_query,
        "explanation": generated.explanation,
        "row_count": len(rows),
        "rows": rows,
    }
    if not rows:
        # Tell the model why it got nothing, so it rephrases instead of
        # re-running near-identical queries until the loop limit trips.
        result["note"] = (
            "No rows matched. The filters may not match the stored values. "
            f"Available values:\n{values}"
        )
    return result


TOOL = Tool(
    name="query_metrics_tool",
    description=(
        "Answer quantitative questions about company financials (revenue and "
        "operating income by company, fiscal year, quarter and business segment) "
        "by running SQL against the metrics database. Use this for numbers, "
        "totals, comparisons and trends rather than searching documents."
    ),
    parameters=string_params(
        search_query="The quantitative question in plain English, e.g. "
        "'total Google Cloud revenue in fiscal 2025'."
    ),
    handler=query_metrics,
)
