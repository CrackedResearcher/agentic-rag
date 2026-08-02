"""Build the agent's knowledge stores from a corpus directory.

Two artifacts are produced under ``data/``:

* ``vector_table.lance`` — chunked, embedded text for semantic search.
* ``financials.db``      — a metrics CSV loaded into SQLite for the SQL tool.

Plus ``manifest.json``, recording what was indexed so the CLI can show it.

All three are derived data. Delete them and re-run ingest to rebuild.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lancedb
from google import genai

from agentic_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, IGNORED_DIRS, Settings
from agentic_rag.loaders import MissingDependency, is_supported, load
from agentic_rag.retrieval import embed_document

logger = logging.getLogger(__name__)

# Small pause between embedding calls to stay inside free-tier rate limits.
EMBED_DELAY_SECONDS = 0.22

#: Called with (file_name, chunks_done, chunks_total) so the CLI can draw progress.
#: ``chunks_total`` is the total across the whole corpus, known before embedding
#: starts because text extraction happens in a first pass.
ProgressCallback = Callable[[str, int, int], None]


@dataclass
class IngestReport:
    """What ingest actually did, for display and for the manifest."""

    docs_dir: Path
    indexed: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    metric_rows: int = 0

    @property
    def chunk_count(self) -> int:
        return sum(entry["chunks"] for entry in self.indexed)

    @property
    def file_count(self) -> int:
        return len(self.indexed)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into overlapping windows.

    The overlap keeps sentences that straddle a boundary retrievable from both
    chunks.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    stride = size - overlap
    return [text[start : start + size] for start in range(0, max(len(text), 1), stride)]


def iter_documents(docs_dir: Path) -> Iterator[Path]:
    """Yield every loadable document under ``docs_dir``, recursively.

    Hidden files and well-known junk directories (``.git``, ``node_modules``,
    virtualenvs) are skipped so pointing the tool at a project folder does
    something sensible.
    """
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(docs_dir)
        if IGNORED_DIRS.intersection(relative.parts):
            continue
        if is_supported(path):
            yield path


def extract_chunks(docs_dir: Path, report: IngestReport) -> list[tuple[str, str]]:
    """Read and chunk every document, recording anything that had to be skipped.

    Runs before any embedding so the total chunk count — and therefore the cost
    of the run — is known up front.

    Returns:
        ``(file_label, chunk_text)`` pairs in corpus order.
    """
    pairs: list[tuple[str, str]] = []

    for path in iter_documents(docs_dir):
        label = str(path.relative_to(docs_dir))
        try:
            text = load(path)
        except (MissingDependency, OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", label, exc)
            report.skipped.append({"file": label, "reason": str(exc)})
            continue

        if not text.strip():
            report.skipped.append({"file": label, "reason": "no extractable text"})
            continue

        chunks = chunk_text(text)
        pairs.extend((label, chunk) for chunk in chunks)
        report.indexed.append({"file": label, "chunks": len(chunks)})
        logger.info("read %s (%d chunks)", label, len(chunks))

    return pairs


def build_vector_table(
    client: genai.Client,
    settings: Settings,
    report: IngestReport,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Embed every document chunk and (re)create the LanceDB table."""
    docs_dir = settings.docs_dir
    pairs = extract_chunks(docs_dir, report)

    if not pairs:
        raise RuntimeError(
            f"No supported documents found in {docs_dir}. "
            "Run `agentic-rag formats` to see which file types are supported."
        )

    total = len(pairs)
    records: list[dict[str, Any]] = []

    for index, (label, chunk) in enumerate(pairs, start=1):
        records.append(
            {
                "id": len(records),
                "file": label,
                "text": chunk,
                "vector": embed_document(chunk, client),
            }
        )
        if on_progress:
            on_progress(label, index, total)
        time.sleep(EMBED_DELAY_SECONDS)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(settings.data_dir)
    db.create_table(settings.vector_table, data=records, mode="overwrite")
    logger.info("indexed %d chunks into '%s'", len(records), settings.vector_table)


def build_metrics_db(settings: Settings) -> int:
    """Load the metrics CSV into SQLite, replacing any existing table.

    Returns the number of rows loaded, or 0 when the corpus has no metrics CSV —
    the SQL tool then reports itself unavailable rather than failing.
    """
    csv_path = settings.metrics_csv
    if not csv_path.exists():
        logger.info("no %s in the corpus — skipping the metrics database", csv_path.name)
        return 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        logger.warning("%s is empty — skipping the metrics database", csv_path)
        return 0

    columns = list(rows[0])
    table = settings.metrics_table
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    definitions = ", ".join(f'"{c}" {_sqlite_type(rows, c)}' for c in columns)
    placeholders = ", ".join("?" for _ in columns)

    with sqlite3.connect(settings.metrics_db) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" ({definitions})')
        conn.executemany(
            f'INSERT INTO "{table}" VALUES ({placeholders})',
            [tuple(_coerce(row[c]) for c in columns) for row in rows],
        )

    logger.info("loaded %d rows into %s (%s)", len(rows), settings.metrics_db.name, table)
    return len(rows)


def _coerce(value: str) -> Any:
    """Turn a CSV string into an int when it looks like one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _sqlite_type(rows: list[dict[str, str]], column: str) -> str:
    """INTEGER when every value in the column parses as one, otherwise TEXT."""
    return "INTEGER" if all(isinstance(_coerce(r[column]), int) for r in rows) else "TEXT"


def write_manifest(settings: Settings, report: IngestReport) -> None:
    """Record what was indexed, so `agentic-rag sources` can show it later."""
    settings.manifest_path.write_text(
        json.dumps(
            {
                "docs_dir": str(report.docs_dir),
                "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "chunk_count": report.chunk_count,
                "metric_rows": report.metric_rows,
                "files": report.indexed,
                "skipped": report.skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_manifest(settings: Settings) -> dict[str, Any] | None:
    """Return the last ingest manifest, or None if the corpus was never built."""
    if not settings.manifest_path.exists():
        return None
    try:
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def ingest(
    client: genai.Client,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Rebuild every knowledge store from ``settings.docs_dir``."""
    if not settings.docs_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {settings.docs_dir}")

    report = IngestReport(docs_dir=settings.docs_dir)
    report.metric_rows = build_metrics_db(settings)
    build_vector_table(client, settings, report, on_progress)
    write_manifest(settings, report)
    return report
