"""Central configuration: paths, model names and API keys.

Everything the app needs to know about its environment lives here so that no
other module has to reach for ``os.getenv`` or hardcode a path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repository root, resolved from this file: src/agentic_rag/config.py -> ../../..
ROOT_DIR = Path(__file__).resolve().parents[2]

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

# Chunking parameters used when indexing the corpus.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Directory names never descended into when scanning a corpus.
IGNORED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".obsidian"})


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve a directory from the environment, expanding ``~``."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings.

    Attributes:
        docs_dir: Source corpus the agent can search and read.
        data_dir: Generated artifacts (vector index, SQLite database).
        vector_table: Name of the LanceDB table holding document chunks.
        metrics_db: SQLite database backing the SQL tool.
        metrics_table: Table inside ``metrics_db`` holding the metrics.
        assistant_name: Persona the agent introduces itself as.
        organization: Company the agent answers on behalf of.
        gemini_api_key: Key for the Gemini chat + embedding models.
        tavily_api_key: Key for web search. Optional; the web search tool
            reports a friendly error to the model when it is absent.
    """

    docs_dir: Path = field(default_factory=lambda: _path_from_env("DOCS_DIR", ROOT_DIR / "docs"))
    data_dir: Path = field(default_factory=lambda: _path_from_env("DATA_DIR", ROOT_DIR / "data"))
    vector_table: str = "vector_table"
    metrics_table: str = "finance"
    assistant_name: str = field(default_factory=lambda: os.getenv("ASSISTANT_NAME", "Sam"))
    organization: str = field(default_factory=lambda: os.getenv("ORGANIZATION", "Acme Corp"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TV_API_KEY"))

    @property
    def metrics_db(self) -> Path:
        return self.data_dir / "financials.db"

    @property
    def metrics_csv(self) -> Path:
        return self.docs_dir / "financials.csv"

    @property
    def manifest_path(self) -> Path:
        """Record of what was indexed, written by ingest and read by the CLI."""
        return self.data_dir / "manifest.json"

    def with_overrides(
        self, docs_dir: Path | None = None, data_dir: Path | None = None
    ) -> Settings:
        """Return a copy with command-line overrides applied."""
        return replace(
            self,
            docs_dir=docs_dir.expanduser().resolve() if docs_dir else self.docs_dir,
            data_dir=data_dir.expanduser().resolve() if data_dir else self.data_dir,
        )

    def require_gemini_key(self) -> str:
        """Return the Gemini API key or explain how to set it."""
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.gemini_api_key


settings = Settings()
