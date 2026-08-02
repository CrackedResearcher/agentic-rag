"""A golden-set evaluator for the agent.

Each case is a question plus what should happen when it is asked. Three things
are scored separately, because when a score drops you need to know *which* part
broke:

* **routing** — did the agent reach for the right tools, and avoid the wrong
  ones (searching the web for an internal policy is a failure even if the
  answer is correct);
* **retrieval** — did the expected document actually come back;
* **answer** — does the reply contain the expected facts, or refuse when the
  corpus genuinely has no answer.

Refusal cases matter most. An assistant that answers confidently about
something absent from the corpus is broken in a way happy-path testing never
reveals.

Cases live in TOML — parsed with the standard library, so no extra dependency.
"""

from __future__ import annotations

import json
import logging
import re
import time
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag.agent import Agent
from agentic_rag.config import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_CASES = ROOT_DIR / "evals" / "golden.toml"

# A full run makes several model calls per case, which overruns the free tier's
# per-minute quota. Rather than pacing every run for the slowest possible tier,
# the runner waits and retries only when the API actually pushes back — fast on
# a paid key, self-throttling on a free one.
MAX_QUOTA_RETRIES = 3
FALLBACK_RETRY_WAIT = 30.0
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota")
_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def is_quota_error(exc: Exception) -> bool:
    """True when the API refused the call for rate-limit reasons."""
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def retry_wait_seconds(exc: Exception) -> float:
    """How long to wait — the server's own suggestion when it gives one."""
    match = _RETRY_DELAY_PATTERN.search(str(exc))
    return float(match.group(1)) + 1 if match else FALLBACK_RETRY_WAIT


# Phrases that signal the agent declined rather than guessed. Matching is
# deliberately generous — the point is to catch confident fabrication, not to
# police wording.
REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "doesn't have",
    "does not have",
    "not have that information",
    "no information",
    "not available",
    "couldn't find",
    "could not find",
    "unable to find",
    "not specified",
    "does not specify",
    "doesn't specify",
    "not mentioned",
    "not contain",
    "don't know",
    "no data",
    "isn't covered",
    "is not covered",
)


@dataclass(frozen=True)
class EvalCase:
    """One question and the behaviour expected of the agent."""

    id: str
    question: str
    expect_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    expect_sources: tuple[str, ...] = ()
    expect_contains: tuple[str, ...] = ()
    must_refuse: bool = False


@dataclass
class Check:
    """The outcome of one scored dimension.

    ``passed`` is None when the case makes no claim about this dimension, which
    keeps unrelated expectations from inflating or deflating a score.
    """

    name: str
    passed: bool | None
    detail: str = ""


@dataclass
class CaseResult:
    """Everything observed while running one case."""

    case: EvalCase
    checks: list[Check] = field(default_factory=list)
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    sources_seen: set[str] = field(default_factory=set)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(c.passed is not False for c in self.checks)

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)


def load_cases(path: Path = DEFAULT_CASES) -> list[EvalCase]:
    """Read cases from a TOML file of ``[[case]]`` tables."""
    if not path.exists():
        raise FileNotFoundError(f"No eval cases at {path}")

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            id=entry["id"],
            question=entry["question"],
            expect_tools=tuple(entry.get("expect_tools", ())),
            forbid_tools=tuple(entry.get("forbid_tools", ())),
            expect_sources=tuple(entry.get("expect_sources", ())),
            expect_contains=tuple(entry.get("expect_contains", ())),
            must_refuse=bool(entry.get("must_refuse", False)),
        )
        for entry in raw.get("case", [])
    ]
    if not cases:
        raise ValueError(f"No [[case]] entries found in {path}")

    duplicates = {c.id for c in cases if [x.id for x in cases].count(c.id) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {', '.join(sorted(duplicates))}")
    return cases


def normalise(text: str) -> str:
    """Lower-case and strip thousands separators so '12,260' matches '12260'."""
    return " ".join(text.lower().replace(",", "").split())


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def sources_from(tool_name: str, args: dict[str, Any], result: Any) -> set[str]:
    """Extract the document names a tool call surfaced."""
    if tool_name == "search_document_tool" and isinstance(result, list):
        return {
            Path(str(row["file"])).name
            for row in result
            if isinstance(row, dict) and row.get("file")
        }
    if tool_name == "read_document_tool" and args.get("file_name"):
        return {Path(str(args["file_name"])).name}
    return set()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_routing(case: EvalCase, tools_called: list[str]) -> Check:
    if not case.expect_tools and not case.forbid_tools:
        return Check("routing", None)

    called = set(tools_called)
    missing = [t for t in case.expect_tools if t not in called]
    forbidden = [t for t in case.forbid_tools if t in called]

    if missing or forbidden:
        parts = []
        if missing:
            parts.append(f"never called {', '.join(missing)}")
        if forbidden:
            parts.append(f"called forbidden {', '.join(forbidden)}")
        return Check("routing", False, "; ".join(parts))
    return Check("routing", True)


def score_retrieval(case: EvalCase, sources_seen: set[str]) -> Check:
    if not case.expect_sources:
        return Check("retrieval", None)

    if any(source in sources_seen for source in case.expect_sources):
        return Check("retrieval", True)

    got = ", ".join(sorted(sources_seen)) or "nothing"
    return Check("retrieval", False, f"wanted one of {', '.join(case.expect_sources)}; got {got}")


def score_answer(case: EvalCase, answer: str) -> Check:
    if case.must_refuse:
        if looks_like_refusal(answer):
            return Check("answer", True)
        return Check("answer", False, "should have declined but gave an answer")

    if not case.expect_contains:
        return Check("answer", None)

    normalised = normalise(answer)
    missing = [want for want in case.expect_contains if normalise(want) not in normalised]
    if missing:
        return Check("answer", False, f"missing {', '.join(repr(m) for m in missing)}")
    return Check("answer", True)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def _attempt(agent: Agent, case: EvalCase) -> CaseResult:
    """One pass at a case, on a fresh conversation."""
    result = CaseResult(case=case)
    agent.reset()

    pending_args: dict[str, dict[str, Any]] = {}
    for event, payload in agent.send(case.question):
        if event == "tool":
            name, args = payload
            result.tools_called.append(name)
            pending_args[name] = args
        elif event == "result":
            name, tool_result = payload
            result.sources_seen |= sources_from(name, pending_args.get(name, {}), tool_result)
        elif event == "answer":
            result.answer = payload

    result.checks = [
        score_routing(case, result.tools_called),
        score_retrieval(case, result.sources_seen),
        score_answer(case, result.answer),
    ]
    return result


def run_case(
    agent: Agent,
    case: EvalCase,
    on_wait: Callable[[EvalCase, float], None] | None = None,
) -> CaseResult:
    """Run one case, retrying the whole case if the API rate-limits us.

    Retrying the case rather than the individual call keeps things simple and
    correct: cases are independent and each starts from a reset conversation,
    so a repeat is a clean repeat.
    """
    for attempt in range(MAX_QUOTA_RETRIES + 1):
        try:
            return _attempt(agent, case)
        except Exception as exc:  # a crashed case is a failed case, not a crashed run
            if is_quota_error(exc) and attempt < MAX_QUOTA_RETRIES:
                wait = retry_wait_seconds(exc)
                logger.info("rate limited on %s; waiting %.0fs", case.id, wait)
                if on_wait:
                    on_wait(case, wait)
                time.sleep(wait)
                continue

            logger.exception("case %s raised", case.id)
            message = str(exc).replace("\n", " ")
            return CaseResult(case=case, error=message[:160])

    return CaseResult(case=case, error="exhausted retries")


def run_cases(
    agent: Agent,
    cases: list[EvalCase],
    on_wait: Callable[[EvalCase, float], None] | None = None,
) -> Iterator[CaseResult]:
    """Run every case in order, yielding results as they complete."""
    for case in cases:
        yield run_case(agent, case, on_wait)


def _rate(passed: int, total: int) -> float:
    return 100.0 * passed / total if total else 0.0


@dataclass
class RunRecord:
    """One saved run, so later runs can be compared against it."""

    timestamp: str
    passed: int
    total: int
    dimensions: dict[str, list[int]]
    cases: dict[str, bool]

    @property
    def rate(self) -> float:
        return _rate(self.passed, self.total)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunRecord:
        return cls(
            timestamp=raw["timestamp"],
            passed=raw["passed"],
            total=raw["total"],
            dimensions=raw.get("dimensions", {}),
            cases=raw.get("cases", {}),
        )


def history_path(data_dir: Path) -> Path:
    """Where run history lives — beside the index, so it is per-corpus."""
    return data_dir / "eval-history.jsonl"


def build_record(results: list[CaseResult]) -> RunRecord:
    summary = summarise(results)
    passed, total = summary["overall"]
    return RunRecord(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        passed=passed,
        total=total,
        dimensions={name: list(summary[name]) for name in ("routing", "retrieval", "answer")},
        cases={result.case.id: result.passed for result in results},
    )


def save_run(path: Path, record: RunRecord) -> None:
    """Append a run to the history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json() + "\n")


def load_history(path: Path, limit: int | None = None) -> list[RunRecord]:
    """Read past runs, oldest first. A corrupt line is skipped, not fatal."""
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(RunRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            logger.warning("skipping malformed history line in %s", path)
    return records[-limit:] if limit else records


def compare(current: RunRecord, previous: RunRecord) -> tuple[list[str], list[str]]:
    """Return ``(regressed, fixed)`` case ids between two runs.

    Only cases present in both runs are compared, so adding or removing cases
    from the golden set never shows up as a regression.
    """
    shared = current.cases.keys() & previous.cases.keys()
    regressed = sorted(c for c in shared if previous.cases[c] and not current.cases[c])
    fixed = sorted(c for c in shared if not previous.cases[c] and current.cases[c])
    return regressed, fixed


def summarise(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """Per-dimension ``(passed, applicable)`` tallies, plus an overall count."""
    summary: dict[str, tuple[int, int]] = {}
    for name in ("routing", "retrieval", "answer"):
        scored = [c for r in results if (c := r.check(name)) and c.passed is not None]
        summary[name] = (sum(1 for c in scored if c.passed), len(scored))
    summary["overall"] = (sum(1 for r in results if r.passed), len(results))
    return summary
