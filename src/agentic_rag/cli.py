"""Command line interface: an interactive chat plus corpus management commands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from google import genai
from rich.box import SIMPLE
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agentic_rag.agent import Agent
from agentic_rag.config import ConfigError, Settings
from agentic_rag.config import settings as default_settings
from agentic_rag.evals import (
    DEFAULT_CASES,
    CaseResult,
    Check,
    EvalCase,
    RunRecord,
    build_record,
    compare,
    history_path,
    load_cases,
    load_history,
    run_cases,
    save_run,
    summarise,
)
from agentic_rag.ingest import ingest, read_manifest
from agentic_rag.loaders import LOADERS, OPTIONAL_FORMATS, supported_extensions
from agentic_rag.retrieval import open_table
from agentic_rag.theme import (
    PALETTES,
    apply_theme,
    console,
    content_width,
    is_narrow,
    shorten_path,
    truncate,
)
from agentic_rag.tools import REGISTRY

SLASH_HELP = {
    "/help": "show this help",
    "/sources": "list the indexed documents",
    "/tools": "list the tools the agent can call",
    "/formats": "list supported file formats",
    "/clear": "start a fresh conversation",
    "/exit": "quit",
}


def _build_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.require_gemini_key())


def _configure_logging(verbose: bool) -> None:
    # Chat prints its own progress, so only warnings reach the console unless
    # the user asks for detail with -v.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _new_table(*columns: str, **kwargs: object) -> Table:
    """A table styled consistently and sized to the terminal."""
    table = Table(box=SIMPLE, header_style="heading", padding=(0, 2), expand=False, **kwargs)
    for column in columns:
        table.add_column(column)
    return table


def _banner(settings: Settings) -> None:
    manifest = read_manifest(settings)
    narrow = is_narrow()

    header = Text()
    header.append("agentic-rag\n", style="heading")
    header.append(f"{settings.assistant_name} · {settings.organization}", style="default")

    if manifest:
        stats = f"{len(manifest['files'])} documents · {manifest['chunk_count']} chunks"
        if manifest.get("metric_rows"):
            stats += f" · {manifest['metric_rows']} metric rows"
        header.append(f"\n{stats}", style="muted")
        # The corpus path is the first thing to go in a small window.
        if not narrow:
            header.append(f"\n{shorten_path(manifest['docs_dir'])}", style="muted")
    else:
        header.append("\nno corpus indexed yet — run `agentic-rag ingest`", style="warn")

    console.print(
        Panel(
            header,
            border_style="accent",
            padding=(0, 2) if narrow else (1, 3),
            width=content_width(),
        )
    )
    # Full command list only when it fits on one line; otherwise a pointer.
    hints = "/help for commands" if narrow else "   ".join(SLASH_HELP)
    console.print(Padding(Text(hints, style="muted"), (0, 0, 1, 2)))


def _render_tool_call(name: str, args: dict[str, object]) -> None:
    # Argument values are truncated so a long query stays on one line.
    rendered = ", ".join(f"{key}={truncate(value)}" for key, value in args.items())
    # no_wrap/overflow have to be passed to print — they are ignored when set
    # on the Text alone — so a long query elides instead of wrapping.
    console.print(Text(f"  ↳ {name}({rendered})", style="tool"), no_wrap=True, overflow="ellipsis")


def _render_answer(settings: Settings, text: str) -> None:
    console.print(
        Panel(
            Markdown(text or "_(no answer)_"),
            title=Text(settings.assistant_name, style="agent"),
            title_align="left",
            border_style="agent.border",
            padding=(0, 1),
            width=content_width(),
        )
    )
    console.print()


def _sources_table(settings: Settings) -> None:
    manifest = read_manifest(settings)
    if not manifest:
        console.print(Text("No corpus indexed yet. Run `agentic-rag ingest`.", style="error"))
        return

    table = _new_table("document", "chunks")
    table.columns[0].overflow = "fold"
    table.columns[1].justify = "right"
    # The path goes in the caption, where wrapping is harmless, and is elided
    # to fit rather than shredding the layout of a narrow window.
    indexed_at = manifest["indexed_at"].replace("T", " ")[:16]
    table.caption = f"{shorten_path(manifest['docs_dir'])} · indexed {indexed_at}"
    table.caption_style = "muted"

    for entry in manifest["files"]:
        table.add_row(entry["file"], str(entry["chunks"]))
    table.add_section()
    table.add_row(
        Text(f"{len(manifest['files'])} files", style="bold"),
        Text(str(manifest["chunk_count"]), style="bold"),
    )
    console.print(table)

    if manifest.get("skipped"):
        skipped = _new_table("skipped", "reason", title="Skipped", title_style="warn")
        skipped.columns[0].style = "warn"
        skipped.columns[1].style = "muted"
        skipped.columns[1].overflow = "fold"
        for entry in manifest["skipped"]:
            skipped.add_row(entry["file"], entry["reason"])
        console.print(skipped)
    console.print()


def _tools_table() -> None:
    table = _new_table("tool", "what it does")
    table.columns[0].style = "accent"
    table.columns[0].no_wrap = True
    # Descriptions wrap into whatever room is left after the tool names.
    table.columns[1].overflow = "fold"
    for tool in REGISTRY.values():
        table.add_row(tool.name, tool.description)
    console.print(table)
    console.print()


def _themes_preview() -> None:
    """Show each palette's colours so a theme can be chosen by eye."""
    table = _new_table("theme", "accent", "you", "agent", "tool", "warn", "error")
    table.columns[0].style = "accent"
    for name, palette in PALETTES.items():
        table.add_row(
            name,
            *(
                Text("████", style=hue)
                for hue in (
                    palette.accent,
                    palette.user,
                    palette.agent,
                    palette.tool,
                    palette.warn,
                    palette.error,
                )
            ),
        )
    console.print(table)
    console.print(
        Text("Use with: agentic-rag --theme <name> chat   (or THEME=<name> in .env)", style="muted")
    )
    console.print()


def _formats_table() -> None:
    table = _new_table("extension", "requires")
    table.columns[0].style = "accent"
    for extension in supported_extensions():
        package = OPTIONAL_FORMATS.get(extension)
        table.add_row(
            extension,
            Text(f"{package}  (uv sync --extra docs)", style="warn")
            if package
            else Text("built in", style="muted"),
        )
    console.print(table)
    console.print(
        Text(
            f"{len(LOADERS)} formats. Add one by registering a loader in "
            "src/agentic_rag/loaders.py",
            style="muted",
        )
    )
    console.print()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def run_chat(settings: Settings, show_tools: bool) -> None:
    """Start an interactive session with the agent."""
    client = _build_client(settings)
    agent = Agent(client=client, settings=settings, table=open_table(settings))

    _banner(settings)

    while True:
        try:
            prompt = console.input("[user]you ›[/user] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        if not prompt:
            continue

        if prompt.startswith("/"):
            if _handle_command(prompt, settings, agent):
                return
            continue

        _answer(agent, settings, prompt, show_tools)


def _handle_command(prompt: str, settings: Settings, agent: Agent) -> bool:
    """Run a slash command. Returns True when the session should end."""
    command = prompt.split()[0].lower()

    if command in {"/exit", "/quit", "/q"}:
        return True
    if command == "/sources":
        _sources_table(settings)
    elif command == "/tools":
        _tools_table()
    elif command == "/formats":
        _formats_table()
    elif command == "/clear":
        agent.reset()
        console.print(Rule("new conversation", style="muted", characters="─"))
        console.print()
    elif command == "/help":
        table = _new_table("command", "description")
        table.columns[0].style = "accent"
        table.columns[0].no_wrap = True
        for name, description in SLASH_HELP.items():
            table.add_row(name, description)
        console.print(table)
        console.print()
    else:
        console.print(Text(f"Unknown command {command}. Try /help.", style="error"))
    return False


def _answer(agent: Agent, settings: Settings, prompt: str, show_tools: bool) -> None:
    """Run one turn, showing tool calls as they happen."""
    console.print()
    try:
        with console.status(
            Text("thinking…", style="muted"), spinner="dots", spinner_style="accent"
        ) as status:
            for event, payload in agent.send(prompt):
                if event == "tool":
                    name, args = payload
                    if show_tools:
                        _render_tool_call(name, args)
                    status.update(Text(f"running {name}…", style="muted"))
                elif event == "answer":
                    status.stop()
                    _render_answer(settings, payload)
    except KeyboardInterrupt:
        console.print(Text("interrupted", style="error"))
        console.print()
    except Exception as exc:  # one bad turn shouldn't end the session
        console.print(Text(f"Something went wrong: {exc}", style="error"))
        console.print()


def run_eval(settings: Settings, cases_path: Path, only: list[str] | None) -> bool:
    """Run the golden set and print a per-dimension scorecard.

    Returns True when everything passed, so the command can exit non-zero in CI.
    """
    cases = load_cases(cases_path)
    partial = bool(only)
    if only:
        wanted = set(only)
        unknown = wanted - {case.id for case in cases}
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in wanted]

    agent = Agent(client=_build_client(settings), settings=settings, table=open_table(settings))

    console.print(
        Text("Evaluating ", style="heading")
        + Text(f"{len(cases)} cases", style="default")
        + Text(f"  ({shorten_path(cases_path)})", style="muted")
    )
    console.print()

    # Failure detail goes below the table, not in a column — squeezed into a
    # fifth column it wraps to a few characters wide and becomes unreadable.
    table = _new_table("case", "routing", "retrieval", "answer")
    table.columns[0].no_wrap = True

    results = []
    with Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[muted]{task.description}"),
        BarColumn(complete_style="accent", finished_style="agent.border"),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("running…", total=len(cases))

        def on_wait(case: EvalCase, seconds: float) -> None:
            # Free-tier keys hit the per-minute quota partway through a run;
            # say so rather than appearing to hang.
            progress.update(task, description=f"rate limited — waiting {seconds:.0f}s")

        for result in run_cases(agent, cases, on_wait):
            results.append(result)
            progress.update(task, advance=1, description=result.case.id)
            table.add_row(*_result_row(result))

    console.print(table)
    _print_failures(results)

    # A filtered run isn't comparable to a full one, so it neither reads from
    # nor writes to the history.
    path = history_path(settings.data_dir)
    previous = None if partial else (load_history(path, limit=1) or [None])[0]
    _print_eval_summary(results, previous)

    if partial:
        console.print(Text("Partial run — not recorded in history.", style="muted"))
        console.print()
    else:
        save_run(path, build_record(results))

    return all(result.passed for result in results)


def _mark(check: Check | None) -> Text:
    """A pass/fail/not-applicable glyph for one dimension."""
    if check is None or check.passed is None:
        return Text("–", style="muted")
    return Text("✓", style="agent.border") if check.passed else Text("✗", style="error")


def _result_row(result: CaseResult) -> tuple[Text, ...]:
    if result.error:
        return (Text(result.case.id, style="error"), *(Text("!", style="error") for _ in range(3)))

    return (
        Text(result.case.id, style="default" if result.passed else "warn"),
        _mark(result.check("routing")),
        _mark(result.check("retrieval")),
        _mark(result.check("answer")),
    )


def _print_failures(results: list[CaseResult]) -> None:
    """List why each failing case failed, with room to actually read it."""
    failures = [r for r in results if not r.passed]
    if not failures:
        return

    table = _new_table("failed", "why")
    table.columns[0].style = "warn"
    table.columns[0].no_wrap = True
    table.columns[1].overflow = "fold"
    table.columns[1].style = "muted"

    for result in failures:
        reason = result.error or "; ".join(c.detail for c in result.checks if c.passed is False)
        table.add_row(result.case.id, reason)
    console.print(table)


def _print_eval_summary(results: list[CaseResult], previous: RunRecord | None) -> None:
    """Scores per dimension, the headline rate, and how it moved since last run."""
    summary = summarise(results)
    passed, total = summary["overall"]
    rate = 100.0 * passed / total if total else 0.0

    console.print(Rule(style="muted"))

    dimensions = Text()
    for name in ("routing", "retrieval", "answer"):
        hit, applicable = summary[name]
        style = "agent.border" if applicable and hit == applicable else "warn"
        dimensions.append(f"{name} {hit}/{applicable}   ", style=style)
    console.print(dimensions)

    headline = Text()
    headline.append(
        f"{passed}/{total} cases passed ", style="agent" if passed == total else "error"
    )
    headline.append(f"({rate:.0f}%)", style="muted")

    if previous:
        delta = passed - previous.passed
        if delta > 0:
            headline.append(f"   ▲ +{delta} ", style="agent.border")
        elif delta < 0:
            headline.append(f"   ▼ {delta} ", style="error")
        else:
            headline.append("   = no change ", style="muted")
        headline.append(f"vs previous run ({previous.rate:.0f}%)", style="muted")
    console.print(headline)

    if previous:
        regressed, fixed = compare(build_record(results), previous)
        if regressed:
            console.print(Text(f"regressed: {', '.join(regressed)}", style="error"))
        if fixed:
            console.print(Text(f"fixed: {', '.join(fixed)}", style="agent.border"))
    console.print()


def _history_table(settings: Settings) -> None:
    """Show past runs so a trend is visible, not just the last comparison."""
    records = load_history(history_path(settings.data_dir), limit=15)
    if not records:
        console.print(Text("No past runs recorded. Run `agentic-rag eval` first.", style="warn"))
        return

    table = _new_table("run", "passed", "rate", "routing", "retrieval", "answer")
    table.columns[0].no_wrap = True
    for index, record in enumerate(records):
        previous = records[index - 1] if index else None
        rate = Text(f"{record.rate:.0f}%")
        if previous:
            delta = record.passed - previous.passed
            rate = Text(f"{record.rate:.0f}%") + Text(
                f"  {'▲' if delta > 0 else '▼' if delta < 0 else '='}"
                f"{f'{delta:+d}' if delta else ''}",
                style="agent.border" if delta > 0 else "error" if delta < 0 else "muted",
            )
        table.add_row(
            record.timestamp.replace("T", " ")[:16],
            f"{record.passed}/{record.total}",
            rate,
            *(
                "{}/{}".format(*record.dimensions.get(name, [0, 0]))
                for name in ("routing", "retrieval", "answer")
            ),
        )
    console.print(table)
    console.print()


def run_ingest(settings: Settings) -> None:
    """Rebuild the vector index and metrics database."""
    # Checked here as well as in ingest() so a bad path fails before the
    # progress bar is drawn.
    if not settings.docs_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {settings.docs_dir}")

    console.print(
        Text("Indexing ", style="heading") + Text(shorten_path(settings.docs_dir), style="muted")
    )
    console.print()

    # The bar and elapsed time are dropped in a narrow window so the file name
    # being embedded stays visible.
    columns = [SpinnerColumn(style="accent"), TextColumn("[muted]{task.description}")]
    if not is_narrow():
        columns += [
            BarColumn(complete_style="accent", finished_style="agent.border"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]

    with Progress(*columns, console=console) as progress:
        task = progress.add_task("reading documents…", total=None)

        def on_progress(file_name: str, done: int, total: int) -> None:
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"embedding {shorten_path(file_name, 40)}",
            )

        report = ingest(_build_client(settings), settings, on_progress)

    console.print()
    console.print(
        Text(f"Indexed {report.file_count} documents ", style="agent")
        + Text(f"({report.chunk_count} chunks)", style="muted")
    )
    if report.metric_rows:
        console.print(
            Text(f"Loaded {report.metric_rows} metric rows into ", style="agent.border")
            + Text(settings.metrics_db.name, style="muted")
        )
    for entry in report.skipped:
        console.print(Text(f"  skipped {entry['file']}: {entry['reason']}", style="warn"))

    console.print()
    console.print(Text("Ready. Start chatting with: agentic-rag chat", style="muted"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="An agent that answers questions from your documents, SQL metrics and the web.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--docs",
        type=Path,
        metavar="DIR",
        help="corpus directory to use (default: ./docs, or $DOCS_DIR)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        metavar="DIR",
        help="where to store the index (default: ./data, or $DATA_DIR)",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(PALETTES),
        help="colour palette (default: nord, or $THEME)",
    )

    commands = parser.add_subparsers(dest="command")
    chat = commands.add_parser("chat", help="start an interactive session (default)")
    chat.add_argument(
        "--hide-tools", action="store_true", help="don't show tool calls as they happen"
    )
    commands.add_parser("ingest", help="index a corpus into the vector store")
    commands.add_parser("sources", help="show what is currently indexed")
    commands.add_parser("formats", help="show supported file formats")
    commands.add_parser("themes", help="preview the colour palettes")

    evaluate = commands.add_parser("eval", help="run the golden-set evaluation")
    evaluate.add_argument(
        "--cases", type=Path, default=DEFAULT_CASES, help=f"case file (default: {DEFAULT_CASES})"
    )
    evaluate.add_argument(
        "--case", action="append", metavar="ID", help="run only this case (repeatable)"
    )
    evaluate.add_argument(
        "--history", action="store_true", help="show past runs instead of running"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    apply_theme(args.theme)
    settings = default_settings.with_overrides(docs_dir=args.docs, data_dir=args.data)

    try:
        if args.command == "ingest":
            run_ingest(settings)
        elif args.command == "sources":
            _sources_table(settings)
        elif args.command == "formats":
            _formats_table()
        elif args.command == "themes":
            _themes_preview()
        elif args.command == "eval":
            if args.history:
                _history_table(settings)
            elif not run_eval(settings, args.cases, args.case):
                return 1
        else:
            run_chat(settings, show_tools=not getattr(args, "hide_tools", False))
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(Text(f"error: {exc}", style="error"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
