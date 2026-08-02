"""Colour palettes and layout helpers for the terminal interface.

Every colour the app uses is named — ``accent``, ``agent``, ``tool`` and so on
— and each palette binds those names to hues. Switching palettes therefore
recolours everything without touching a line of interface code.

Hues are hex so they look the same across terminals; Rich downgrades them for
256-colour and 16-colour terminals automatically. De-emphasis always uses
``dim`` rather than a fixed grey, so it stays readable on light backgrounds.

Pick one with ``--theme <name>`` or ``THEME=<name>`` in ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

load_dotenv()


@dataclass(frozen=True)
class Palette:
    """Six hues, which is all the interface needs.

    Attributes:
        accent: Structure — banner border, table headings, spinner, progress.
        user: The user's input prompt.
        agent: The answer panel's border and title.
        tool: The tool-call trace.
        warn: Warnings and skipped files.
        error: Failures.
    """

    accent: str
    user: str
    agent: str
    tool: str
    warn: str
    error: str

    def as_theme(self) -> Theme:
        return Theme(
            {
                "accent": self.accent,
                "heading": f"bold {self.accent}",
                "user": f"bold {self.user}",
                "agent": f"bold {self.agent}",
                "agent.border": self.agent,
                "tool": self.tool,
                "warn": self.warn,
                "error": f"bold {self.error}",
                "muted": "dim",
            }
        )


PALETTES: dict[str, Palette] = {
    # https://www.nordtheme.com — cool and low contrast
    "nord": Palette(
        accent="#88c0d0",
        user="#8fbcbb",
        agent="#a3be8c",
        tool="#5e81ac",
        warn="#ebcb8b",
        error="#bf616a",
    ),
    # https://catppuccin.com (Mocha) — soft pastels, gentle on the eyes
    "catppuccin": Palette(
        accent="#89b4fa",
        user="#94e2d5",
        agent="#a6e3a1",
        tool="#7f849c",
        warn="#f9e2af",
        error="#f38ba8",
    ),
    # https://github.com/morhetz/gruvbox — warm and earthy
    "gruvbox": Palette(
        accent="#83a598",
        user="#8ec07c",
        agent="#b8bb26",
        tool="#928374",
        warn="#fabd2f",
        error="#fb4934",
    ),
    # https://ethanschoonover.com/solarized — the classic, works on light terminals
    "solarized": Palette(
        accent="#268bd2",
        user="#2aa198",
        agent="#859900",
        tool="#657b83",
        warn="#b58900",
        error="#dc322f",
    ),
    # https://draculatheme.com — high contrast, vivid
    "dracula": Palette(
        accent="#8be9fd",
        user="#50fa7b",
        agent="#bd93f9",
        tool="#6272a4",
        warn="#f1fa8c",
        error="#ff5555",
    ),
}

DEFAULT_THEME = "nord"


def resolve_theme(name: str | None) -> Palette:
    """Return the named palette, falling back to the default if unknown."""
    return PALETTES.get((name or "").lower().strip() or DEFAULT_THEME, PALETTES[DEFAULT_THEME])


console = Console(theme=resolve_theme(os.getenv("THEME")).as_theme())


def apply_theme(name: str | None) -> None:
    """Recolour the console, for the ``--theme`` flag."""
    if name:
        console.push_theme(resolve_theme(name).as_theme())


# Prose is hard to read across a very wide terminal, so answers are capped
# rather than stretched to the full window.
MAX_CONTENT_WIDTH = 96

# Below this width the banner and hints switch to their compact form.
NARROW_WIDTH = 72


def content_width() -> int:
    """Width for prose blocks: the terminal, capped for readability."""
    return min(console.width, MAX_CONTENT_WIDTH)


def is_narrow() -> bool:
    """True in a small window, where compact layouts read better."""
    return console.width < NARROW_WIDTH


def shorten_path(path: Path | str, limit: int | None = None) -> str:
    """Render a path compactly: home as ``~``, middle elided if still too long.

    Long absolute paths otherwise wrap across several lines and break the
    layout of panels and table titles.
    """
    text = str(path)
    try:
        home = str(Path.home())
        if text.startswith(home):
            text = "~" + text[len(home) :]
    except (OSError, RuntimeError):  # no resolvable home directory
        pass

    limit = limit or max(24, console.width - 24)
    if len(text) <= limit:
        return text

    parts = Path(text).parts
    tail = Path(*parts[-2:]) if len(parts) > 2 else Path(text).name
    return f"{parts[0]}…/{tail}"


def truncate(value: object, limit: int = 72) -> str:
    """Shorten a tool argument for the one-line call trace."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
