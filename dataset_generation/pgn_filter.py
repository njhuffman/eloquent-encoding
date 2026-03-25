"""Header-only PGN checks without loading full games."""

from __future__ import annotations

import re

_TIME_CONTROL_RE = re.compile(r'\[TimeControl\s+"([^"]*)"\]', re.IGNORECASE)


def game_text_time_control(game_text: str) -> str | None:
    m = _TIME_CONTROL_RE.search(game_text)
    return m.group(1) if m else None


def game_text_matches_time_control(game_text: str, required: str) -> bool:
    tc = game_text_time_control(game_text)
    return tc == required
