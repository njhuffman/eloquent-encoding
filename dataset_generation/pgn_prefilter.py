"""
Fast header-only checks for PGN streams (no python-chess).

Aligned with builder stratum / time-control semantics and
scripts/pgn_zst_white_elo_histogram.py regex style.
"""

from __future__ import annotations

import re

from dataset_generation.recipe import Recipe, SourcePlan

_TIME_CONTROL_RE = re.compile(r'\[TimeControl\s+"([^"]*)"\]', re.IGNORECASE)
_WHITE_ELO_RE = re.compile(r'\[WhiteElo\s+"(\d+)"\]', re.IGNORECASE)
_BLACK_ELO_RE = re.compile(r'\[BlackElo\s+"(\d+)"\]', re.IGNORECASE)


def elo_pair_from_tag_strings(
    white_raw: str | None, black_raw: str | None
) -> tuple[int | None, int | None]:
    """Digits-only tag groups from regex, or None if missing."""
    w = int(white_raw) if white_raw is not None else None
    b = int(black_raw) if black_raw is not None else None
    return w, b


def parse_header_tag_line(line: str, state: dict[str, str | None]) -> None:
    """Update ``state`` keys ``time_control``, ``white_elo``, ``black_elo`` from one tag line."""
    stripped = line.strip()
    if not stripped.startswith("["):
        return
    m = _TIME_CONTROL_RE.match(stripped)
    if m:
        state["time_control"] = m.group(1)
        return
    m = _WHITE_ELO_RE.search(stripped)
    if m:
        state["white_elo"] = m.group(1)
        return
    m = _BLACK_ELO_RE.search(stripped)
    if m:
        state["black_elo"] = m.group(1)


def header_section_ended(line: str, saw_tag_line: bool) -> bool:
    """True when movetext starts or blank line ends the header block (standard PGN)."""
    if not saw_tag_line:
        return False
    s = line.strip()
    if s == "":
        return True
    return not s.startswith("[")


def time_control_matches(parsed: str | None, required: str | None) -> bool:
    if required is None:
        return True
    return parsed == required


def game_matches_stratum(
    white: int | None,
    black: int | None,
    bucket_by: str,
    lo: int,
    hi: int,
) -> bool:
    if white is None or black is None:
        return False
    if bucket_by == "white":
        return lo <= white <= hi
    if bucket_by == "black":
        return lo <= black <= hi
    if bucket_by == "both":
        return lo <= white <= hi and lo <= black <= hi
    raise ValueError(bucket_by)


def any_unfilled_stratum_may_match(
    white: int | None,
    black: int | None,
    *,
    recipe: Recipe,
    plan: SourcePlan,
    accepted: list[int],
) -> bool:
    """
    True iff some stratum still needs games and this (white, black) pair could match it.
    Mirrors builder logic before expensive board walks / parsing.
    """
    if white is None or black is None:
        return False
    for s, st in enumerate(plan.strata):
        if accepted[s] >= st.take_games:
            continue
        if game_matches_stratum(
            white, black, recipe.bucket_by, st.elo_min, st.elo_max
        ):
            return True
    return False


def all_strata_quotas_met(plan: SourcePlan, accepted: list[int]) -> bool:
    """True when every stratum has reached its ``take_games`` quota."""
    return all(accepted[s] >= plan.strata[s].take_games for s in range(len(plan.strata)))


def passes_header_prefilter(
    *,
    parsed_tc: str | None,
    white_elo: int | None,
    black_elo: int | None,
    recipe: Recipe,
    plan: SourcePlan,
    accepted: list[int],
) -> bool:
    """Whether to parse movetext with python-chess for this game."""
    if not time_control_matches(parsed_tc, recipe.time_control):
        return False
    return any_unfilled_stratum_may_match(
        white_elo, black_elo, recipe=recipe, plan=plan, accepted=accepted
    )
