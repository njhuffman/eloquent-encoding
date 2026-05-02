from __future__ import annotations

import io
from collections.abc import Iterator
from typing import BinaryIO

import zstandard

from dataset_generation.pgn_prefilter import (
    all_strata_quotas_met,
    elo_pair_from_tag_strings,
    header_section_ended,
    parse_header_tag_line,
    passes_header_prefilter,
)
from dataset_generation.recipe import Recipe, SourcePlan

# Lichess PGN: each game starts with [Event
_EVENT_PREFIX = "[Event"


def iter_pgn_game_texts(text_stream: io.TextIOBase) -> Iterator[str]:
    """
    Split a PGN text stream into one string per game (headers + movetext).
    Uses [Event at line start as delimiter, same idea as scripts/pgn_zst_white_elo_histogram.py.
    """
    buf: list[str] = []
    seen_first = False
    for line in text_stream:
        if line.lstrip().startswith(_EVENT_PREFIX):
            if seen_first:
                yield "".join(buf)
                buf = []
            seen_first = True
        buf.append(line)
    if buf:
        yield "".join(buf)


def iter_pgn_games_from_zstd_binary(raw: BinaryIO) -> Iterator[str]:
    """Decompress zstd from a binary stream and yield PGN game strings."""
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(raw)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    yield from iter_pgn_game_texts(text)


def iter_filtered_pgn_game_texts_from_zstd(
    raw: BinaryIO,
    *,
    recipe: Recipe,
    plan: SourcePlan,
    accepted: list[int],
) -> Iterator[str]:
    """
    Like ``iter_pgn_games_from_zstd_binary`` but skips python-chess parsing for games that
    cannot match any unfilled stratum (header regex only) or fail time control.

    ``accepted`` is the mutable per-stratum accepted-game counts for the current plan;
    it must be the same list passed to the builder loop so quotas stay consistent.
    """
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(raw)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    yield from _iter_filtered_pgn_game_texts(text, recipe=recipe, plan=plan, accepted=accepted)


def _iter_filtered_pgn_game_texts(
    text_stream: io.TextIOBase,
    *,
    recipe: Recipe,
    plan: SourcePlan,
    accepted: list[int],
) -> Iterator[str]:
    """Line-based: regex headers, skip movetext unless prefilter passes."""
    mode = "seek_event"
    buf: list[str] = []
    header_tags: dict[str, str | None] = {
        "time_control": None,
        "white_elo": None,
        "black_elo": None,
    }
    saw_tag_line = False

    for line in text_stream:
        # Parent updates ``accepted`` only between yields; once full, exit immediately
        # so we do not decompress/scan the rest of the .pgn.zst (would otherwise happen
        # while skipping games that fail the header prefilter).
        if all_strata_quotas_met(plan, accepted):
            return
        ls = line.lstrip()
        if ls.startswith(_EVENT_PREFIX):
            if mode == "buffer_game":
                yield "".join(buf)
            buf = [line]
            header_tags = {"time_control": None, "white_elo": None, "black_elo": None}
            saw_tag_line = False
            mode = "headers"
            continue

        if mode == "seek_event":
            continue

        if mode == "headers":
            buf.append(line)
            if line.strip().startswith("["):
                saw_tag_line = True
                parse_header_tag_line(line, header_tags)
            if header_section_ended(line, saw_tag_line):
                w_elo, b_elo = elo_pair_from_tag_strings(
                    header_tags.get("white_elo"),
                    header_tags.get("black_elo"),
                )
                if passes_header_prefilter(
                    parsed_tc=header_tags.get("time_control"),
                    white_elo=w_elo,
                    black_elo=b_elo,
                    recipe=recipe,
                    plan=plan,
                    accepted=accepted,
                ):
                    mode = "buffer_game"
                else:
                    mode = "skip_game"
                    buf = []
            continue

        if mode == "buffer_game":
            buf.append(line)
            continue

        if mode == "skip_game":
            continue

    if mode == "buffer_game" and buf:
        yield "".join(buf)
