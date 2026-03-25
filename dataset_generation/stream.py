from __future__ import annotations

import io
from collections.abc import Iterator
from typing import BinaryIO

import zstandard

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
