#!/usr/bin/env python3
"""Extract a per-(player,color) position dataset for unsupervised style clustering
within a single rating band.

The clustering UNIT is ``(player, color)``: a player's moves as White and as Black
are separate units. The output h5 has one row per position (same packed schema the
frozen encoder was trained on, produced by the *exact* reuse of
``collect_candidate_positions`` / ``board_at_ply`` / ``board_to_packed`` /
``legal_from_u64`` / ``legal_to_u64``) plus two extra columns: ``unit_id`` (int32) and
``split`` (int8, 0=train / 1=test). A per-game split holds out a fraction of each unit's
games for a divergence test, and a whole-player "novel" holdout removes some players
entirely from clustering (their units are flagged ``is_novel`` in the units table but
their positions are still extracted for the novel-player test).

Two streaming passes over the (zstd) PGN:
  Pass 1 (header-only line scan, fast): count in-band games per (player,color); decide
          qualifying units (>= min_games), pick novel players, assign stable unit_ids.
  Pass 2 (full parse): re-stream, parse only games where a qualifying unit participates
          in-band, extract every candidate position for the qualifying side(s).

CPU only. Writes rows incrementally (resizable h5 datasets) so RAM stays bounded.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import zstandard
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess.pgn  # noqa: E402

from dataset_generation.candidate_collect import (  # noqa: E402
    board_at_ply,
    collect_candidate_positions,
)
from style_policy.board_encode import (  # noqa: E402
    board_to_packed,
    legal_from_u64,
    legal_to_u64,
)
from style_policy.packed_codec import PACKED_BOARD_LEN  # noqa: E402

_EVENT_PREFIX = "[Event "
_CHUNK = 8192  # h5 dataset chunk (matches PackedBatchWriter / j3 datasets: uncompressed)
_FLUSH_ROWS = 100_000  # buffer this many rows before writing a block


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------
def _tag_value(line: str) -> str | None:
    """Value between the first and last double-quote on a PGN tag line."""
    a = line.find('"')
    b = line.rfind('"')
    if a == -1 or b <= a:
        return None
    return line[a + 1 : b]


def _parse_elo(raw: str | None) -> int | None:
    if raw is None or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _stable_bucket(site: str) -> int:
    """Deterministic 0..999 bucket for a game id (stable across processes/runs)."""
    h = hashlib.md5(site.encode("utf-8")).hexdigest()
    return int(h, 16) % 1000


# ---------------------------------------------------------------------------
# Incremental h5 writer: PackedBatchWriter schema + unit_id + split
# ---------------------------------------------------------------------------
class _StyleWriter:
    _SCALAR = (
        ("from_legal_u64", np.uint64),
        ("to_legal_u64", np.uint64),
        ("from_sq", np.uint8),
        ("to_sq", np.uint8),
        ("promotion", np.uint8),
        ("elo_to_move", np.int16),
        ("opp_elo", np.int16),
        ("result", np.int8),
        ("unit_id", np.int32),
        ("split", np.int8),
    )
    _HIST = ("hist_from", "hist_to", "hist_cap")

    def __init__(self, path: Path) -> None:
        self._f = h5py.File(path, "w")
        self._n = 0
        self._f.create_dataset(
            "packed_pre", shape=(0, PACKED_BOARD_LEN), maxshape=(None, PACKED_BOARD_LEN),
            dtype=np.uint8, chunks=(_CHUNK, PACKED_BOARD_LEN),
        )
        for name, dt in self._SCALAR:
            self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(_CHUNK,))
        for name in self._HIST:
            self._f.create_dataset(
                name, shape=(0, 4), maxshape=(None, 4), dtype=np.int8, chunks=(_CHUNK, 4)
            )
        self._buf: dict[str, list] = {"packed_pre": []}
        for name, _ in self._SCALAR:
            self._buf[name] = []
        for name in self._HIST:
            self._buf[name] = []

    def append(self, *, packed_pre, from_legal_u64, to_legal_u64, from_sq, to_sq,
               promotion, elo_to_move, opp_elo, result, unit_id, split,
               hist_from, hist_to, hist_cap) -> None:
        b = self._buf
        b["packed_pre"].append(np.asarray(packed_pre, dtype=np.uint8).reshape(PACKED_BOARD_LEN))
        b["from_legal_u64"].append(np.uint64(from_legal_u64))
        b["to_legal_u64"].append(np.uint64(to_legal_u64))
        b["from_sq"].append(from_sq)
        b["to_sq"].append(to_sq)
        b["promotion"].append(promotion)
        b["elo_to_move"].append(elo_to_move)
        b["opp_elo"].append(opp_elo)
        b["result"].append(result)
        b["unit_id"].append(unit_id)
        b["split"].append(split)
        b["hist_from"].append(np.asarray(hist_from, dtype=np.int8))
        b["hist_to"].append(np.asarray(hist_to, dtype=np.int8))
        b["hist_cap"].append(np.asarray(hist_cap, dtype=np.int8))
        if len(b["packed_pre"]) >= _FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        m = len(self._buf["packed_pre"])
        if m == 0:
            return
        o = self._n
        d = self._f["packed_pre"]
        d.resize((o + m, PACKED_BOARD_LEN))
        d[o : o + m] = np.asarray(self._buf["packed_pre"], dtype=np.uint8)
        self._buf["packed_pre"].clear()
        for name, _ in self._SCALAR:
            d = self._f[name]
            d.resize((o + m,))
            d[o : o + m] = np.asarray(self._buf[name], dtype=d.dtype)
            self._buf[name].clear()
        for name in self._HIST:
            d = self._f[name]
            d.resize((o + m, 4))
            d[o : o + m] = np.asarray(self._buf[name], dtype=np.int8)
            self._buf[name].clear()
        self._n += m

    def close(self) -> None:
        self.flush()
        self._f.attrs["row_count"] = self._n
        self._f.close()

    @property
    def n(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Pass 1: header-only census
# ---------------------------------------------------------------------------
def pass1_census(pgn: Path, band_min: int, band_max: int) -> dict[tuple[str, int], int]:
    """Count in-band games per (username, color). color 0=White, 1=Black."""
    counts: dict[tuple[str, int], int] = {}
    raw = open(pgn, "rb")
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        cur: dict[str, str | None] = {}
        started = False
        n_games = 0
        pbar = tqdm(desc="pass1 (headers)", unit=" games")

        def finalize(g: dict[str, str | None]) -> None:
            we = _parse_elo(g.get("WhiteElo"))
            be = _parse_elo(g.get("BlackElo"))
            w = g.get("White")
            b = g.get("Black")
            if w is not None and we is not None and band_min <= we < band_max:
                counts[(w, 0)] = counts.get((w, 0), 0) + 1
            if b is not None and be is not None and band_min <= be < band_max:
                counts[(b, 1)] = counts.get((b, 1), 0) + 1

        for line in text:
            if line.startswith(_EVENT_PREFIX):
                if started:
                    finalize(cur)
                    n_games += 1
                    if n_games % 50000 == 0:
                        pbar.update(50000)
                cur = {}
                started = True
                continue
            if not started:
                continue
            if line.startswith('[White "'):
                cur["White"] = _tag_value(line)
            elif line.startswith('[Black "'):
                cur["Black"] = _tag_value(line)
            elif line.startswith('[WhiteElo '):
                cur["WhiteElo"] = _tag_value(line)
            elif line.startswith('[BlackElo '):
                cur["BlackElo"] = _tag_value(line)
        if started:
            finalize(cur)
            n_games += 1
        pbar.update(n_games % 50000)
        pbar.close()
        print(f"pass1: scanned {n_games:,} games", file=sys.stderr)
    finally:
        raw.close()
    return counts


# ---------------------------------------------------------------------------
# Unit table construction
# ---------------------------------------------------------------------------
def build_units(counts, min_games, n_novel, seed):
    """Return (unit_map, units) where unit_map[(user,color)]=unit_id and units is a list
    of dicts with the per-unit table fields. Sorted by (username, color) for determinism."""
    qualifying = sorted(
        [(u, c, n) for (u, c), n in counts.items() if n >= min_games]
    )  # sorted by (username, color)
    qual_players = sorted({u for (u, c, n) in qualifying})

    rng = np.random.default_rng(seed)
    k = min(n_novel, len(qual_players))
    if k > 0:
        chosen = rng.choice(np.array(qual_players, dtype=object), size=k, replace=False)
        novel_players = set(chosen.tolist())
    else:
        novel_players = set()

    unit_map: dict[tuple[str, int], int] = {}
    units = []
    for uid, (user, color, ngames) in enumerate(qualifying):
        unit_map[(user, color)] = uid
        units.append(
            {
                "unit_id": uid,
                "username": user,
                "color": color,
                "n_games": ngames,
                "is_novel": user in novel_players,
                "n_positions": 0,
                "elo_sum": 0,  # accumulated in pass 2, divided at the end
            }
        )
    return unit_map, units, qual_players, novel_players


# ---------------------------------------------------------------------------
# Pass 2: full parse + extraction
# ---------------------------------------------------------------------------
def _iter_qualifying_games(pgn, unit_map, band_min, band_max):
    """Line-level scan: yield ``(game_text, w_ok, b_ok, unit_w, unit_b, site)`` only for
    games where at least one qualifying unit participates in-band. Header parsing is cheap
    line-prefix matching; the full game text is built only for games we will actually parse
    (the vast non-qualifying majority is skipped without allocating movetext)."""
    raw = open(pgn, "rb")
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        buf: list[str] = []
        hdr: dict[str, str | None] = {}
        mode = "seek_event"  # seek_event | headers | buffer | skip
        w_ok = b_ok = False
        unit_w = unit_b = None
        site = ""

        def decide():
            we = _parse_elo(hdr.get("WhiteElo"))
            be = _parse_elo(hdr.get("BlackElo"))
            w = hdr.get("White")
            b = hdr.get("Black")
            uw = unit_map.get((w, 0)) if w is not None else None
            ub = unit_map.get((b, 1)) if b is not None else None
            wok = uw is not None and we is not None and band_min <= we < band_max
            bok = ub is not None and be is not None and band_min <= be < band_max
            return wok, bok, uw, ub, hdr.get("Site", "") or ""

        for line in text:
            if line.startswith(_EVENT_PREFIX):
                if mode == "buffer" and buf:
                    yield "".join(buf), w_ok, b_ok, unit_w, unit_b, site
                buf = [line]
                hdr = {}
                mode = "headers"
                continue
            if mode == "seek_event":
                continue
            if mode == "headers":
                buf.append(line)
                if line.startswith('[White "'):
                    hdr["White"] = _tag_value(line)
                elif line.startswith('[Black "'):
                    hdr["Black"] = _tag_value(line)
                elif line.startswith('[WhiteElo '):
                    hdr["WhiteElo"] = _tag_value(line)
                elif line.startswith('[BlackElo '):
                    hdr["BlackElo"] = _tag_value(line)
                elif line.startswith('[Site "'):
                    hdr["Site"] = _tag_value(line)
                elif line.strip() == "":  # blank line: end of header section
                    w_ok, b_ok, unit_w, unit_b, site = decide()
                    if w_ok or b_ok:
                        mode = "buffer"
                    else:
                        mode = "skip"
                        buf = []
                continue
            if mode == "buffer":
                buf.append(line)
                continue
            # mode == "skip": drop movetext lines until next [Event
        if mode == "buffer" and buf:
            yield "".join(buf), w_ok, b_ok, unit_w, unit_b, site
    finally:
        raw.close()


def pass2_extract(pgn, unit_map, units, writer, band_min, band_max, skip, test_frac, cap=0):
    test_cut = int(round(test_frac * 1000))
    n_parsed = 0
    pbar = tqdm(
        _iter_qualifying_games(pgn, unit_map, band_min, band_max),
        desc="pass2 (extract)", unit=" games",
    )
    for text, w_ok, b_ok, unit_w, unit_b, site in pbar:
        # Per-unit cap: skip the (expensive) full parse when every qualifying side of this
        # game has already reached the cap. Recomputed per game so it kicks in as units fill.
        w_need = w_ok and (cap <= 0 or units[unit_w]["n_positions"] < cap)
        b_need = b_ok and (cap <= 0 or units[unit_b]["n_positions"] < cap)
        if not (w_need or b_need):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        mainline, cands = collect_candidate_positions(
            game, skip_opening_plies=skip, exclude_single_legal_move=False
        )
        if not cands:
            continue
        n_parsed += 1
        split = 1 if _stable_bucket(site) < test_cut else 0
        for ply, stm, elo, opp_elo, result, move, hist in cands:
            if stm == 0 and w_need:
                uid = unit_w
            elif stm == 1 and b_need:
                uid = unit_b
            else:
                continue
            if cap > 0 and units[uid]["n_positions"] >= cap:
                continue  # this unit hit the cap mid-game
            board = board_at_ply(mainline, ply)
            promotion = int(move.promotion) if move.promotion is not None else 0
            writer.append(
                packed_pre=board_to_packed(board),
                from_legal_u64=legal_from_u64(board),
                to_legal_u64=legal_to_u64(board, move.from_square),
                from_sq=int(move.from_square),
                to_sq=int(move.to_square),
                promotion=promotion,
                elo_to_move=int(elo),
                opp_elo=int(opp_elo),
                result=int(result),
                unit_id=uid,
                split=split,
                hist_from=[h[0] for h in hist],
                hist_to=[h[1] for h in hist],
                hist_cap=[h[2] for h in hist],
            )
            units[uid]["n_positions"] += 1
            units[uid]["elo_sum"] += int(elo)
        if n_parsed % 20000 == 0:
            pbar.set_postfix(rows=writer.n, parsed=n_parsed, refresh=False)
    print(f"pass2: fully parsed {n_parsed:,} qualifying games", file=sys.stderr)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_units_npz(path, units):
    unit_id = np.array([u["unit_id"] for u in units], dtype=np.int32)
    username = np.array([u["username"] for u in units], dtype=object)
    color = np.array([u["color"] for u in units], dtype=np.int8)
    n_games = np.array([u["n_games"] for u in units], dtype=np.int32)
    n_positions = np.array([u["n_positions"] for u in units], dtype=np.int64)
    elo_mean = np.array(
        [(u["elo_sum"] / u["n_positions"]) if u["n_positions"] else 0.0 for u in units],
        dtype=np.float32,
    )
    is_novel = np.array([u["is_novel"] for u in units], dtype=bool)
    np.savez(
        path,
        unit_id=unit_id,
        username=username,
        color=color,
        n_games=n_games,
        n_positions=n_positions,
        elo_mean=elo_mean,
        is_novel=is_novel,
    )


def print_summary(units, writer_n):
    n_units = len(units)
    n_novel = sum(1 for u in units if u["is_novel"])
    players = {u["username"] for u in units}
    total_pos = sum(u["n_positions"] for u in units)
    print("\n===== SUMMARY =====")
    print(f"qualifying units : {n_units:,}  (novel: {n_novel:,}, non-novel: {n_units - n_novel:,})")
    print(f"distinct players : {len(players):,}")
    print(f"total positions  : {total_pos:,}  (rows written: {writer_n:,})")


def self_check(h5_path, units, train_test_split_counts):
    print("\n===== SELF-CHECK =====")
    with h5py.File(h5_path, "r") as f:
        assert f["packed_pre"].shape[1] == PACKED_BOARD_LEN, "packed_pre width != 34"
        n = f["packed_pre"].shape[0]
        uid = f["unit_id"][:]
        split = f["split"][:]
        print(f"h5 rows: {n:,}  packed_pre shape: {tuple(f['packed_pre'].shape)}")
        max_uid = len(units) - 1
        assert uid.min() >= 0 and uid.max() <= max_uid, (
            f"unit_id out of range: [{uid.min()},{uid.max()}] vs table 0..{max_uid}"
        )
        # dtype parity with training h5 schema
        for name, dt in (
            ("packed_pre", np.uint8), ("from_sq", np.uint8), ("to_sq", np.uint8),
            ("from_legal_u64", np.uint64), ("to_legal_u64", np.uint64),
            ("hist_from", np.int8), ("hist_to", np.int8), ("hist_cap", np.int8),
            ("elo_to_move", np.int16), ("opp_elo", np.int16), ("result", np.int8),
            ("promotion", np.uint8),
        ):
            assert f[name].dtype == np.dtype(dt), f"{name} dtype {f[name].dtype} != {dt}"
        train_n = int((split == 0).sum())
        test_n = int((split == 1).sum())
        print(f"split rows: train={train_n:,}  test={test_n:,}")

    # per-unit train-position coverage (non-novel units must have >=1 train position)
    zero_train = [
        u for u in units if not u["is_novel"] and train_test_split_counts[u["unit_id"]][0] == 0
    ]
    print(f"non-novel units with 0 train positions: {len(zero_train)}")
    if zero_train:
        for u in zero_train[:10]:
            print(f"  WARN zero-train unit {u['unit_id']} {u['username']} color={u['color']} "
                  f"games={u['n_games']} pos={u['n_positions']}")
    assert not zero_train, "some non-novel units have no train positions"

    # per-split novel vs non-novel breakdown
    nn_train = sum(train_test_split_counts[u["unit_id"]][0] for u in units if not u["is_novel"])
    nn_test = sum(train_test_split_counts[u["unit_id"]][1] for u in units if not u["is_novel"])
    nv_train = sum(train_test_split_counts[u["unit_id"]][0] for u in units if u["is_novel"])
    nv_test = sum(train_test_split_counts[u["unit_id"]][1] for u in units if u["is_novel"])
    print(f"non-novel positions: train={nn_train:,}  test={nn_test:,}")
    print(f"novel     positions: train={nv_train:,}  test={nv_test:,}")

    print("example units:")
    for u in units[:3]:
        print(f"  unit {u['unit_id']}: {u['username']} color={u['color']} "
              f"games={u['n_games']} positions={u['n_positions']} novel={u['is_novel']}")
    print("self-check OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", type=Path,
                    default=Path("/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst"))
    ap.add_argument("--band-min", type=int, default=1500)
    ap.add_argument("--band-max", type=int, default=1600)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--n-novel-players", type=int, default=300)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--skip-opening-plies", type=int, default=4)
    ap.add_argument("--max-positions-per-unit", type=int, default=600,
                    help="cap positions kept per (player,color) unit (0 = uncapped)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/eloquence_bulk/databases/style_1500_2025_01.h5"))
    ap.add_argument("--out-units", type=Path,
                    default=Path("/mnt/eloquence_bulk/databases/style_1500_2025_01_units.npz"))
    args = ap.parse_args()

    print(f"PGN: {args.pgn}")
    print(f"band=[{args.band_min},{args.band_max})  min_games={args.min_games}  "
          f"n_novel={args.n_novel_players}  test_frac={args.test_frac}  "
          f"skip_opening_plies={args.skip_opening_plies}  seed={args.seed}")

    # Pass 1
    counts = pass1_census(args.pgn, args.band_min, args.band_max)
    print(f"pass1: {len(counts):,} (player,color) sides had >=1 in-band game", file=sys.stderr)

    unit_map, units, qual_players, novel_players = build_units(
        counts, args.min_games, args.n_novel_players, args.seed
    )
    print(f"qualifying units: {len(units):,}  qualifying players: {len(qual_players):,}  "
          f"novel players chosen: {len(novel_players):,}")
    if not units:
        raise SystemExit("no qualifying units; aborting")

    # Pass 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = _StyleWriter(args.out)
    try:
        pass2_extract(args.pgn, unit_map, units, writer,
                      args.band_min, args.band_max, args.skip_opening_plies, args.test_frac,
                      cap=args.max_positions_per_unit)
    finally:
        writer.close()

    write_units_npz(args.out_units, units)

    # Per-unit per-split counts (needed for self-check / breakdown) — recomputed from h5.
    with h5py.File(args.out, "r") as f:
        uid = f["unit_id"][:].astype(np.int64)
        split = f["split"][:]
    n_units = len(units)
    train_counts = np.bincount(uid[split == 0], minlength=n_units)
    test_counts = np.bincount(uid[split == 1], minlength=n_units)
    per_unit = {i: [int(train_counts[i]), int(test_counts[i])] for i in range(n_units)}

    print_summary(units, writer.n)
    self_check(args.out, units, per_unit)

    print(f"\nh5   : {args.out}")
    print(f"units: {args.out_units}")


if __name__ == "__main__":
    main()
