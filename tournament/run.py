"""Incremental tournament runner. Plays every unordered bot pair over the opening book in BOTH
colors, skipping games already in results.jsonl (append-only). Adding a bot later ⇒ only its new
pairings are missing ⇒ it plays just those. Games start from balanced openings so bots can play at
their real strength without repeating identical games."""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import chess
from tournament.bots import load_registry, BotFactory


def play_one(white, black, fen: str, max_plies: int) -> str:
    board = chess.Board(fen)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        bot = white if board.turn == chess.WHITE else black
        mv = bot.choose_move(board)
        if mv not in board.legal_moves:
            mv = next(iter(board.legal_moves))
        board.push(mv)
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return "draw"
    return "white" if o.winner == chess.WHITE else "black"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="tournament/bots.yaml")
    ap.add_argument("--openings", default="tournament/openings.jsonl")
    ap.add_argument("--results", default="tournament/results.jsonl")
    ap.add_argument("--max-openings", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    reg = load_registry(a.registry)
    openings = [json.loads(l) for l in open(a.openings)]
    if a.max_openings:
        openings = openings[:a.max_openings]
    played = set()
    if Path(a.results).exists():
        for l in open(a.results):
            g = json.loads(l); played.add((g["a"], g["b"], g["opening_id"], g["a_color"]))

    ids = [e["id"] for e in reg]
    todo = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        a_id, b_id = sorted([ids[i], ids[j]])
        for op in openings:
            for a_color in ("white", "black"):
                if (a_id, b_id, op["opening_id"], a_color) not in played:
                    todo.append((a_id, b_id, op, a_color))
    print(f"{len(todo)} games to play; {len(played)} already recorded; {len(ids)} bots x {len(openings)} openings", flush=True)
    if not todo:
        return 0

    fac = BotFactory(device=a.device)
    need = {x for t in todo for x in (t[0], t[1])}
    bots = {e["id"]: fac.build(e) for e in reg if e["id"] in need}
    out = open(a.results, "a")
    import time; t0 = time.time()
    for n, (a_id, b_id, op, a_color) in enumerate(todo):
        white_id = a_id if a_color == "white" else b_id
        black_id = b_id if a_color == "white" else a_id
        res = play_one(bots[white_id], bots[black_id], op["fen"], a.max_plies)
        result = "draw" if res == "draw" else ("a" if res == a_color else "b")
        out.write(json.dumps({"a": a_id, "b": b_id, "opening_id": op["opening_id"],
                              "a_color": a_color, "result": result}) + "\n"); out.flush()
        if n and n % 100 == 0:
            print(f"  {n}/{len(todo)}  ({n/(time.time()-t0):.1f} games/s)", flush=True)
    out.close()
    for b in bots.values():
        if hasattr(b, "close"):
            b.close()
    print(f"done: {len(todo)} games in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
