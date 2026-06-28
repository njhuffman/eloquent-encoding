import pytest, chess

pytest.importorskip("maia2")

@pytest.fixture(scope="module")
def maia():
    from style_policy.maia2_bot import load_maia2
    try:
        return load_maia2(device="cpu")   # weights cached at maia2_models/
    except Exception as e:
        pytest.skip(f"maia2 unavailable: {e}")

def test_maia2bot_returns_legal_move(maia):
    from style_policy.maia2_bot import Maia2Bot
    model, prep = maia
    bot = Maia2Bot(model, prep, self_elo=1500, seed=0)
    for fen in [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"]:
        b = chess.Board(fen)
        assert bot.choose_move(b) in b.legal_moves

def test_maia2bot_seed_deterministic(maia):
    from style_policy.maia2_bot import Maia2Bot
    model, prep = maia
    a = Maia2Bot(model, prep, 1500, seed=7).choose_move(chess.Board())
    b = Maia2Bot(model, prep, 1500, seed=7).choose_move(chess.Board())
    assert a == b
