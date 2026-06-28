import os, pytest, chess

CKPT = "style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="wdl_16M checkpoint missing")

@pytest.fixture(scope="module")
def bot0():
    from style_policy.search_bot import ExpectimaxBot
    return ExpectimaxBot(CKPT, elo=1500, depth=0, width=4, device="cpu", seed=0)

def test_policy_topk_shape(bot0):
    from style_policy.search_bot import policy_topk
    b = chess.Board()
    out = policy_topk(bot0.model, b, bot0._elo_idx, 4, "cpu")
    assert 1 <= len(out) <= 4
    moves = [m for m, _ in out]
    assert all(m in b.legal_moves for m in moves)
    probs = [p for _, p in out]
    assert probs == sorted(probs, reverse=True) and all(0 < p <= 1 for p in probs)

def test_depth0_is_policy_argmax(bot0):
    from style_policy.search_bot import policy_topk
    b = chess.Board()
    assert bot0.choose_move(b) == policy_topk(bot0.model, b, bot0._elo_idx, 4, "cpu")[0][0]

def test_depths_return_legal_and_deterministic():
    from style_policy.search_bot import ExpectimaxBot
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    for d in (1, 2):
        m1 = ExpectimaxBot(CKPT, 1500, d, width=4, device="cpu").choose_move(b.copy())
        m2 = ExpectimaxBot(CKPT, 1500, d, width=4, device="cpu").choose_move(b.copy())
        assert m1 in b.legal_moves and m1 == m2  # legal + deterministic
